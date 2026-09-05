"""Transport-agnostic core of the agent-memory service.

Wraps a single shared ``ReasonGraph`` (models loaded once and kept warm, the
backend persisting the shared graph). Knowledge sessions are scope tags: a push
is stored under its session; a query seeds from a session but traversal crosses
all sessions, so agents discover connections beyond their own memory.
"""

from __future__ import annotations

import asyncio
import logging

from reasongraph import ReasonGraph
from reasongraph._extraction import ExtractorFn

logger = logging.getLogger(__name__)


class MemoryService:
    """Async agent-memory + discovery service over one shared ReasonGraph.

    Args:
        graph: A ready ReasonGraph, or None to build one from the other args.
        backend/embed_model/rerank_model/synthesizer: forwarded to ReasonGraph
            when ``graph`` is None. Use a persistent backend (e.g. PostgresBackend)
            for a real multi-agent service.
        extractor: Optional shared entity extractor (kept warm). Defaults to the
            graph's default (gliner_small-v2.5).
        canonicalizer: Optional entity canonicalizer (callable or alias Mapping)
            forwarded to ReasonGraph when ``graph`` is None. Applied to every write
            path, so surface variants of an entity collapse to one bridge across
            sessions -- the light stand-in for coreference.
    """

    def __init__(
        self,
        graph: ReasonGraph | None = None,
        *,
        backend=None,
        embed_model=None,
        rerank_model=None,
        synthesizer=None,
        extractor: ExtractorFn | None = None,
        causal_extractor=None,
        canonicalizer=None,
        defer_extraction: bool = False,
        dedup_threshold: float | None = None,
    ) -> None:
        self.graph = graph or ReasonGraph(
            backend=backend, embed_model=embed_model,
            rerank_model=rerank_model, synthesizer=synthesizer,
            causal_extractor=causal_extractor, canonicalizer=canonicalizer,
        )
        self.extractor = extractor
        # Writes run a (sync, CPU-bound) extraction model; serialize them so
        # concurrent agent pushes don't contend on it. Reads stay concurrent.
        self._write_lock = asyncio.Lock()
        # Deferred extraction: when on, a push stores the fact immediately (one
        # embedding) and a background worker runs the heavy entity/causal
        # extraction afterward, so the write path isn't gated on the model.
        self.defer_extraction = defer_extraction
        # Semantic dedup on write: a push whose embedding is at least this similar
        # to an existing fact unions its scopes onto that fact instead of adding a
        # paraphrase. None = off (exact duplicates still merge via upsert).
        self.dedup_threshold = dedup_threshold
        self._enrich_queue: asyncio.Queue | None = None
        self._enrich_worker_task: asyncio.Task | None = None
        self._enrich_inflight = 0   # items dequeued but not yet finished

    async def initialize(self) -> None:
        await self.graph.initialize()
        if self.defer_extraction:
            self._enrich_queue = asyncio.Queue()
            self._enrich_worker_task = asyncio.create_task(self._enrich_worker())

    async def close(self) -> None:
        if self._enrich_worker_task is not None:
            if self._enrich_queue is not None:
                await self._enrich_queue.join()  # finish pending enrichment first
            self._enrich_worker_task.cancel()
            try:
                await self._enrich_worker_task
            except asyncio.CancelledError:
                pass
            self._enrich_worker_task = None
        await self.graph.close()

    async def _enrich_worker(self) -> None:
        """Drain the enrichment queue, running full extraction on each deferred fact."""
        assert self._enrich_queue is not None
        while True:
            session, text, *rest = await self._enrich_queue.get()
            self._enrich_inflight += 1
            resolve = rest[0] if rest else None
            # A queued item carries either one session name or a list of scope
            # tags (callers that tag a fact with several scopes at once).
            scopes = [session] if isinstance(session, str) else list(session)
            try:
                # The models are synchronous CPU work; run them in a worker thread
                # so the event loop keeps serving reads while a fact is enriched.
                entities, causal_results, causal_fn = await self._extract_off_loop(text)
                async with self._write_lock:
                    await self.graph.add_texts(
                        [text], extractor=lambda _t: entities, scopes=scopes,
                        causal_extractor=(lambda _ts: causal_results) if causal_fn else None,
                        causal=None if causal_fn else False,
                        resolve_conflicts=resolve,
                    )
            except Exception:  # a bad fact must not kill the worker
                logger.exception("deferred extraction failed for a memory")
            finally:
                self._enrich_inflight -= 1
                self._enrich_queue.task_done()

    def _entity_extractor(self):
        if self.extractor is not None:
            return self.extractor
        g = self.graph
        if not hasattr(g, "_default_extractor"):
            g._default_extractor = g._build_default_extractor()
        return g._default_extractor

    async def _extract_off_loop(self, text: str):
        """Run entity + causal extraction for one text in a thread. Returns
        ``(entities, causal_results, causal_fn)``; ``causal_fn`` is None when
        causal extraction is disabled or unavailable."""
        ext = self._entity_extractor()
        causal_fn = None
        if self.graph._causal_extractor_arg is not False:
            causal_fn = getattr(ext, "extract_causal", None) or self.graph._resolve_causal_extractor()
        entities = await asyncio.to_thread(ext, text)
        causal_results = await asyncio.to_thread(causal_fn, [text]) if causal_fn else None
        return entities, causal_results, causal_fn

    async def _dedup(self, text: str, scopes: list[str]) -> bool:
        """True when ``text`` near-duplicates an existing fact (scopes were unioned
        onto it); the caller must then skip adding/enriching it."""
        if self.dedup_threshold is None:
            return False
        dup = await self.graph._find_duplicate(text, self.dedup_threshold)
        if dup is None:
            return False
        if scopes:
            await self.graph.add_nodes([(dup, "text")], scopes=scopes)
        return True

    async def __aenter__(self) -> "MemoryService":
        await self.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- write path (an agent pushes memory) --

    async def push(self, session: str, text: str, *, resolve_conflicts: bool | None = None) -> dict:
        """Store one memory in a session; returns the extracted entities.

        With ``defer_extraction`` the fact is stored immediately and its entities
        are extracted in the background, so ``entities`` is empty and ``deferred``
        is True; the fact is queryable at once, its bridges appear shortly after.
        """
        if self._enrich_queue is not None:
            async with self._write_lock:
                if await self._dedup(text, [session]):
                    return {"session": session, "entities": [], "deferred": False, "duplicate": True}
                await self.graph.add_text(
                    text, extractor=lambda _t: [], causal=False, scopes=[session]
                )
            await self._enrich_queue.put((session, text, resolve_conflicts))
            return {"session": session, "entities": [], "deferred": True}
        async with self._write_lock:
            entities = await self.graph.add_text(
                text, extractor=self.extractor, scopes=[session],
                dedup_threshold=self.dedup_threshold, resolve_conflicts=resolve_conflicts,
            )
        return {"session": session, "entities": entities}

    async def push_many(self, session: str, texts: list[str], *,
                        resolve_conflicts: bool | None = None) -> dict:
        if self._enrich_queue is not None:
            fresh: list[str] = []
            async with self._write_lock:
                for text in texts:
                    if await self._dedup(text, [session]):
                        continue
                    await self.graph.add_text(
                        text, extractor=lambda _t: [], causal=False, scopes=[session]
                    )
                    fresh.append(text)
            for text in fresh:
                await self._enrich_queue.put((session, text, resolve_conflicts))
            return {"session": session, "count": len(texts), "deferred": True,
                    "entities": [[] for _ in texts], "duplicates": len(texts) - len(fresh)}
        async with self._write_lock:
            per_text = await self.graph.add_texts(
                texts, extractor=self.extractor, scopes=[session],
                dedup_threshold=self.dedup_threshold, resolve_conflicts=resolve_conflicts,
            )
        return {"session": session, "count": len(texts), "entities": per_text}

    async def supersede(
        self, session: str, old_text: str, new_text: str,
        purge_orphans: bool = False,
    ) -> dict:
        """Replace a stale fact with a corrected one in a session.

        With ``purge_orphans`` the old fact's now-unreferenced entities are removed
        too (complete erasure); shared entities survive.
        """
        async with self._write_lock:
            await self.graph.add_text(new_text, extractor=self.extractor, scopes=[session])
            deleted = False
            if old_text != new_text:
                deleted = await self.graph.delete(old_text, purge_orphans=purge_orphans)
        return {"superseded": deleted, "new": new_text}

    async def delete(self, text: str, purge_orphans: bool = False) -> dict:
        """Delete a fact by exact content so an agent can self-correct memory.

        With ``purge_orphans`` the fact's now-unreferenced entities are removed
        too (right-to-be-forgotten); entities still bridging other facts survive.
        """
        async with self._write_lock:
            deleted = await self.graph.delete(text, purge_orphans=purge_orphans)
        return {"deleted": deleted}

    async def history(self, text: str) -> dict:
        """Supersession audit for a fact: what it replaced and what replaced it."""
        return await self.graph.supersession_history(text)

    async def trace(
        self, content: str, *, direction: str = "effects", session: str | None = None,
        max_depth: int = 6, isolate: bool | None = None,
    ) -> dict:
        """Walk the causal graph from ``content``. ``direction='effects'`` traces
        downstream impact; ``'causes'`` traces back to root causes."""
        method = self.graph.trace_causes if direction == "causes" else self.graph.trace_effects
        return await method(
            content, scopes=self._scopes(session), max_depth=max_depth, isolate=isolate,
        )

    async def what_if(
        self, content: str, *, origin: str | None = None, direction: str = "effects",
        session: str | None = None, max_depth: int = 6, isolate: bool | None = None,
    ) -> dict:
        """Counterfactual: if ``content`` were false, which downstream effects would
        collapse (lose all causal support) vs survive via an alternate path."""
        return await self.graph.what_if(
            content, origin=origin, direction=direction, scopes=self._scopes(session),
            max_depth=max_depth, isolate=isolate,
        )

    async def forget(self) -> dict:
        """Drop facts not accessed within the graph's forget window."""
        async with self._write_lock:
            return {"deleted": await self.graph.delete_stale()}

    async def maybe_forget(self) -> dict:
        """Throttled forget sweep (per the graph's ``forget_every``). Safe to call
        often; the graph rate-limits actual deletion. Used by the app scheduler."""
        async with self._write_lock:
            return {"deleted": await self.graph.maybe_forget()}

    # -- read path (query with reasoning, discover connections) --

    @staticmethod
    def _scopes(session: str | None) -> list[str] | None:
        return [session] if session else None

    async def query(
        self, query: str, *, session: str | None = None,
        hops: int = 4, top_k: int = 5, search_mode: str = "embedding",
        recency_weight: float = 0.0, isolate: bool | None = None,
        detailed: bool = False, as_of=None, include_superseded: bool = False,
        walk_scopes=None,
    ) -> list:
        """Ranked facts. Seeds from ``session`` (or everywhere if None). Traversal
        crosses sessions by default; ``isolate=True`` confines it to ``session``
        (multi-tenant). ``recency_weight`` (0-1) favours newer facts. With
        ``detailed`` each result is a dict with score/created_at/scopes.
        ``as_of`` (datetime) time-travels to what was current then;
        ``include_superseded`` also returns retired facts."""
        method = self.graph.query_detailed if detailed else self.graph.query
        return await method(
            query, scopes=self._scopes(session), hops=hops, top_k=top_k,
            search_mode=search_mode, recency_weight=recency_weight, isolate=isolate,
            as_of=as_of, include_superseded=include_superseded, walk_scopes=walk_scopes,
        )

    async def causal_chain(
        self, from_content: str, to_content: str, *, session: str | None = None,
        max_depth: int = 6, isolate: bool | None = None, walk_scopes=None,
    ) -> dict:
        """Directed causal path from the fact nearest ``from_content`` to the one
        nearest ``to_content`` (or ``{"chain": None}`` when none exists)."""
        chain = await self.graph.causal_chain(
            from_content, to_content, scopes=self._scopes(session),
            max_depth=max_depth, isolate=isolate, walk_scopes=walk_scopes,
        )
        return {"from": from_content, "to": to_content, "chain": chain}

    async def discover(
        self, query: str, *, session: str | None = None,
        hops: int = 4, top_k: int = 5, max_results: int = 10,
        search_mode: str = "embedding", isolate: bool | None = None,
        include_superseded: bool = False, walk_scopes=None,
    ) -> list[dict]:
        """Connection paths: how each reached fact links back to a seed, with
        cross-session discoveries flagged. ``isolate=True`` confines the walk to
        ``session``."""
        return await self.graph.discover(
            query, scopes=self._scopes(session), hops=hops, top_k=top_k,
            max_results=max_results, search_mode=search_mode, isolate=isolate,
            include_superseded=include_superseded, walk_scopes=walk_scopes,
        )

    async def answer(
        self, query: str, *, session: str | None = None,
        use_discover: bool = True, hops: int = 4, top_k: int = 5,
    ) -> str:
        """Logical free-text answer via the configured synthesizer."""
        return await self.graph.answer(
            query, scopes=self._scopes(session), use_discover=use_discover,
            hops=hops, top_k=top_k,
        )

    # -- introspection --

    async def list_sessions(self) -> list[str]:
        scopes: set[str] = set()
        for node in await self.graph.get_all_nodes():
            scopes |= node.scopes
        return sorted(scopes)

    async def stats(self) -> dict:
        nodes = await self.graph.get_all_nodes()
        edges = await self.graph.get_all_edges()
        scopes: set[str] = set()
        for n in nodes:
            scopes |= n.scopes
        return {
            "facts": sum(1 for n in nodes if n.type == "text"),
            "entities": sum(1 for n in nodes if n.type == "entity"),
            "edges": len(edges),
            "sessions": len(scopes),
            # Facts still waiting for deferred entity/causal extraction. 0 means
            # every bridge is in place; clients can poll this after a push.
            "pending": self.pending_extractions,
        }

    @property
    def pending_extractions(self) -> int:
        """Facts queued *or in flight* for deferred extraction (0 when extraction is
        synchronous). Reaches 0 only after the last fact's extraction and any
        contradiction check have completed, so clients can poll it safely."""
        if self._enrich_queue is None:
            return 0
        return self._enrich_queue.qsize() + self._enrich_inflight
