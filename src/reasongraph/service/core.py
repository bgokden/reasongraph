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
        defer_extraction: bool = False,
    ) -> None:
        self.graph = graph or ReasonGraph(
            backend=backend, embed_model=embed_model,
            rerank_model=rerank_model, synthesizer=synthesizer,
            causal_extractor=causal_extractor,
        )
        self.extractor = extractor
        # Writes run a (sync, CPU-bound) extraction model; serialize them so
        # concurrent agent pushes don't contend on it. Reads stay concurrent.
        self._write_lock = asyncio.Lock()
        # Deferred extraction: when on, a push stores the fact immediately (one
        # embedding) and a background worker runs the heavy entity/causal
        # extraction afterward, so the write path isn't gated on the model.
        self.defer_extraction = defer_extraction
        self._enrich_queue: asyncio.Queue | None = None
        self._enrich_worker_task: asyncio.Task | None = None

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
            session, text = await self._enrich_queue.get()
            try:
                async with self._write_lock:
                    await self.graph.add_texts(
                        [text], extractor=self.extractor, scopes=[session]
                    )
            except Exception:  # a bad fact must not kill the worker
                logger.exception("deferred extraction failed for a memory")
            finally:
                self._enrich_queue.task_done()

    async def __aenter__(self) -> "MemoryService":
        await self.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- write path (an agent pushes memory) --

    async def push(self, session: str, text: str) -> dict:
        """Store one memory in a session; returns the extracted entities.

        With ``defer_extraction`` the fact is stored immediately and its entities
        are extracted in the background, so ``entities`` is empty and ``deferred``
        is True; the fact is queryable at once, its bridges appear shortly after.
        """
        if self._enrich_queue is not None:
            async with self._write_lock:
                await self.graph.add_text(
                    text, extractor=lambda _t: [], causal=False, scopes=[session]
                )
            await self._enrich_queue.put((session, text))
            return {"session": session, "entities": [], "deferred": True}
        async with self._write_lock:
            entities = await self.graph.add_text(
                text, extractor=self.extractor, scopes=[session]
            )
        return {"session": session, "entities": entities}

    async def push_many(self, session: str, texts: list[str]) -> dict:
        if self._enrich_queue is not None:
            async with self._write_lock:
                for text in texts:
                    await self.graph.add_text(
                        text, extractor=lambda _t: [], causal=False, scopes=[session]
                    )
            for text in texts:
                await self._enrich_queue.put((session, text))
            return {"session": session, "count": len(texts),
                    "entities": [[] for _ in texts], "deferred": True}
        async with self._write_lock:
            per_text = await self.graph.add_texts(
                texts, extractor=self.extractor, scopes=[session]
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
        detailed: bool = False,
    ) -> list:
        """Ranked facts. Seeds from ``session`` (or everywhere if None). Traversal
        crosses sessions by default; ``isolate=True`` confines it to ``session``
        (multi-tenant). ``recency_weight`` (0-1) favours newer facts. With
        ``detailed`` each result is a dict with score/created_at/scopes."""
        method = self.graph.query_detailed if detailed else self.graph.query
        return await method(
            query, scopes=self._scopes(session), hops=hops, top_k=top_k,
            search_mode=search_mode, recency_weight=recency_weight, isolate=isolate,
        )

    async def discover(
        self, query: str, *, session: str | None = None,
        hops: int = 4, top_k: int = 5, max_results: int = 10,
        search_mode: str = "embedding", isolate: bool | None = None,
    ) -> list[dict]:
        """Connection paths: how each reached fact links back to a seed, with
        cross-session discoveries flagged. ``isolate=True`` confines the walk to
        ``session``."""
        return await self.graph.discover(
            query, scopes=self._scopes(session), hops=hops, top_k=top_k,
            max_results=max_results, search_mode=search_mode, isolate=isolate,
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
        }
