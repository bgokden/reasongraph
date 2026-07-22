"""Transport-agnostic core of the agent-memory service.

Wraps a single shared ``ReasonGraph`` (models loaded once and kept warm, the
backend persisting the shared graph). Knowledge sessions are scope tags: a push
is stored under its session; a query seeds from a session but traversal crosses
all sessions, so agents discover connections beyond their own memory.
"""

from __future__ import annotations

import asyncio

from reasongraph import ReasonGraph
from reasongraph._extraction import ExtractorFn


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

    async def initialize(self) -> None:
        await self.graph.initialize()

    async def close(self) -> None:
        await self.graph.close()

    async def __aenter__(self) -> "MemoryService":
        await self.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- write path (an agent pushes memory) --

    async def push(self, session: str, text: str) -> dict:
        """Store one memory in a session; returns the extracted entities."""
        async with self._write_lock:
            entities = await self.graph.add_text(
                text, extractor=self.extractor, scopes=[session]
            )
        return {"session": session, "entities": entities}

    async def push_many(self, session: str, texts: list[str]) -> dict:
        async with self._write_lock:
            per_text = await self.graph.add_texts(
                texts, extractor=self.extractor, scopes=[session]
            )
        return {"session": session, "count": len(texts), "entities": per_text}

    async def supersede(self, session: str, old_text: str, new_text: str) -> dict:
        """Replace a stale fact with a corrected one in a session."""
        async with self._write_lock:
            await self.graph.add_text(new_text, extractor=self.extractor, scopes=[session])
            deleted = False
            if old_text != new_text:
                deleted = await self.graph.delete(old_text)
        return {"superseded": deleted, "new": new_text}

    async def forget(self) -> dict:
        """Drop facts not accessed within the graph's forget window."""
        async with self._write_lock:
            return {"deleted": await self.graph.delete_stale()}

    # -- read path (query with reasoning, discover connections) --

    @staticmethod
    def _scopes(session: str | None) -> list[str] | None:
        return [session] if session else None

    async def query(
        self, query: str, *, session: str | None = None,
        hops: int = 4, top_k: int = 5, search_mode: str = "embedding",
    ) -> list[str]:
        """Ranked facts. Seeds from ``session`` (or everywhere if None); the
        multi-hop traversal crosses sessions."""
        return await self.graph.query(
            query, scopes=self._scopes(session), hops=hops, top_k=top_k,
            search_mode=search_mode,
        )

    async def discover(
        self, query: str, *, session: str | None = None,
        hops: int = 4, top_k: int = 5, max_results: int = 10,
    ) -> list[dict]:
        """Connection paths: how each reached fact links back to a seed, with
        cross-session discoveries flagged."""
        return await self.graph.discover(
            query, scopes=self._scopes(session), hops=hops, top_k=top_k,
            max_results=max_results,
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
