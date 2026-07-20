from __future__ import annotations

import asyncio
import json
from datetime import datetime
from importlib import resources

from reasongraph._embeddings import EmbeddingManager, EmbedderLike
from reasongraph._extraction import NERExtractor, GLiNER2Extractor, ExtractorFn, CausalExtractorFn
from reasongraph._types import Node, Edge
from reasongraph.backends._base import Backend
from reasongraph.backends._memory import MemoryBackend


class ReasonGraph:
    """A graph-based reasoning engine with embedding search and multi-hop traversal.

    Uses an async-first design with sync convenience wrappers.
    Defaults to a pure Python in-memory backend with brute-force cosine similarity.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        embed_model: EmbedderLike = None,
        rerank_model: str | None = None,
        forget_after: int = 30,
        forget_every: float | None = None,
    ) -> None:
        self.backend = backend or MemoryBackend()
        self.embeddings = EmbeddingManager(
            embed_model=embed_model, rerank_model=rerank_model
        )
        self.forget_after = forget_after
        self.forget_every = forget_every
        self._last_forget: datetime | None = None

    # -- Lifecycle --

    async def initialize(self) -> None:
        """Initialize the backend (create tables etc.)."""
        await self.backend.initialize()

    async def close(self) -> None:
        """Close the backend and release resources."""
        await self.backend.close()

    async def __aenter__(self) -> ReasonGraph:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # -- Core operations --

    async def add_nodes(
        self,
        nodes: list[tuple[str, str]],
        scopes: set[str] | list[str] | None = None,
    ) -> None:
        """Add nodes to the graph.

        Args:
            nodes: List of (content, type) tuples. Type is 'text' or 'entity'.
            scopes: Optional free-text scope tags to attach to every node.
                Existing nodes accumulate scopes (union), never lose them.
        """
        scope_set = set(scopes) if scopes else set()
        texts = [content for content, _ in nodes]
        embeddings = self.embeddings.encode_batch(texts)
        node_objs = [
            # Each node gets its own set copy; the backend may merge into it.
            Node(content=content, type=node_type, embedding=emb, scopes=set(scope_set))
            for (content, node_type), emb in zip(nodes, embeddings)
        ]
        await self.backend.insert_nodes(node_objs)

    async def add_edges(self, edges: list[tuple[str, str]]) -> None:
        """Add edges to the graph.

        Args:
            edges: List of (from_content, to_content) tuples.
        """
        edge_objs = [Edge(from_content=f, to_content=t) for f, t in edges]
        await self.backend.insert_edges(edge_objs)

    async def add_text(
        self,
        text: str,
        extractor: ExtractorFn | None = None,
        scopes: set[str] | list[str] | None = None,
    ) -> list[str]:
        """Add text to the graph with automatic entity extraction.

        Creates a text node for the input, extracts entities using the provided
        extractor, creates entity nodes, and links each entity to the text node.

        Args:
            text: The text content to add.
            extractor: A callable(str) -> list[str] that extracts entity strings.
                Defaults to GLiNER2Extractor if gliner2 is installed, otherwise
                falls back to NERExtractor (dslim/bert-base-NER).
            scopes: Optional free-text scope tags for the text and its entities.

        Returns:
            List of extracted entity strings.
        """
        result = await self.add_texts([text], extractor=extractor, scopes=scopes)
        return result[0]

    async def add_texts(
        self,
        texts: list[str],
        extractor: ExtractorFn | None = None,
        causal_extractor: CausalExtractorFn | None = None,
        scopes: set[str] | list[str] | None = None,
    ) -> list[list[str]]:
        """Add multiple texts with automatic entity and causal extraction.

        Processes texts in batch. Entity nodes that appear across multiple
        texts are shared (deduplicated by the backend upsert).

        When a causal_extractor is provided, each text is also analyzed for
        cause-effect relations. Cause and effect spans are added as text
        nodes with edges: cause_span -> original_text and
        effect_span -> original_text, plus cause_span -> effect_span.

        Args:
            texts: List of text strings to add.
            extractor: A callable(str) -> list[str] for entity extraction.
                Defaults to GLiNER2Extractor if gliner2 is installed, otherwise
                falls back to NERExtractor (dslim/bert-base-NER).
            causal_extractor: A callable(list[str]) -> list[dict] for
                cause-effect extraction. Each dict should have 'causal' (bool)
                and 'relations' (list of {'cause': str, 'effect': str}).
                Auto-enabled when the default GLiNER2Extractor is used.

        Returns:
            List of entity lists, one per input text.
        """
        if extractor is None:
            if not hasattr(self, "_default_extractor"):
                try:
                    import gliner2 as _gliner2_check  # noqa: F811
                    self._default_extractor = GLiNER2Extractor()
                except ImportError:
                    self._default_extractor = NERExtractor()
            extractor = self._default_extractor

        if causal_extractor is None and hasattr(extractor, "extract_causal"):
            causal_extractor = extractor.extract_causal

        all_entities = []
        all_nodes = []
        all_edges = []

        # NER entity extraction
        for text in texts:
            entities = extractor(text)
            all_entities.append(entities)
            all_nodes.append((text, "text"))
            for entity in entities:
                all_nodes.append((entity, "entity"))
                all_edges.append((entity, text))

        # Causal relation extraction
        if causal_extractor is not None:
            causal_results = causal_extractor(texts)
            for text, result in zip(texts, causal_results):
                if not result.get("causal"):
                    continue
                for rel in result.get("relations", []):
                    cause = rel.get("cause", "").strip()
                    effect = rel.get("effect", "").strip()
                    if not cause or not effect:
                        continue
                    # Add cause and effect as entity nodes so they serve as
                    # graph connectors but don't appear in query results.
                    all_nodes.append((cause, "entity"))
                    all_nodes.append((effect, "entity"))
                    # cause -> effect (causal link)
                    all_edges.append((cause, effect))
                    # Both link back to the source sentence
                    all_edges.append((cause, text))
                    all_edges.append((effect, text))

        if all_nodes:
            await self.add_nodes(all_nodes, scopes=scopes)
        if all_edges:
            await self.add_edges(all_edges)

        return all_entities

    async def query(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 4,
        rerank_top_k: int = 4,
        search_mode: str = "embedding",
        rrf_k: int = 60,
        recency_weight: float = 0.0,
        scopes: set[str] | list[str] | None = None,
    ) -> list[str]:
        """Query the graph with vector similarity and multi-hop traversal.

        Args:
            query: The search query text.
            top_k: Number of initial seeds from vector search.
            hops: Number of graph traversal hops.
            rerank_top_k: Number of results to keep after reranking at each hop.
            search_mode: 'embedding', 'keyword', or 'hybrid'.
            rrf_k: RRF smoothing constant for hybrid mode (default 60).
            recency_weight: In [0, 1]. When > 0, blends recency (by created_at)
                into reranking so newer facts outrank older contradicting ones.
                0 (default) leaves ranking unchanged.
            scopes: Optional free-text scope tags. When given, the initial
                seeds are drawn only from nodes carrying at least one of these
                scopes; traversal then follows edges across all scopes, so
                reasoning still connects facts beyond the seed scope.

        Returns:
            List of text-type node contents in relevance order.
        """
        if search_mode not in ("embedding", "keyword", "hybrid"):
            raise ValueError(f"search_mode must be 'embedding', 'keyword', or 'hybrid', got '{search_mode}'")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError(f"recency_weight must be in [0, 1], got {recency_weight}")

        scope_set = set(scopes) if scopes else None
        embedding = self.embeddings.encode(query)

        if search_mode == "embedding":
            seeds = await self.backend.knn_search(embedding, top_k, scopes=scope_set)
        elif search_mode == "keyword":
            seeds = await self.backend.hybrid_search(
                embedding, query, top_k, keyword_only=True, scopes=scope_set,
            )
        else:
            seeds = await self.backend.hybrid_search(
                embedding, query, top_k, rrf_k=rrf_k, scopes=scope_set,
            )

        visited: set[str] = set()
        results: list[dict[str, str]] = []

        for _ in range(hops):
            # Separate entity nodes from text nodes.  Entity nodes serve as
            # graph bridges -- we always traverse their edges -- but they
            # should not compete with text nodes for rerank budget.
            text_seeds = [s for s in seeds if s.get("type") == "text"]
            entity_seeds = [s for s in seeds if s.get("type") != "text"]

            # For recency-weighted ranking, attach created_at to the text seeds
            # before reranking (fetched only when the feature is enabled).
            if recency_weight > 0 and text_seeds:
                created = await self.backend.get_created_at(
                    [s["content"] for s in text_seeds]
                )
                for s in text_seeds:
                    s.setdefault("created_at", created.get(s["content"]))

            # Split text seeds by provenance: chain continuations (from
            # text->text edges) get priority access to the rerank budget,
            # bridge discoveries (from entity->text edges) fill remaining
            # slots.  On the first hop there is no provenance tag, so all
            # seeds go into the chain pool (they came from the initial
            # vector search, not from entity bridges).
            chain_pool = [s for s in text_seeds if s.get("_source", "chain") == "chain"]
            bridge_pool = [s for s in text_seeds if s.get("_source") == "bridge"]

            ranked_chain = self.embeddings.rerank(query, chain_pool, rerank_top_k, recency_weight)
            remaining_budget = max(0, rerank_top_k - len(ranked_chain))
            if remaining_budget > 0 and bridge_pool:
                ranked_bridge = self.embeddings.rerank(query, bridge_pool, remaining_budget, recency_weight)
            else:
                ranked_bridge = []
            ranked = ranked_chain + ranked_bridge

            chain_next: list[dict[str, str]] = []
            entity_next: list[dict[str, str]] = []

            for seed in ranked:
                if seed["content"] in visited:
                    continue
                results.append(seed)
                visited.add(seed["content"])
                neighbors = await self.backend.get_neighbors(seed["content"])
                for n in neighbors:
                    if n["content"] not in visited:
                        if n["type"] == "text":
                            n["_source"] = "chain"
                            chain_next.append(n)
                        else:
                            entity_next.append(n)

            # Traverse entity edges transparently (no rerank cost)
            bridge_next: list[dict[str, str]] = []
            for seed in entity_seeds:
                if seed["content"] in visited:
                    continue
                visited.add(seed["content"])
                neighbors = await self.backend.get_neighbors(seed["content"])
                for n in neighbors:
                    if n["content"] not in visited:
                        if n["type"] == "text":
                            n["_source"] = "bridge"
                            bridge_next.append(n)
                        else:
                            entity_next.append(n)

            seeds = chain_next + bridge_next + entity_next

        return [node["content"] for node in results if node["type"] == "text"]

    async def load_dataset(self, name: str) -> None:
        """Load a built-in dataset into the graph.

        Args:
            name: Dataset name (e.g. 'syllogisms', 'causal', 'taxonomy').
        """
        from reasongraph.datasets import load_dataset as _load
        data = _load(name)
        if data["nodes"]:
            await self.add_nodes(
                [(n["content"], n["type"]) for n in data["nodes"]]
            )
        if data["edges"]:
            await self.add_edges(
                [(e["from"], e["to"]) for e in data["edges"]]
            )

    async def delete_stale(self) -> int:
        """Delete nodes not accessed within forget_after days."""
        return await self.backend.delete_stale_nodes(self.forget_after)

    async def maybe_forget(self) -> int:
        """Run ``delete_stale()`` at most once per ``forget_every`` seconds.

        Designed to be called liberally (e.g. after each write or at the end of
        a session), so regular-interval cleanup works without wiring an external
        scheduler. Actual sweeps are throttled by wall-clock time.

        Returns the number of nodes deleted on this call: 0 when ``forget_every``
        is ``None`` (feature disabled), when the throttle interval has not yet
        elapsed, or when nothing was stale.
        """
        if self.forget_every is None:
            return 0
        now = datetime.now()
        if self._last_forget is not None:
            if (now - self._last_forget).total_seconds() < self.forget_every:
                return 0
        self._last_forget = now
        return await self.delete_stale()

    async def delete(self, content: str) -> bool:
        """Delete a single node and its incident edges by exact content.

        Returns True if a node was deleted, False if no node matched. Shared
        entity nodes are not touched; only the named node and the edges
        incident to it are removed.
        """
        return await self.backend.delete_nodes([content]) > 0

    async def supersede(
        self,
        old_content: str,
        new_text: str,
        extractor: ExtractorFn | None = None,
    ) -> list[str]:
        """Replace a stale fact with a corrected one.

        Adds ``new_text`` (with entity extraction) and then deletes the node
        ``old_content``, so the superseded fact can no longer surface in
        queries. The new text is added before the old node is removed, so any
        entity shared between them survives and keeps bridging the graph.

        When ``old_content`` equals ``new_text`` the call is a no-op
        replacement: the fact is (re)added and kept, never deleted.

        Args:
            old_content: Exact content of the node to remove.
            new_text: The corrected text to add in its place.
            extractor: Optional entity extractor for the new text (defaults to
                the same extractor ``add_text`` uses).

        Returns:
            The entities extracted from ``new_text``.
        """
        entities = await self.add_text(new_text, extractor=extractor)
        if old_content != new_text:
            await self.delete(old_content)
        return entities

    async def get_all_nodes(
        self, scopes: set[str] | list[str] | None = None
    ) -> list[Node]:
        """Return all nodes in the graph, or only those in the given scopes."""
        return await self.backend.get_all_nodes(scopes=set(scopes) if scopes else None)

    async def get_all_edges(self) -> list[Edge]:
        """Return all edges in the graph."""
        return await self.backend.get_all_edges()

    # -- Sync convenience wrappers --

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot use sync methods from within a running event loop. "
                "Use the async methods directly instead."
            )
        return asyncio.run(coro)

    def initialize_sync(self) -> None:
        self._run(self.initialize())

    def close_sync(self) -> None:
        self._run(self.close())

    def add_nodes_sync(
        self, nodes: list[tuple[str, str]],
        scopes: set[str] | list[str] | None = None,
    ) -> None:
        self._run(self.add_nodes(nodes, scopes=scopes))

    def add_edges_sync(self, edges: list[tuple[str, str]]) -> None:
        self._run(self.add_edges(edges))

    def add_text_sync(
        self, text: str, extractor: ExtractorFn | None = None,
        scopes: set[str] | list[str] | None = None,
    ) -> list[str]:
        return self._run(self.add_text(text, extractor, scopes))

    def add_texts_sync(
        self,
        texts: list[str],
        extractor: ExtractorFn | None = None,
        causal_extractor: CausalExtractorFn | None = None,
        scopes: set[str] | list[str] | None = None,
    ) -> list[list[str]]:
        return self._run(self.add_texts(texts, extractor, causal_extractor, scopes))

    def query_sync(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 4,
        rerank_top_k: int = 4,
        search_mode: str = "embedding",
        rrf_k: int = 60,
        recency_weight: float = 0.0,
        scopes: set[str] | list[str] | None = None,
    ) -> list[str]:
        return self._run(self.query(
            query, top_k, hops, rerank_top_k, search_mode, rrf_k, recency_weight, scopes,
        ))

    def load_dataset_sync(self, name: str) -> None:
        self._run(self.load_dataset(name))

    def delete_stale_sync(self) -> int:
        return self._run(self.delete_stale())

    def maybe_forget_sync(self) -> int:
        return self._run(self.maybe_forget())

    def delete_sync(self, content: str) -> bool:
        return self._run(self.delete(content))

    def supersede_sync(
        self,
        old_content: str,
        new_text: str,
        extractor: ExtractorFn | None = None,
    ) -> list[str]:
        return self._run(self.supersede(old_content, new_text, extractor))
