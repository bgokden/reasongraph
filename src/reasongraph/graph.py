from __future__ import annotations

import asyncio
import json
from importlib import resources

from reasongraph._embeddings import EmbeddingManager
from reasongraph._extraction import NERExtractor, GLiNER2Extractor, ExtractorFn, CausalExtractorFn
from reasongraph._types import Node, Edge
from reasongraph.backends._base import Backend
from reasongraph.backends._sqlite import SqliteBackend


class ReasonGraph:
    """A graph-based reasoning engine with embedding search and multi-hop traversal.

    Uses an async-first design with sync convenience wrappers.
    Defaults to an in-memory SQLite backend with brute-force cosine similarity.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        embed_model: str | None = None,
        rerank_model: str | None = None,
        forget_after: int = 30,
    ) -> None:
        self.backend = backend or SqliteBackend()
        self.embeddings = EmbeddingManager(
            embed_model=embed_model, rerank_model=rerank_model
        )
        self.forget_after = forget_after

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

    async def add_nodes(self, nodes: list[tuple[str, str]]) -> None:
        """Add nodes to the graph.

        Args:
            nodes: List of (content, type) tuples. Type is 'text' or 'entity'.
        """
        texts = [content for content, _ in nodes]
        embeddings = self.embeddings.encode_batch(texts)
        node_objs = [
            Node(content=content, type=node_type, embedding=emb)
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
    ) -> list[str]:
        """Add text to the graph with automatic entity extraction.

        Creates a text node for the input, extracts entities using the provided
        extractor (or the default HF NER model), creates entity nodes, and
        links each entity to the text node.

        Args:
            text: The text content to add.
            extractor: A callable(str) -> list[str] that extracts entity strings.
                Defaults to the built-in HF NER extractor (dslim/bert-base-NER).

        Returns:
            List of extracted entity strings.
        """
        result = await self.add_texts([text], extractor=extractor)
        return result[0]

    async def add_texts(
        self,
        texts: list[str],
        extractor: ExtractorFn | None = None,
        causal_extractor: CausalExtractorFn | None = None,
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
                Defaults to built-in NER (dslim/bert-base-NER).
            causal_extractor: A callable(list[str]) -> list[dict] for
                cause-effect extraction. Each dict should have 'causal' (bool)
                and 'relations' (list of {'cause': str, 'effect': str}).
                Pass CausalExtractor() to use SocioCausaNet.

        Returns:
            List of entity lists, one per input text.
        """
        if extractor is None:
            if not hasattr(self, "_default_extractor"):
                self._default_extractor = NERExtractor()
            extractor = self._default_extractor

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
            await self.add_nodes(all_nodes)
        if all_edges:
            await self.add_edges(all_edges)

        return all_entities

    async def query(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 2,
        rerank_top_k: int = 3,
        search_mode: str = "embedding",
        embedding_weight: float = 0.7,
    ) -> list[str]:
        """Query the graph with vector similarity and multi-hop traversal.

        Args:
            query: The search query text.
            top_k: Number of initial seeds from vector search.
            hops: Number of graph traversal hops.
            rerank_top_k: Number of results to keep after reranking at each hop.
            search_mode: 'embedding', 'keyword', or 'hybrid'.
            embedding_weight: Weight for embedding score in hybrid mode (0-1).
                The keyword (trigram) weight is 1 - embedding_weight.

        Returns:
            List of text-type node contents in relevance order.
        """
        if search_mode not in ("embedding", "keyword", "hybrid"):
            raise ValueError(f"search_mode must be 'embedding', 'keyword', or 'hybrid', got '{search_mode}'")

        embedding = self.embeddings.encode(query)

        if search_mode == "embedding":
            seeds = await self.backend.knn_search(embedding, top_k)
        elif search_mode == "keyword":
            seeds = await self.backend.hybrid_search(
                embedding, query, top_k, embedding_weight=0.0,
            )
        else:
            seeds = await self.backend.hybrid_search(
                embedding, query, top_k, embedding_weight=embedding_weight,
            )

        visited: set[str] = set()
        results: list[dict[str, str]] = []

        for _ in range(hops):
            # Separate entity nodes from text nodes.  Entity nodes serve as
            # graph bridges -- we always traverse their edges -- but they
            # should not compete with text nodes for rerank budget.
            text_seeds = [s for s in seeds if s.get("type") == "text"]
            entity_seeds = [s for s in seeds if s.get("type") != "text"]

            ranked = self.embeddings.rerank(query, text_seeds, rerank_top_k)
            next_seeds: list[dict[str, str]] = []

            for seed in ranked:
                if seed["content"] in visited:
                    continue
                results.append(seed)
                visited.add(seed["content"])
                neighbors = await self.backend.get_neighbors(seed["content"])
                next_seeds.extend(neighbors)

            # Traverse entity edges transparently (no rerank cost)
            for seed in entity_seeds:
                if seed["content"] in visited:
                    continue
                visited.add(seed["content"])
                neighbors = await self.backend.get_neighbors(seed["content"])
                next_seeds.extend(neighbors)

            seeds = next_seeds

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

    async def get_all_nodes(self) -> list[Node]:
        """Return all nodes in the graph."""
        return await self.backend.get_all_nodes()

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

    def add_nodes_sync(self, nodes: list[tuple[str, str]]) -> None:
        self._run(self.add_nodes(nodes))

    def add_edges_sync(self, edges: list[tuple[str, str]]) -> None:
        self._run(self.add_edges(edges))

    def add_text_sync(self, text: str, extractor: ExtractorFn | None = None) -> list[str]:
        return self._run(self.add_text(text, extractor))

    def add_texts_sync(
        self,
        texts: list[str],
        extractor: ExtractorFn | None = None,
        causal_extractor: CausalExtractorFn | None = None,
    ) -> list[list[str]]:
        return self._run(self.add_texts(texts, extractor, causal_extractor))

    def query_sync(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 2,
        rerank_top_k: int = 3,
        search_mode: str = "embedding",
        embedding_weight: float = 0.7,
    ) -> list[str]:
        return self._run(self.query(
            query, top_k, hops, rerank_top_k, search_mode, embedding_weight,
        ))

    def load_dataset_sync(self, name: str) -> None:
        self._run(self.load_dataset(name))

    def delete_stale_sync(self) -> int:
        return self._run(self.delete_stale())
