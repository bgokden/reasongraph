from __future__ import annotations

from abc import ABC, abstractmethod

from reasongraph._types import Node, Edge


class Backend(ABC):
    """Abstract base class for graph storage backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables/schema if they don't exist."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources (connections, pools, etc.)."""

    @abstractmethod
    async def insert_nodes(self, nodes: list[Node]) -> None:
        """Insert or upsert a batch of nodes (must have embeddings set)."""

    @abstractmethod
    async def insert_edges(self, edges: list[Edge]) -> None:
        """Insert a batch of edges, ignoring duplicates."""

    @abstractmethod
    async def knn_search(
        self, embedding: list[float], top_k: int
    ) -> list[dict[str, str]]:
        """Return the top_k closest nodes as dicts with 'content' and 'type' keys."""

    @abstractmethod
    async def get_neighbors(self, content: str) -> list[dict[str, str]]:
        """Return all direct neighbors (both directions) as dicts with 'content' and 'type'."""

    @abstractmethod
    async def delete_stale_nodes(self, days: int) -> int:
        """Delete nodes not accessed within the given number of days. Return count deleted."""

    @abstractmethod
    async def get_all_nodes(self) -> list[Node]:
        """Return every node in the graph."""

    @abstractmethod
    async def hybrid_search(
        self, embedding: list[float], query_text: str, top_k: int,
        embedding_weight: float = 0.7,
    ) -> list[dict[str, str]]:
        """Combined embedding + trigram search.

        Scores = embedding_weight * cosine_sim + (1 - embedding_weight) * trigram_sim.
        Returns top_k nodes as dicts with 'content' and 'type' keys.
        """

    @abstractmethod
    async def get_all_edges(self) -> list[Edge]:
        """Return every edge in the graph."""
