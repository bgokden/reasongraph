from __future__ import annotations

from abc import ABC, abstractmethod

from reasongraph._types import Node, Edge


class Backend(ABC):
    """Abstract base class for graph storage backends.

    Scopes are free-text tags on nodes, not partitions. The graph is shared:
    edges and traversal cross scopes freely. A ``scopes`` filter on the
    seed-producing searches (``knn_search`` / ``hybrid_search``) narrows only
    which nodes a query starts from -- ``None`` means search all nodes.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables/schema if they don't exist."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources (connections, pools, etc.)."""

    @abstractmethod
    async def insert_nodes(self, nodes: list[Node]) -> None:
        """Insert or upsert a batch of nodes (must have embeddings set).

        On upsert of an existing node, its scopes are unioned with the incoming
        node's scopes (adding the same content under a new scope tags it).
        """

    @abstractmethod
    async def insert_edges(self, edges: list[Edge]) -> None:
        """Insert a batch of edges, ignoring duplicates."""

    @abstractmethod
    async def knn_search(
        self, embedding: list[float], top_k: int,
        scopes: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Return the top_k closest nodes as dicts with 'content' and 'type' keys.

        When ``scopes`` is given, only nodes carrying at least one of those
        scopes are eligible.
        """

    @abstractmethod
    async def get_neighbors(
        self, content: str, scopes: set[str] | None = None
    ) -> list[dict[str, str]]:
        """Return all direct neighbors (both directions) as dicts with 'content' and 'type'.

        When ``scopes`` is given, only neighbors carrying at least one of those
        scopes are returned. This is opt-in traversal isolation: passing the query
        scope confines a walk to one tenant instead of crossing scopes freely.
        """

    @abstractmethod
    async def delete_stale_nodes(self, days: int) -> int:
        """Delete nodes not accessed within the given number of days. Return count deleted."""

    async def list_scopes(self, prefix: str | None = None) -> list[str]:
        """Distinct scope tags (optionally starting with ``prefix``), without loading nodes."""
        raise NotImplementedError

    async def count_nodes(self, node_type: str | None = None, scopes: set[str] | None = None) -> int:
        """Number of nodes, optionally of one type and/or carrying one of ``scopes``."""
        raise NotImplementedError

    async def get_node_types(self, contents: list[str]) -> dict[str, str]:
        """``content -> type`` for the given nodes (missing nodes are absent)."""
        raise NotImplementedError

    async def nearest_neighbors(self, content: str, query_embedding, limit: int,
                                scopes: set[str] | None = None) -> list[dict[str, str]]:
        """Like :meth:`get_neighbors`, but at most ``limit`` neighbours, the ones nearest
        to ``query_embedding``. Hub entities ("Apple" on ten thousand facts) must not
        turn a walk into a scan. Default: fetch all and rank in Python."""
        neighbors = await self.get_neighbors(content, scopes)
        if len(neighbors) <= limit:
            return neighbors
        return await self._rank_neighbors(neighbors, query_embedding, limit)

    async def _rank_neighbors(self, neighbors, query_embedding, limit):
        raise NotImplementedError

    async def count_edges(self) -> int:
        """Number of edges, without loading them."""
        raise NotImplementedError

    async def entities_starting_with(self, word: str, limit: int = 20) -> list[str]:
        """Entity nodes whose first word is ``word`` (case-insensitive), at most ``limit``."""
        raise NotImplementedError

    async def nodes_in_scopes(self, scopes: set[str]) -> list[str]:
        """Contents of every node carrying at least one of ``scopes``."""
        raise NotImplementedError

    async def remove_scopes(self, contents: list[str], scopes: set[str]) -> int:
        """Drop ``scopes`` from the given nodes (the nodes themselves stay). Return rows removed."""
        raise NotImplementedError

    @abstractmethod
    async def delete_nodes(self, contents: list[str]) -> int:
        """Delete the given nodes and their incident edges. Return count deleted."""

    @abstractmethod
    async def get_created_at(self, contents: list[str]) -> dict[str, str]:
        """Return {content: created_at ISO string} for the given contents.

        Missing contents are omitted. Used for recency-weighted ranking.
        """

    @abstractmethod
    async def get_scopes(self, contents: list[str]) -> dict[str, set[str]]:
        """Return {content: scope set} for the given contents.

        Contents with no scopes (or not present) are omitted -- callers treat a
        missing key as the empty set. This is the bounded alternative to
        ``get_all_nodes`` when only the scopes of a known set of nodes are
        needed (e.g. tagging discovered facts), so it scales independently of
        the total graph size.
        """

    @abstractmethod
    async def set_invalid(self, contents: list[str], when: datetime) -> None:
        """Stamp the given facts as retired at ``when`` (soft-supersede).

        The nodes stay in the graph; ``invalid_at`` marks them as no longer current.
        """

    @abstractmethod
    async def get_validity(self, contents: list[str]) -> dict[str, str | None]:
        """Return {content: invalid_at ISO string or None} for the given contents.

        Missing contents are omitted. A present key with ``None`` means still valid;
        an ISO string is the retirement time. Bounded lookup (like get_created_at),
        used to exclude retired facts and for time-travel queries.
        """

    @abstractmethod
    async def get_causal_relations(self, contents: list[str]) -> dict[str, list[dict]]:
        """Return {fact content: [{'cause','effect'}, ...]} for the given facts.

        A fact's causal relations are the directed ``"causes"`` edges (cause ->
        effect span) where both spans link back to that fact. Facts with none are
        omitted. Bounded by ``contents``, so it scales with the result set rather
        than the whole graph.
        """

    @abstractmethod
    async def get_all_nodes(self, scopes: set[str] | None = None) -> list[Node]:
        """Return every node in the graph, or only those in the given scopes."""

    @abstractmethod
    async def hybrid_search(
        self, embedding: list[float], query_text: str, top_k: int,
        rrf_k: int = 60, keyword_only: bool = False,
        scopes: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Combined embedding + trigram search using Reciprocal Rank Fusion.

        Each node is ranked independently by cosine similarity and by trigram
        similarity.  The final score is:
            rrf_score(d) = 1/(rrf_k + rank_emb(d)) + 1/(rrf_k + rank_kw(d))

        When keyword_only=True, nodes are ranked by trigram similarity alone
        (no embedding component).

        Args:
            embedding: Query embedding vector.
            query_text: Raw query string for trigram matching.
            top_k: Number of results to return.
            rrf_k: RRF smoothing constant (default 60).
            keyword_only: If True, rank by trigram similarity only.
            scopes: If given, only nodes carrying at least one of those scopes
                are eligible.

        Returns:
            Top-k nodes as dicts with 'content' and 'type' keys.
        """

    @abstractmethod
    async def get_all_edges(self) -> list[Edge]:
        """Return every edge in the graph."""
