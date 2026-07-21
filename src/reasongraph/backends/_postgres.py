from __future__ import annotations

from datetime import datetime, timedelta

from reasongraph._types import Node, Edge
from reasongraph.backends._base import Backend

try:
    from psycopg_pool import AsyncConnectionPool
    from pgvector.psycopg import register_vector_async

    _HAS_POSTGRES = True
except ImportError:
    _HAS_POSTGRES = False


class PostgresBackend(Backend):
    """PostgreSQL + pgvector backend for scalable vector search.

    Requires: pip install reasongraph[postgres]
    """

    def __init__(self, database_url: str) -> None:
        if not _HAS_POSTGRES:
            raise ImportError(
                "PostgreSQL dependencies not installed. "
                "Install with: pip install reasongraph[postgres]"
            )
        self.database_url = database_url
        self._pool: AsyncConnectionPool | None = None

    async def _get_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return self._pool

    async def initialize(self) -> None:
        self._pool = AsyncConnectionPool(self.database_url, open=False)
        await self._pool.open()

        async with self._pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector_async(conn)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    content TEXT PRIMARY KEY,
                    embedding VECTOR(384) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_accessed TIMESTAMP DEFAULT NOW(),
                    type TEXT NOT NULL CHECK (type IN ('text', 'entity'))
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    from_content TEXT NOT NULL REFERENCES nodes(content) ON DELETE CASCADE,
                    to_content TEXT NOT NULL REFERENCES nodes(content) ON DELETE CASCADE,
                    last_accessed TIMESTAMP DEFAULT NOW(),
                    UNIQUE(from_content, to_content)
                )
            """)

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS edges_from_idx ON edges (from_content)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS edges_to_idx ON edges (to_content)"
            )

            # Free-text scope tags (many-to-many). Not partitions: used only to
            # narrow query seeds; the graph and traversal stay shared.
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS node_scopes (
                    node_content TEXT NOT NULL REFERENCES nodes(content) ON DELETE CASCADE,
                    scope TEXT NOT NULL,
                    PRIMARY KEY (node_content, scope)
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS node_scopes_scope_idx ON node_scopes (scope)"
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def insert_nodes(self, nodes: list[Node]) -> None:
        pool = await self._get_pool()
        now = datetime.now()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                data = [
                    (node.content, node.embedding, node.created_at, now, node.type)
                    for node in nodes
                ]
                await cur.executemany(
                    """
                    INSERT INTO nodes (content, embedding, created_at, last_accessed, type)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (content) DO UPDATE SET last_accessed = NOW()
                    """,
                    data,
                )
                scope_rows = [
                    (node.content, scope) for node in nodes for scope in node.scopes
                ]
                if scope_rows:
                    await cur.executemany(
                        """
                        INSERT INTO node_scopes (node_content, scope)
                        VALUES (%s, %s)
                        ON CONFLICT (node_content, scope) DO NOTHING
                        """,
                        scope_rows,
                    )

    async def insert_edges(self, edges: list[Edge]) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                data = [(e.from_content, e.to_content) for e in edges]
                await cur.executemany(
                    """
                    INSERT INTO edges (from_content, to_content)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    data,
                )

    async def knn_search(
        self, embedding: list[float], top_k: int,
        scopes: set[str] | None = None,
    ) -> list[dict[str, str]]:
        pool = await self._get_pool()
        vec_str = f"[{', '.join(map(str, embedding))}]"
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if scopes:
                    await cur.execute(
                        f"""
                        SELECT content, type
                        FROM nodes
                        WHERE content IN (
                            SELECT node_content FROM node_scopes WHERE scope = ANY(%s)
                        )
                        ORDER BY embedding <=> '{vec_str}'
                        LIMIT {top_k}
                        """,
                        (list(scopes),),
                    )
                else:
                    await cur.execute(
                        f"""
                        SELECT content, type
                        FROM nodes
                        ORDER BY embedding <=> '{vec_str}'
                        LIMIT {top_k}
                        """
                    )
                rows = await cur.fetchall()
                return [{"content": row[0], "type": row[1]} for row in rows]

    async def hybrid_search(
        self, embedding: list[float], query_text: str, top_k: int,
        rrf_k: int = 60, keyword_only: bool = False,
        scopes: set[str] | None = None,
    ) -> list[dict[str, str]]:
        pool = await self._get_pool()
        vec_str = f"[{', '.join(map(str, embedding))}]"
        # Optional "restrict to nodes in these scopes" clause + its parameter.
        scope_sql = (
            "WHERE content IN (SELECT node_content FROM node_scopes WHERE scope = ANY(%s))"
            if scopes else ""
        )
        scope_param = [list(scopes)] if scopes else []
        async with pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            async with conn.cursor() as cur:
                if keyword_only:
                    await cur.execute(
                        f"""
                        SELECT content, type
                        FROM nodes
                        {scope_sql}
                        ORDER BY similarity(content, %s) DESC
                        LIMIT %s
                        """,
                        (*scope_param, query_text, top_k),
                    )
                    return [
                        {"content": row[0], "type": row[1]}
                        for row in await cur.fetchall()
                    ]

                # RRF entirely in SQL using window functions
                await cur.execute(
                    f"""
                    WITH emb_ranked AS (
                        SELECT content, type,
                            ROW_NUMBER() OVER (
                                ORDER BY embedding <=> '{vec_str}'
                            ) AS rank
                        FROM nodes
                        {scope_sql}
                    ),
                    kw_ranked AS (
                        SELECT content,
                            ROW_NUMBER() OVER (
                                ORDER BY similarity(content, %s) DESC
                            ) AS rank
                        FROM nodes
                        {scope_sql}
                    )
                    SELECT e.content, e.type
                    FROM emb_ranked e
                    JOIN kw_ranked k ON e.content = k.content
                    ORDER BY 1.0 / (%s + e.rank) + 1.0 / (%s + k.rank) DESC
                    LIMIT %s
                    """,
                    # Placeholder order: emb-CTE scope, kw similarity, kw-CTE
                    # scope, then the two rrf_k and the limit.
                    (*scope_param, query_text, *scope_param, rrf_k, rrf_k, top_k),
                )
                return [
                    {"content": row[0], "type": row[1]}
                    for row in await cur.fetchall()
                ]

    async def get_neighbors(self, content: str) -> list[dict[str, str]]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT DISTINCT n.content, n.type FROM nodes n
                    INNER JOIN edges e ON (e.to_content = n.content AND e.from_content = %s)
                                       OR (e.from_content = n.content AND e.to_content = %s)
                    """,
                    (content, content),
                )
                return [
                    {"content": row[0], "type": row[1]}
                    for row in await cur.fetchall()
                ]

    async def delete_stale_nodes(self, days: int) -> int:
        pool = await self._get_pool()
        cutoff = datetime.now() - timedelta(days=days)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM nodes WHERE last_accessed < %s", (cutoff,)
                )
                return cur.rowcount

    async def delete_nodes(self, contents: list[str]) -> int:
        if not contents:
            return 0
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # CASCADE on the edges foreign key removes incident edges
                await cur.execute(
                    "DELETE FROM nodes WHERE content = ANY(%s)", (contents,)
                )
                return cur.rowcount

    async def get_created_at(self, contents: list[str]) -> dict[str, str]:
        if not contents:
            return {}
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT content, created_at FROM nodes WHERE content = ANY(%s)",
                    (contents,),
                )
                return {
                    row[0]: row[1].isoformat() for row in await cur.fetchall()
                }

    async def get_scopes(self, contents: list[str]) -> dict[str, set[str]]:
        if not contents:
            return {}
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT node_content, scope FROM node_scopes WHERE node_content = ANY(%s)",
                    (contents,),
                )
                result: dict[str, set[str]] = {}
                for node_content, scope in await cur.fetchall():
                    result.setdefault(node_content, set()).add(scope)
                return result

    async def get_all_nodes(self, scopes: set[str] | None = None) -> list[Node]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if scopes:
                    await cur.execute(
                        """
                        SELECT content, type, embedding, created_at, last_accessed
                        FROM nodes
                        WHERE content IN (
                            SELECT node_content FROM node_scopes WHERE scope = ANY(%s)
                        )
                        """,
                        (list(scopes),),
                    )
                else:
                    await cur.execute(
                        "SELECT content, type, embedding, created_at, last_accessed FROM nodes"
                    )
                rows = await cur.fetchall()

                await cur.execute("SELECT node_content, scope FROM node_scopes")
                scope_map: dict[str, set[str]] = {}
                for node_content, scope in await cur.fetchall():
                    scope_map.setdefault(node_content, set()).add(scope)

                nodes = []
                for content, node_type, emb, created_at, last_accessed in rows:
                    nodes.append(Node(
                        content=content,
                        type=node_type,
                        embedding=list(emb) if emb is not None else None,
                        created_at=created_at,
                        last_accessed=last_accessed,
                        scopes=scope_map.get(content, set()),
                    ))
                return nodes

    async def get_all_edges(self) -> list[Edge]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT from_content, to_content, last_accessed FROM edges"
                )
                return [
                    Edge(from_content=row[0], to_content=row[1], last_accessed=row[2])
                    for row in await cur.fetchall()
                ]
