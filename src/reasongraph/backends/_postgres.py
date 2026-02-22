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
        self, embedding: list[float], top_k: int
    ) -> list[dict[str, str]]:
        pool = await self._get_pool()
        vec_str = f"[{', '.join(map(str, embedding))}]"
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
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
        embedding_weight: float = 0.7,
    ) -> list[dict[str, str]]:
        pool = await self._get_pool()
        vec_str = f"[{', '.join(map(str, embedding))}]"
        keyword_weight = 1.0 - embedding_weight
        async with pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT content, type,
                        {embedding_weight} * (1 - (embedding <=> '{vec_str}'))
                        + {keyword_weight} * similarity(content, %s)
                        AS score
                    FROM nodes
                    ORDER BY score DESC
                    LIMIT {top_k}
                    """,
                    (query_text,),
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

    async def get_all_nodes(self) -> list[Node]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT content, type, embedding, created_at, last_accessed FROM nodes"
                )
                nodes = []
                for content, node_type, emb, created_at, last_accessed in await cur.fetchall():
                    nodes.append(Node(
                        content=content,
                        type=node_type,
                        embedding=list(emb) if emb else None,
                        created_at=created_at,
                        last_accessed=last_accessed,
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
