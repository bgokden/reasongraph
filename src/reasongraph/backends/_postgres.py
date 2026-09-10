from __future__ import annotations

import os

import logging

import warnings
from datetime import datetime, timedelta

from reasongraph._types import Node, Edge
from reasongraph.backends._base import Backend

try:
    from psycopg_pool import AsyncConnectionPool
    from pgvector.psycopg import register_vector_async

    _HAS_POSTGRES = True
except ImportError:
    _HAS_POSTGRES = False


def _as_list(emb) -> list[float]:
    """pgvector returns a ``Vector`` (not iterable) when its adapters are registered
    on the connection, and a plain list/array otherwise. Normalise both."""
    to_list = getattr(emb, "to_list", None)
    if callable(to_list):
        return list(to_list())
    tolist = getattr(emb, "tolist", None)
    if callable(tolist):
        return list(tolist())
    return list(emb)


logger = logging.getLogger(__name__)

# One well-known key for the schema-creation advisory lock (any constant; must match across processes).
_SCHEMA_LOCK = 8145270113


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
        # Repeatable vector search, off by default: see _tune().
        self._deterministic = os.environ.get("REASONGRAPH_PG_DETERMINISTIC", "").lower() in ("1", "true", "yes")
        self._ef_search = os.environ.get("REASONGRAPH_PG_EF_SEARCH", "").strip()

    async def _tune(self, conn) -> None:
        """Per-connection settings for vector search, when repeatability across REBUILDS is wanted.

        Measured at 100k facts: a running service is already repeatable without any of this. The same
        question returns the same facts every time from a fixed index, and the order within a result is
        fixed for free. What drifts is a rebuild: two indexes built independently over identical data
        return slightly different neighbours, and that can change a root cause, not only filler.
        Forcing agreement across rebuilds means exact search, which cost about 3.6x on recall in that
        measurement, so it belongs to a migration and not to serving. Enable with
        ``REASONGRAPH_PG_DETERMINISTIC=1``; ``REASONGRAPH_PG_EF_SEARCH`` widens the candidate window.
        """
        if not self._deterministic:
            return
        # Exact search: the secondary sort key defeats the approximate index on purpose, which is
        # what makes two independent builds agree. Slow, deliberately, and only for a migration.
        await conn.execute("SET enable_indexscan = off")
        await conn.execute("SET max_parallel_workers_per_gather = 0")
        if self._ef_search:
            await conn.execute(f"SET hnsw.ef_search = {int(self._ef_search)}")

    async def _get_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return self._pool

    async def initialize(self) -> None:
        self._pool = AsyncConnectionPool(self.database_url, open=False)
        await self._pool.open()

        async with self._pool.connection() as conn:
            # Several processes (API + extraction workers) may start together; concurrent
            # CREATE TABLE/INDEX on the same objects deadlocks in Postgres. One advisory lock
            # serialises the whole schema step; the rest is IF NOT EXISTS, so the losers no-op.
            await conn.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_LOCK,))
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

            # Approximate-nearest-neighbour index for the cosine (<=>) KNN in
            # knn_search/hybrid_search. Without it, KNN is a full sequential scan
            # whose latency grows linearly with the graph. HNSW (pgvector >= 0.5.0)
            # suits an incrementally written table; on older servers we skip the
            # index (correct, just slower) rather than fail initialization.
            await self._create_vector_index(conn)
            await self._create_trigram_index(conn)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    from_content TEXT NOT NULL REFERENCES nodes(content) ON DELETE CASCADE,
                    to_content TEXT NOT NULL REFERENCES nodes(content) ON DELETE CASCADE,
                    last_accessed TIMESTAMP DEFAULT NOW(),
                    label TEXT,
                    UNIQUE(from_content, to_content)
                )
            """)
            # Migrate pre-existing DBs created before the typed-edge column.
            await conn.execute("ALTER TABLE edges ADD COLUMN IF NOT EXISTS label TEXT")
            # Temporal validity: retirement time for soft-superseded facts.
            await conn.execute(
                "ALTER TABLE nodes ADD COLUMN IF NOT EXISTS invalid_at TIMESTAMP"
            )

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS nodes_entity_lower_idx ON nodes (LOWER(content) text_pattern_ops) WHERE type = 'entity'"
            )
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
            await conn.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_LOCK,))

    async def _create_trigram_index(self, conn) -> None:
        """pg_trgm plus a GIN index so the lexical channel of hybrid_search is index-assisted
        (word-similarity operators). Without the extension the channel degrades to a scan."""
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS nodes_content_trgm_idx "
                "ON nodes USING gin (content gin_trgm_ops)"
            )
            self._trigram = True
        except Exception as exc:  # pragma: no cover - depends on the server
            logger.warning("pg_trgm unavailable, hybrid search will scan: %s", exc)
            self._trigram = False

    @staticmethod
    def _supports_hnsw(version: str | None) -> bool:
        """HNSW was added in pgvector 0.5.0."""
        if not version:
            return False
        try:
            parts = tuple(int(p) for p in version.split(".")[:2])
        except ValueError:
            return False
        return parts >= (0, 5)

    async def _create_vector_index(self, conn) -> None:
        row = await (await conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )).fetchone()
        version = row[0] if row else None
        if not self._supports_hnsw(version):
            warnings.warn(
                f"pgvector {version or 'unknown'} lacks HNSW (needs >= 0.5.0); "
                "vector KNN will use a sequential scan. Upgrade pgvector for "
                "index-accelerated search.",
                stacklevel=2,
            )
            return
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS nodes_embedding_hnsw "
            "ON nodes USING hnsw (embedding vector_cosine_ops)"
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


    # Transient write failures under concurrency: a deadlock or serialization failure, and the
    # pipeline abort psycopg raises for the rest of an executemany batch once one statement failed.
    # They mean "try again", not "this write is wrong", so retry the whole operation briefly.
    _RETRY_SQLSTATES = ("40001", "40P01", "25P02", "55P03")

    async def _with_retry(self, op, attempts: int = 4):
        import asyncio as _asyncio
        import random as _random

        for attempt in range(attempts):
            try:
                return await op()
            except Exception as exc:
                state = getattr(exc, "sqlstate", None) or getattr(getattr(exc, "diag", None), "sqlstate", None)
                transient = state in self._RETRY_SQLSTATES or "pipeline" in str(exc).lower()
                if not transient or attempt == attempts - 1:
                    raise
                await _asyncio.sleep((0.05 * 2 ** attempt) * (1 + _random.random()))

    async def insert_nodes(self, nodes: list[Node]) -> None:
        return await self._with_retry(lambda: self._insert_nodes(nodes))

    async def _insert_nodes(self, nodes: list[Node]) -> None:
        pool = await self._get_pool()
        now = datetime.now()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Sorted by primary key: two writers upserting overlapping entity nodes take the
                # row locks in the same order, so they queue instead of deadlocking.
                data = [
                    (node.content, node.embedding, node.created_at, now, node.type)
                    for node in sorted(nodes, key=lambda n: n.content)
                ]
                await cur.executemany(
                    """
                    INSERT INTO nodes (content, embedding, created_at, last_accessed, type)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (content) DO UPDATE
                        SET last_accessed = NOW(), invalid_at = NULL
                    """,
                    data,
                )
                scope_rows = sorted(
                    (node.content, scope) for node in nodes for scope in node.scopes
                )
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
        return await self._with_retry(lambda: self._insert_edges(edges))

    async def _insert_edges(self, edges: list[Edge]) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                data = sorted(
                    (e.from_content, e.to_content, e.label) for e in edges
                )   # same ordering rule as insert_nodes: no lock-order deadlock between writers
                await cur.executemany(
                    """
                    INSERT INTO edges (from_content, to_content, label)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (from_content, to_content)
                    DO UPDATE SET label = COALESCE(EXCLUDED.label, edges.label)
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
            await self._tune(conn)
            async with conn.cursor() as cur:
                if scopes:
                    await cur.execute(
                        f"""
                        SELECT content, type, score FROM (
                            SELECT content, type, 1 - (embedding <=> '{vec_str}') AS score,
                                   embedding <=> '{vec_str}' AS dist
                            FROM nodes
                            WHERE content IN (
                                SELECT node_content FROM node_scopes WHERE scope = ANY(%s)
                            )
                            ORDER BY embedding <=> '{vec_str}'
                            LIMIT {top_k}
                        ) t ORDER BY dist, content
                        """,
                        (list(scopes),),
                    )
                else:
                    await cur.execute(
                        f"""
                        SELECT content, type, score FROM (
                            SELECT content, type, 1 - (embedding <=> '{vec_str}') AS score,
                                   embedding <=> '{vec_str}' AS dist
                            FROM nodes
                            ORDER BY embedding <=> '{vec_str}'
                            LIMIT {top_k}
                        ) t ORDER BY dist, content
                        """
                    )
                rows = await cur.fetchall()
                # "score" is the cosine similarity the index already computed, so
                # callers (dedup, conflict candidates) need not re-encode the hits.
                return [{"content": row[0], "type": row[1], "score": float(row[2])} for row in rows]

    async def hybrid_search(
        self, embedding: list[float], query_text: str, top_k: int,
        rrf_k: int = 60, keyword_only: bool = False,
        scopes: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Seeds by cosine and by words. The lexical channel uses pg_trgm's strict word
        similarity (a query word matching a word in the fact: names, codes, numbers, compounds),
        index-assisted, ranked; both channels are windowed and fused by reciprocal rank so a fact
        found by one channel only still counts."""
        pool = await self._get_pool()
        vec_str = f"[{', '.join(map(str, embedding))}]"
        window = max(top_k * 10, 100)
        scope_sql = (
            "AND content IN (SELECT node_content FROM node_scopes WHERE scope = ANY(%s))"
            if scopes else ""
        )
        scope_param = [list(scopes)] if scopes else []
        async with pool.connection() as conn:
            await self._tune(conn)
            if getattr(self, "_trigram", None) is None:
                await self._create_trigram_index(conn)
            async with conn.cursor() as cur:
                # a query word counts as matching a fact word above this trigram similarity
                await cur.execute("SET pg_trgm.strict_word_similarity_threshold = 0.45")
                kw_sql = f"""
                    SELECT content, type,
                           ROW_NUMBER() OVER (ORDER BY strict_word_similarity(%s, content) DESC) AS rank
                    FROM nodes
                    WHERE type = 'text' AND %s <<%% content {scope_sql}
                    LIMIT %s
                """
                kw_params = (query_text, query_text, *scope_param, window)
                if keyword_only:
                    await cur.execute(kw_sql, kw_params)
                    rows = await cur.fetchall()
                    if not rows:   # nothing above the threshold: fall back to whole-string similarity
                        await cur.execute(
                            f"SELECT content, type FROM nodes WHERE TRUE {scope_sql} "
                            "ORDER BY similarity(content, %s) DESC LIMIT %s",
                            (*scope_param, query_text, top_k),
                        )
                        rows = await cur.fetchall()
                    return [{"content": r[0], "type": r[1]} for r in rows[:top_k]]

                await cur.execute(
                    f"""
                    WITH emb_ranked AS (
                        SELECT content, type,
                               ROW_NUMBER() OVER (ORDER BY embedding <=> '{vec_str}') AS rank
                        FROM nodes
                        WHERE TRUE {scope_sql}
                        ORDER BY embedding <=> '{vec_str}'
                        LIMIT %s
                    ),
                    kw_ranked AS ({kw_sql})
                    SELECT COALESCE(e.content, k.content) AS content,
                           COALESCE(e.type, k.type) AS type,
                           1.0 / (%s + COALESCE(e.rank, %s)) + 1.0 / (%s + COALESCE(k.rank, %s)) AS score
                    FROM emb_ranked e
                    FULL OUTER JOIN kw_ranked k ON e.content = k.content
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (*scope_param, window, *kw_params, rrf_k, window + 1, rrf_k, window + 1, top_k),
                )
                return [{"content": row[0], "type": row[1]} for row in await cur.fetchall()]

    async def get_neighbors(
        self, content: str, scopes: set[str] | None = None
    ) -> list[dict[str, str]]:
        pool = await self._get_pool()
        params: list = [content, content, content]
        scope_clause = ""
        if scopes:
            scope_clause = (
                " AND n.content IN "
                "(SELECT node_content FROM node_scopes WHERE scope = ANY(%s))"
            )
            params.append(sorted(scopes))
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT n.content, n.type, e.label,
                           CASE WHEN e.from_content = %s THEN 'out' ELSE 'in' END AS direction
                    FROM nodes n
                    INNER JOIN edges e ON (e.to_content = n.content AND e.from_content = %s)
                                       OR (e.from_content = n.content AND e.to_content = %s)
                    WHERE 1=1{scope_clause}
                    """,
                    params,
                )
                # Dedup by neighbor, preferring a labeled edge so causal links surface.
                neighbors: dict[str, dict[str, str]] = {}
                for c, node_type, label, direction in await cur.fetchall():
                    existing = neighbors.get(c)
                    if existing is None or (label is not None and existing["label"] is None):
                        neighbors[c] = {
                            "content": c, "type": node_type,
                            "label": label, "direction": direction,
                        }
                return list(neighbors.values())

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

    async def set_invalid(self, contents: list[str], when: datetime) -> None:
        if not contents:
            return
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE nodes SET invalid_at = %s WHERE content = ANY(%s)",
                    (when, contents),
                )

    async def get_validity(self, contents: list[str]) -> dict[str, str | None]:
        if not contents:
            return {}
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT content, invalid_at FROM nodes WHERE content = ANY(%s)",
                    (contents,),
                )
                return {
                    row[0]: (row[1].isoformat() if row[1] else None)
                    for row in await cur.fetchall()
                }

    async def list_scopes(self, prefix: str | None = None) -> list[str]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if prefix:
                    pat = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                    await cur.execute("SELECT DISTINCT scope FROM node_scopes WHERE scope LIKE %s ORDER BY scope", (pat,))
                else:
                    await cur.execute("SELECT DISTINCT scope FROM node_scopes ORDER BY scope")
                return [r[0] for r in await cur.fetchall()]

    async def count_nodes(self, node_type: str | None = None, scopes: set[str] | None = None) -> int:
        pool = await self._get_pool()
        where, params = [], []
        if node_type:
            where.append("type = %s"); params.append(node_type)
        if scopes:
            where.append("content IN (SELECT node_content FROM node_scopes WHERE scope = ANY(%s))"); params.append(sorted(scopes))
        sql = "SELECT COUNT(*) FROM nodes" + (" WHERE " + " AND ".join(where) if where else "")
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                return int((await cur.fetchone())[0])

    async def get_node_types(self, contents: list[str]) -> dict[str, str]:
        if not contents:
            return {}
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT content, type FROM nodes WHERE content = ANY(%s)", (contents,))
                return {c: t for c, t in await cur.fetchall()}

    async def nearest_neighbors(self, content: str, query_embedding, limit: int,
                                scopes: set[str] | None = None) -> list[dict[str, str]]:
        """The ``limit`` neighbours nearest to the query, ordered by vector distance.

        Measured at 100k facts on the 24,524-neighbour hub: this OR-join form gives
        discover p95 235 ms; a UNION ALL CTE over the two edge indexes (0.7.24) was
        3.8x slower because the CTE materialised and lost the join order. Keep this.
        """
        pool = await self._get_pool()
        params: list = [content, content, content]
        scope_clause = ""
        if scopes:
            scope_clause = " AND n.content IN (SELECT node_content FROM node_scopes WHERE scope = ANY(%s))"
            params.append(sorted(scopes))
        params += [list(map(float, query_embedding)), int(limit)]
        async with pool.connection() as conn:
            await self._tune(conn)
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT n.content, n.type, e.label,
                           CASE WHEN e.from_content = %s THEN 'out' ELSE 'in' END AS direction
                    FROM nodes n
                    INNER JOIN edges e ON (e.to_content = n.content AND e.from_content = %s)
                                       OR (e.from_content = n.content AND e.to_content = %s)
                    WHERE 1=1{scope_clause}
                    ORDER BY n.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    params,
                )
                neighbors: dict[str, dict[str, str]] = {}
                for c, node_type, label, direction in await cur.fetchall():
                    existing = neighbors.get(c)
                    if existing is None or (label is not None and existing["label"] is None):
                        neighbors[c] = {"content": c, "type": node_type, "label": label, "direction": direction}
                return list(neighbors.values())

    async def nearest_neighbors_many(self, contents: list[str], query_embedding, limit: int,
                                     scopes: set[str] | None = None) -> dict[str, list[dict[str, str]]]:
        """One walk level in one round trip. A LATERAL keeps the per-source OR-join and its
        per-source LIMIT, so each source still gets its own index-driven top-``limit`` (the plan
        W29 measured); only the number of round trips changes."""
        if not contents:
            return {}
        pool = await self._get_pool()
        scope_clause = ""
        params: list = [list(dict.fromkeys(contents))]
        if scopes:
            scope_clause = " AND n.content IN (SELECT node_content FROM node_scopes WHERE scope = ANY(%s))"
        vec = list(map(float, query_embedding))
        async with pool.connection() as conn:
            await self._tune(conn)
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT s.src, x.content, x.type, x.label,
                           CASE WHEN x.from_content = s.src THEN 'out' ELSE 'in' END AS direction
                    FROM unnest(%s::text[]) AS s(src)
                    CROSS JOIN LATERAL (
                        SELECT n.content, n.type, e.label, e.from_content
                        FROM nodes n
                        INNER JOIN edges e ON (e.to_content = n.content AND e.from_content = s.src)
                                           OR (e.from_content = n.content AND e.to_content = s.src)
                        WHERE 1=1{scope_clause}
                        ORDER BY n.embedding <=> %s::vector
                        LIMIT %s
                    ) AS x
                    """,
                    (*params, *([sorted(scopes)] if scopes else []), vec, int(limit)),
                )
                out: dict[str, dict[str, dict[str, str]]] = {c: {} for c in contents}
                for src, c, node_type, label, direction in await cur.fetchall():
                    bucket = out.setdefault(src, {})
                    existing = bucket.get(c)
                    if existing is None or (label is not None and existing["label"] is None):
                        bucket[c] = {"content": c, "type": node_type, "label": label, "direction": direction}
                return {src: list(v.values()) for src, v in out.items()}

    async def count_edges(self) -> int:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM edges")
                return int((await cur.fetchone())[0])

    async def entities_starting_with(self, word: str, limit: int = 20) -> list[str]:
        pool = await self._get_pool()
        w = word.lower()
        pattern = w.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + " %"
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT content FROM nodes WHERE type = 'entity' AND (LOWER(content) = %s OR LOWER(content) LIKE %s) LIMIT %s",
                    (w, pattern, limit),
                )
                return [row[0] for row in await cur.fetchall()]

    async def nodes_in_scopes(self, scopes: set[str]) -> list[str]:
        scopes = list(scopes)
        if not scopes:
            return []
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT DISTINCT node_content FROM node_scopes WHERE scope = ANY(%s)", (scopes,)
                )
                return [row[0] for row in await cur.fetchall()]

    async def remove_scopes(self, contents: list[str], scopes: set[str]) -> int:
        scopes = list(scopes)
        if not contents or not scopes:
            return 0
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM node_scopes WHERE node_content = ANY(%s) AND scope = ANY(%s)",
                    (contents, scopes),
                )
                return cur.rowcount or 0

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

    async def get_causal_relations(self, contents: list[str]) -> dict[str, list[dict]]:
        if not contents:
            return {}
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT DISTINCT c2.to_content AS fact,
                           ce.from_content AS cause, ce.to_content AS effect
                    FROM edges ce
                    JOIN edges c2 ON c2.from_content = ce.from_content
                    JOIN edges e2 ON e2.from_content = ce.to_content
                                 AND e2.to_content = c2.to_content
                    WHERE ce.label = 'causes' AND c2.to_content = ANY(%s)
                    """,
                    (list(contents),),
                )
                result: dict[str, list[dict]] = {}
                for fact, cause, effect in await cur.fetchall():
                    result.setdefault(fact, []).append({"cause": cause, "effect": effect})
                return result

    async def get_all_nodes(self, scopes: set[str] | None = None) -> list[Node]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if scopes:
                    await cur.execute(
                        """
                        SELECT content, type, embedding, created_at, last_accessed, invalid_at
                        FROM nodes
                        WHERE content IN (
                            SELECT node_content FROM node_scopes WHERE scope = ANY(%s)
                        )
                        """,
                        (list(scopes),),
                    )
                else:
                    await cur.execute(
                        "SELECT content, type, embedding, created_at, last_accessed, "
                        "invalid_at FROM nodes"
                    )
                rows = await cur.fetchall()

                await cur.execute("SELECT node_content, scope FROM node_scopes")
                scope_map: dict[str, set[str]] = {}
                for node_content, scope in await cur.fetchall():
                    scope_map.setdefault(node_content, set()).add(scope)

                nodes = []
                for content, node_type, emb, created_at, last_accessed, invalid_at in rows:
                    nodes.append(Node(
                        content=content,
                        type=node_type,
                        embedding=_as_list(emb) if emb is not None else None,
                        created_at=created_at,
                        last_accessed=last_accessed,
                        scopes=scope_map.get(content, set()),
                        invalid_at=invalid_at,
                    ))
                return nodes

    async def get_all_edges(self) -> list[Edge]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT from_content, to_content, last_accessed, label FROM edges"
                )
                return [
                    Edge(from_content=row[0], to_content=row[1], last_accessed=row[2], label=row[3])
                    for row in await cur.fetchall()
                ]
