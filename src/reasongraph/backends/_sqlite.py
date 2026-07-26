from __future__ import annotations

import struct
from datetime import datetime, timedelta

import aiosqlite
import sqlite_vec

from reasongraph._types import Node, Edge
from reasongraph.backends._base import Backend


def _embedding_to_blob(embedding: list[float]) -> bytes:
    """Pack a float list into a compact binary blob (same format sqlite-vec uses)."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _blob_to_list(blob: bytes) -> list[float]:
    """Unpack a binary blob back to a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _escape_fts5(query: str) -> str:
    """Escape a query string for safe use in FTS5 MATCH.

    Wraps the query in double quotes so FTS5 treats it as a literal phrase,
    and escapes any internal double quotes by doubling them.
    """
    return '"' + query.replace('"', '""') + '"'


class SqliteBackend(Backend):
    """SQLite backend with sqlite-vec for vector search and FTS5 trigram for text search."""

    def __init__(self, db_path: str = ":memory:", embedding_dim: int = 384) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._db: aiosqlite.Connection | None = None

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return self._db

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)

        # Load sqlite-vec extension
        await self._db.enable_load_extension(True)
        await self._db.load_extension(sqlite_vec.loadable_path())
        await self._db.enable_load_extension(False)

        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                type TEXT NOT NULL CHECK (type IN ('text', 'entity'))
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                from_content TEXT NOT NULL REFERENCES nodes(content) ON DELETE CASCADE,
                to_content TEXT NOT NULL REFERENCES nodes(content) ON DELETE CASCADE,
                last_accessed TEXT NOT NULL,
                label TEXT,
                UNIQUE(from_content, to_content)
            )
        """)

        # Migrate pre-existing DBs that were created before the typed-edge column.
        cursor = await self._db.execute("PRAGMA table_info(edges)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "label" not in columns:
            await self._db.execute("ALTER TABLE edges ADD COLUMN label TEXT")

        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS edges_from_idx ON edges (from_content)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS edges_to_idx ON edges (to_content)"
        )

        # vec0 virtual table for cosine-distance vector search
        await self._db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes
            USING vec0(node_id INTEGER PRIMARY KEY, embedding float[{self.embedding_dim}] distance_metric=cosine)
        """)

        # FTS5 with trigram tokenizer for substring text search
        await self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_nodes
            USING fts5(content, tokenize='trigram')
        """)

        # Free-text scope tags (many-to-many). Not partitions: used only to
        # narrow query seeds; the graph and traversal stay shared.
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS node_scopes (
                node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                scope TEXT NOT NULL,
                PRIMARY KEY (node_id, scope)
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS node_scopes_scope_idx ON node_scopes (scope)"
        )

        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def insert_nodes(self, nodes: list[Node]) -> None:
        db = await self._conn()
        now = datetime.now().isoformat()

        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"Node '{node.content}' has no embedding")

        # Deduplicate within the batch (keep first occurrence). Scopes are
        # accumulated in a separate dict so the caller's Node objects are never
        # mutated.
        seen: dict[str, Node] = {}
        merged_scopes: dict[str, set[str]] = {}
        unique_nodes: list[Node] = []
        for node in nodes:
            if node.content in seen:
                merged_scopes[node.content] |= node.scopes
            else:
                seen[node.content] = node
                merged_scopes[node.content] = set(node.scopes)
                unique_nodes.append(node)

        # Map existing contents to their node ids (needed to attach scopes)
        contents = [node.content for node in unique_nodes]
        placeholders = ",".join("?" for _ in contents)
        cursor = await db.execute(
            f"SELECT content, id FROM nodes WHERE content IN ({placeholders})",
            contents,
        )
        existing = {row[0]: row[1] for row in await cursor.fetchall()}

        for node in unique_nodes:
            if node.content in existing:
                # Existing node: bump last_accessed and union in any new scopes
                await db.execute(
                    "UPDATE nodes SET last_accessed = ? WHERE content = ?",
                    (now, node.content),
                )
                await self._insert_scopes(db, existing[node.content], merged_scopes[node.content])
            else:
                # New node: insert into all three tables plus its scopes
                blob = _embedding_to_blob(node.embedding)
                cursor = await db.execute(
                    """
                    INSERT INTO nodes (content, embedding, created_at, last_accessed, type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (node.content, blob, node.created_at.isoformat(), now, node.type),
                )
                row_id = cursor.lastrowid
                await db.execute(
                    "INSERT INTO vec_nodes (node_id, embedding) VALUES (?, ?)",
                    (row_id, blob),
                )
                await db.execute(
                    "INSERT INTO fts_nodes (rowid, content) VALUES (?, ?)",
                    (row_id, node.content),
                )
                await self._insert_scopes(db, row_id, merged_scopes[node.content])

        await db.commit()

    @staticmethod
    async def _insert_scopes(db, node_id: int, scopes: set[str]) -> None:
        if not scopes:
            return
        await db.executemany(
            "INSERT OR IGNORE INTO node_scopes (node_id, scope) VALUES (?, ?)",
            [(node_id, scope) for scope in scopes],
        )

    async def insert_edges(self, edges: list[Edge]) -> None:
        db = await self._conn()
        now = datetime.now().isoformat()
        rows = [(e.from_content, e.to_content, now, e.label) for e in edges]
        await db.executemany(
            """
            INSERT INTO edges (from_content, to_content, last_accessed, label)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(from_content, to_content)
            DO UPDATE SET label = COALESCE(excluded.label, edges.label)
            """,
            rows,
        )
        await db.commit()

    async def knn_search(
        self, embedding: list[float], top_k: int,
        scopes: set[str] | None = None,
    ) -> list[dict[str, str]]:
        db = await self._conn()
        query_blob = _embedding_to_blob(embedding)

        if scopes:
            # Exact KNN restricted to the scope subset via the vec_distance_cosine
            # scalar (the vec0 index can't be filtered by a many-to-many tag).
            scope_ph = ",".join("?" for _ in scopes)
            cursor = await db.execute(
                f"""
                SELECT n.content, n.type
                FROM nodes n
                WHERE n.id IN (SELECT node_id FROM node_scopes WHERE scope IN ({scope_ph}))
                ORDER BY vec_distance_cosine(n.embedding, ?)
                LIMIT ?
                """,
                [*scopes, query_blob, top_k],
            )
        else:
            cursor = await db.execute(
                """
                SELECT n.content, n.type
                FROM vec_nodes v
                JOIN nodes n ON n.id = v.node_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (query_blob, top_k),
            )
        rows = await cursor.fetchall()

        now = datetime.now().isoformat()
        results = []
        for content, node_type in rows:
            await db.execute(
                "UPDATE nodes SET last_accessed = ? WHERE content = ?",
                (now, content),
            )
            results.append({"content": content, "type": node_type})
        await db.commit()
        return results

    async def hybrid_search(
        self, embedding: list[float], query_text: str, top_k: int,
        rrf_k: int = 60, keyword_only: bool = False,
        scopes: set[str] | None = None,
    ) -> list[dict[str, str]]:
        db = await self._conn()
        short_query = len(query_text) < 3
        # Optional "restrict to nodes in these scopes" clause + its parameters.
        scope_ph = ",".join("?" for _ in scopes) if scopes else ""
        scope_ids_sql = (
            f"n.id IN (SELECT node_id FROM node_scopes WHERE scope IN ({scope_ph}))"
            if scopes else "1=1"
        )
        scope_params = list(scopes) if scopes else []

        if keyword_only:
            if short_query:
                # FTS5 trigram needs >= 3 chars; fall back to LIKE
                cursor = await db.execute(
                    f"""
                    SELECT n.content, n.type FROM nodes n
                    WHERE n.content LIKE ? AND {scope_ids_sql}
                    LIMIT ?
                    """,
                    [f"%{query_text}%", *scope_params, top_k],
                )
            else:
                escaped = _escape_fts5(query_text)
                cursor = await db.execute(
                    f"""
                    SELECT n.content, n.type
                    FROM fts_nodes f
                    JOIN nodes n ON n.id = f.rowid
                    WHERE fts_nodes MATCH ? AND {scope_ids_sql}
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    [escaped, *scope_params, top_k],
                )
        else:
            if short_query:
                # Can't use FTS5; fall back to embedding-only
                return await self.knn_search(embedding, top_k, scopes)

            query_blob = _embedding_to_blob(embedding)
            escaped = _escape_fts5(query_text)

            if scopes:
                # Scoped RRF: rank the scope subset with the vec_distance_cosine
                # scalar (exact) and scope-filtered FTS matches.
                cursor = await db.execute(
                    f"""
                    WITH scoped AS (
                        SELECT DISTINCT node_id FROM node_scopes WHERE scope IN ({scope_ph})
                    ),
                    emb_ranked AS (
                        SELECT n.id AS node_id,
                            ROW_NUMBER() OVER (ORDER BY vec_distance_cosine(n.embedding, ?)) AS rank
                        FROM nodes n
                        WHERE n.id IN (SELECT node_id FROM scoped)
                    ),
                    kw_ranked AS (
                        SELECT f.rowid AS node_id,
                            ROW_NUMBER() OVER (ORDER BY f.rank) AS kw_rank
                        FROM fts_nodes f
                        WHERE fts_nodes MATCH ? AND f.rowid IN (SELECT node_id FROM scoped)
                    )
                    SELECT n.content, n.type
                    FROM nodes n
                    LEFT JOIN emb_ranked e ON e.node_id = n.id
                    LEFT JOIN kw_ranked k ON k.node_id = n.id
                    WHERE e.node_id IS NOT NULL OR k.node_id IS NOT NULL
                    ORDER BY COALESCE(1.0 / (? + e.rank), 0) + COALESCE(1.0 / (? + k.kw_rank), 0) DESC
                    LIMIT ?
                    """,
                    [*scope_params, query_blob, escaped, rrf_k, rrf_k, top_k],
                )
            else:
                fetch_k = top_k * 3
                cursor = await db.execute(
                    """
                    WITH emb_ranked AS (
                        SELECT node_id,
                            ROW_NUMBER() OVER (ORDER BY distance) AS rank
                        FROM vec_nodes
                        WHERE embedding MATCH ? AND k = ?
                    ),
                    kw_ranked AS (
                        SELECT rowid AS node_id,
                            ROW_NUMBER() OVER (ORDER BY rank) AS kw_rank
                        FROM fts_nodes
                        WHERE fts_nodes MATCH ?
                    )
                    SELECT n.content, n.type
                    FROM nodes n
                    LEFT JOIN emb_ranked e ON e.node_id = n.id
                    LEFT JOIN kw_ranked k ON k.node_id = n.id
                    WHERE e.node_id IS NOT NULL OR k.node_id IS NOT NULL
                    ORDER BY COALESCE(1.0 / (? + e.rank), 0) + COALESCE(1.0 / (? + k.kw_rank), 0) DESC
                    LIMIT ?
                    """,
                    (query_blob, fetch_k, escaped, rrf_k, rrf_k, top_k),
                )

        rows = await cursor.fetchall()
        now = datetime.now().isoformat()
        results = []
        for content, node_type in rows:
            await db.execute(
                "UPDATE nodes SET last_accessed = ? WHERE content = ?",
                (now, content),
            )
            results.append({"content": content, "type": node_type})
        await db.commit()
        return results

    async def get_neighbors(
        self, content: str, scopes: set[str] | None = None
    ) -> list[dict[str, str]]:
        db = await self._conn()
        params: list = [content, content, content]
        scope_clause = ""
        if scopes:
            placeholders = ", ".join("?" for _ in scopes)
            scope_clause = (
                f" WHERE n.id IN "
                f"(SELECT node_id FROM node_scopes WHERE scope IN ({placeholders}))"
            )
            params.extend(sorted(scopes))
        cursor = await db.execute(
            f"""
            SELECT n.content, n.type, e.label,
                   CASE WHEN e.from_content = ? THEN 'out' ELSE 'in' END AS direction
            FROM nodes n
            INNER JOIN edges e ON (e.to_content = n.content AND e.from_content = ?)
                               OR (e.from_content = n.content AND e.to_content = ?)
            {scope_clause}
            """,
            params,
        )
        # Dedup by neighbor, preferring a labeled edge so causal links surface.
        neighbors: dict[str, dict[str, str]] = {}
        for c, node_type, label, direction in await cursor.fetchall():
            existing = neighbors.get(c)
            if existing is None or (label is not None and existing["label"] is None):
                neighbors[c] = {"content": c, "type": node_type, "label": label, "direction": direction}
        return list(neighbors.values())

    async def delete_stale_nodes(self, days: int) -> int:
        db = await self._conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Fetch stale node IDs before deleting (needed for vec_nodes/fts_nodes cleanup)
        cursor = await db.execute(
            "SELECT id FROM nodes WHERE last_accessed < ?", (cutoff,)
        )
        stale_ids = [row[0] for row in await cursor.fetchall()]

        if not stale_ids:
            return 0

        placeholders = ",".join("?" for _ in stale_ids)

        # Delete from virtual tables first
        await db.execute(
            f"DELETE FROM vec_nodes WHERE node_id IN ({placeholders})",
            stale_ids,
        )
        await db.execute(
            f"DELETE FROM fts_nodes WHERE rowid IN ({placeholders})",
            stale_ids,
        )

        # Delete from main table (CASCADE handles edges)
        cursor = await db.execute(
            f"DELETE FROM nodes WHERE id IN ({placeholders})",
            stale_ids,
        )
        await db.commit()
        return cursor.rowcount

    async def delete_nodes(self, contents: list[str]) -> int:
        if not contents:
            return 0
        db = await self._conn()

        # Look up ids first, needed for vec_nodes/fts_nodes cleanup
        placeholders = ",".join("?" for _ in contents)
        cursor = await db.execute(
            f"SELECT id FROM nodes WHERE content IN ({placeholders})", contents
        )
        ids = [row[0] for row in await cursor.fetchall()]

        if not ids:
            return 0

        id_placeholders = ",".join("?" for _ in ids)

        # Delete from virtual tables first
        await db.execute(
            f"DELETE FROM vec_nodes WHERE node_id IN ({id_placeholders})", ids
        )
        await db.execute(
            f"DELETE FROM fts_nodes WHERE rowid IN ({id_placeholders})", ids
        )

        # Delete from main table (CASCADE handles edges)
        cursor = await db.execute(
            f"DELETE FROM nodes WHERE id IN ({id_placeholders})", ids
        )
        await db.commit()
        return cursor.rowcount

    async def get_created_at(self, contents: list[str]) -> dict[str, str]:
        if not contents:
            return {}
        db = await self._conn()
        placeholders = ",".join("?" for _ in contents)
        cursor = await db.execute(
            f"SELECT content, created_at FROM nodes WHERE content IN ({placeholders})",
            contents,
        )
        return {row[0]: row[1] for row in await cursor.fetchall()}

    async def get_scopes(self, contents: list[str]) -> dict[str, set[str]]:
        if not contents:
            return {}
        db = await self._conn()
        placeholders = ",".join("?" for _ in contents)
        cursor = await db.execute(
            f"""
            SELECT n.content, s.scope
            FROM nodes n
            JOIN node_scopes s ON s.node_id = n.id
            WHERE n.content IN ({placeholders})
            """,
            contents,
        )
        result: dict[str, set[str]] = {}
        for content, scope in await cursor.fetchall():
            result.setdefault(content, set()).add(scope)
        return result

    async def get_causal_relations(self, contents: list[str]) -> dict[str, list[dict]]:
        if not contents:
            return {}
        db = await self._conn()
        placeholders = ",".join("?" for _ in contents)
        cursor = await db.execute(
            f"""
            SELECT c2.to_content AS fact, ce.from_content AS cause, ce.to_content AS effect
            FROM edges ce
            JOIN edges c2 ON c2.from_content = ce.from_content
            JOIN edges e2 ON e2.from_content = ce.to_content AND e2.to_content = c2.to_content
            WHERE ce.label = 'causes' AND c2.to_content IN ({placeholders})
            """,
            contents,
        )
        result: dict[str, list[dict]] = {}
        seen: set[tuple[str, str, str]] = set()
        for fact, cause, effect in await cursor.fetchall():
            key = (fact, cause, effect)
            if key in seen:
                continue
            seen.add(key)
            result.setdefault(fact, []).append({"cause": cause, "effect": effect})
        return result

    async def get_all_nodes(self, scopes: set[str] | None = None) -> list[Node]:
        db = await self._conn()
        if scopes:
            scope_ph = ",".join("?" for _ in scopes)
            cursor = await db.execute(
                f"""
                SELECT n.id, n.content, n.type, n.embedding, n.created_at, n.last_accessed
                FROM nodes n
                WHERE n.id IN (SELECT node_id FROM node_scopes WHERE scope IN ({scope_ph}))
                """,
                list(scopes),
            )
        else:
            cursor = await db.execute(
                "SELECT id, content, type, embedding, created_at, last_accessed FROM nodes"
            )
        rows = await cursor.fetchall()

        # Fetch scope tags for the returned nodes
        scope_map: dict[int, set[str]] = {}
        cursor = await db.execute("SELECT node_id, scope FROM node_scopes")
        for node_id, scope in await cursor.fetchall():
            scope_map.setdefault(node_id, set()).add(scope)

        nodes = []
        for node_id, content, node_type, blob, created_at, last_accessed in rows:
            nodes.append(Node(
                content=content,
                type=node_type,
                embedding=_blob_to_list(blob),
                created_at=datetime.fromisoformat(created_at),
                last_accessed=datetime.fromisoformat(last_accessed),
                scopes=scope_map.get(node_id, set()),
            ))
        return nodes

    async def get_all_edges(self) -> list[Edge]:
        db = await self._conn()
        cursor = await db.execute(
            "SELECT from_content, to_content, last_accessed, label FROM edges"
        )
        return [
            Edge(
                from_content=row[0],
                to_content=row[1],
                last_accessed=datetime.fromisoformat(row[2]),
                label=row[3],
            )
            for row in await cursor.fetchall()
        ]
