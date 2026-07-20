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
                UNIQUE(from_content, to_content)
            )
        """)

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

        # Deduplicate within the batch (keep first occurrence)
        seen: dict[str, Node] = {}
        unique_nodes: list[Node] = []
        for node in nodes:
            if node.content not in seen:
                seen[node.content] = node
                unique_nodes.append(node)

        # Check which contents already exist in the database
        contents = [node.content for node in unique_nodes]
        placeholders = ",".join("?" for _ in contents)
        cursor = await db.execute(
            f"SELECT content FROM nodes WHERE content IN ({placeholders})",
            contents,
        )
        existing = {row[0] for row in await cursor.fetchall()}

        # Update last_accessed for existing nodes
        for node in unique_nodes:
            if node.content in existing:
                await db.execute(
                    "UPDATE nodes SET last_accessed = ? WHERE content = ?",
                    (now, node.content),
                )

        # Insert new nodes into all three tables
        for node in unique_nodes:
            if node.content not in existing:
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

        await db.commit()

    async def insert_edges(self, edges: list[Edge]) -> None:
        db = await self._conn()
        now = datetime.now().isoformat()
        rows = [(e.from_content, e.to_content, now) for e in edges]
        await db.executemany(
            """
            INSERT INTO edges (from_content, to_content, last_accessed)
            VALUES (?, ?, ?)
            ON CONFLICT(from_content, to_content) DO NOTHING
            """,
            rows,
        )
        await db.commit()

    async def knn_search(
        self, embedding: list[float], top_k: int
    ) -> list[dict[str, str]]:
        db = await self._conn()
        query_blob = _embedding_to_blob(embedding)

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
    ) -> list[dict[str, str]]:
        db = await self._conn()
        short_query = len(query_text) < 3

        if keyword_only:
            if short_query:
                # FTS5 trigram needs >= 3 chars; fall back to LIKE
                cursor = await db.execute(
                    """
                    SELECT content, type FROM nodes
                    WHERE content LIKE ?
                    LIMIT ?
                    """,
                    (f"%{query_text}%", top_k),
                )
            else:
                escaped = _escape_fts5(query_text)
                cursor = await db.execute(
                    """
                    SELECT n.content, n.type
                    FROM fts_nodes f
                    JOIN nodes n ON n.id = f.rowid
                    WHERE fts_nodes MATCH ?
                    ORDER BY f.rank
                    LIMIT ?
                    """,
                    (escaped, top_k),
                )
        else:
            if short_query:
                # Can't use FTS5; fall back to embedding-only
                return await self.knn_search(embedding, top_k)

            query_blob = _embedding_to_blob(embedding)
            escaped = _escape_fts5(query_text)
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

    async def get_neighbors(self, content: str) -> list[dict[str, str]]:
        db = await self._conn()
        cursor = await db.execute(
            """
            SELECT DISTINCT n.content, n.type FROM nodes n
            INNER JOIN edges e ON (e.to_content = n.content AND e.from_content = ?)
                               OR (e.from_content = n.content AND e.to_content = ?)
            """,
            (content, content),
        )
        return [{"content": row[0], "type": row[1]} for row in await cursor.fetchall()]

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

    async def get_all_nodes(self) -> list[Node]:
        db = await self._conn()
        cursor = await db.execute(
            "SELECT content, type, embedding, created_at, last_accessed FROM nodes"
        )
        nodes = []
        for content, node_type, blob, created_at, last_accessed in await cursor.fetchall():
            nodes.append(Node(
                content=content,
                type=node_type,
                embedding=_blob_to_list(blob),
                created_at=datetime.fromisoformat(created_at),
                last_accessed=datetime.fromisoformat(last_accessed),
            ))
        return nodes

    async def get_all_edges(self) -> list[Edge]:
        db = await self._conn()
        cursor = await db.execute(
            "SELECT from_content, to_content, last_accessed FROM edges"
        )
        return [
            Edge(
                from_content=row[0],
                to_content=row[1],
                last_accessed=datetime.fromisoformat(row[2]),
            )
            for row in await cursor.fetchall()
        ]
