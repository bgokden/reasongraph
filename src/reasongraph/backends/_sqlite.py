from __future__ import annotations

import struct
from datetime import datetime, timedelta

import aiosqlite
import numpy as np

from reasongraph._types import Node, Edge
from reasongraph.backends._base import Backend


def _trigrams(text: str) -> set[str]:
    """Extract character trigrams from text, lowercased with padding."""
    if not text:
        return set()
    s = f"  {text.lower()}  "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _trigram_similarity(a: str, b: str) -> float:
    """Jaccard similarity between trigram sets of two strings."""
    trgm_a = _trigrams(a)
    trgm_b = _trigrams(b)
    if not trgm_a or not trgm_b:
        return 0.0
    intersection = len(trgm_a & trgm_b)
    union = len(trgm_a | trgm_b)
    return intersection / union if union else 0.0


def _embedding_to_blob(embedding: list[float]) -> bytes:
    """Pack a float list into a compact binary blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    """Unpack a binary blob back to a numpy array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


class SqliteBackend(Backend):
    """SQLite backend with brute-force numpy cosine similarity for vector search."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return self._db

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                content TEXT PRIMARY KEY,
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
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def insert_nodes(self, nodes: list[Node]) -> None:
        db = await self._conn()
        now = datetime.now().isoformat()
        rows = []
        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"Node '{node.content}' has no embedding")
            rows.append((
                node.content,
                _embedding_to_blob(node.embedding),
                node.created_at.isoformat(),
                now,
                node.type,
            ))
        await db.executemany(
            """
            INSERT INTO nodes (content, embedding, created_at, last_accessed, type)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(content) DO UPDATE SET last_accessed = excluded.last_accessed
            """,
            rows,
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
        query_vec = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        cursor = await db.execute("SELECT content, type, embedding FROM nodes")
        rows = await cursor.fetchall()
        if not rows:
            return []

        scored = []
        for content, node_type, blob in rows:
            node_vec = _blob_to_embedding(blob)
            node_norm = np.linalg.norm(node_vec)
            if node_norm == 0:
                continue
            similarity = float(np.dot(query_vec, node_vec) / (query_norm * node_norm))
            scored.append((similarity, content, node_type))

        scored.sort(key=lambda x: x[0], reverse=True)
        now = datetime.now().isoformat()
        results = []
        for _, content, node_type in scored[:top_k]:
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

        cursor = await db.execute("SELECT content, type, embedding FROM nodes")
        rows = await cursor.fetchall()
        if not rows:
            return []

        # Compute trigram scores for all nodes
        trgm_scores = []
        for content, node_type, blob in rows:
            trgm_scores.append((_trigram_similarity(query_text, content), content, node_type))

        if keyword_only:
            trgm_scores.sort(key=lambda x: x[0], reverse=True)
            now = datetime.now().isoformat()
            results = []
            for _, content, node_type in trgm_scores[:top_k]:
                await db.execute(
                    "UPDATE nodes SET last_accessed = ? WHERE content = ?",
                    (now, content),
                )
                results.append({"content": content, "type": node_type})
            await db.commit()
            return results

        # Compute embedding cosine scores for all nodes
        query_vec = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        emb_scores = []
        for content, node_type, blob in rows:
            cosine_sim = 0.0
            if query_norm > 0:
                node_vec = _blob_to_embedding(blob)
                node_norm = np.linalg.norm(node_vec)
                if node_norm > 0:
                    cosine_sim = float(np.dot(query_vec, node_vec) / (query_norm * node_norm))
            emb_scores.append((cosine_sim, content, node_type))

        # Rank independently (1-based ranks)
        emb_scores.sort(key=lambda x: x[0], reverse=True)
        trgm_scores.sort(key=lambda x: x[0], reverse=True)

        emb_rank = {item[1]: rank for rank, item in enumerate(emb_scores, start=1)}
        trgm_rank = {item[1]: rank for rank, item in enumerate(trgm_scores, start=1)}
        content_type = {content: node_type for _, content, node_type in emb_scores}

        # Reciprocal Rank Fusion
        rrf_scored = []
        for content in emb_rank:
            score = 1.0 / (rrf_k + emb_rank[content]) + 1.0 / (rrf_k + trgm_rank[content])
            rrf_scored.append((score, content, content_type[content]))

        rrf_scored.sort(key=lambda x: x[0], reverse=True)
        now = datetime.now().isoformat()
        results = []
        for _, content, node_type in rrf_scored[:top_k]:
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
        cursor = await db.execute(
            "DELETE FROM nodes WHERE last_accessed < ?", (cutoff,)
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
                embedding=_blob_to_embedding(blob).tolist(),
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
