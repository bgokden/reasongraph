from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np

from reasongraph._types import Node, Edge
from reasongraph.backends._base import Backend


class MemoryBackend(Backend):
    """Pure Python in-memory backend with optional JSON file persistence.

    Uses numpy brute-force cosine similarity for KNN search and simple
    substring matching for keyword search. Zero external dependencies
    beyond numpy (already a transitive dep of sentence-transformers).

    Args:
        file_path: Optional path to a JSON file. When set, data is loaded
            on ``initialize()`` and saved on ``close()``.
    """

    def __init__(self, file_path: str | None = None) -> None:
        self.file_path = file_path
        self._nodes: dict[str, Node] = {}
        self._edges: set[tuple[str, str]] = set()

    async def initialize(self) -> None:
        if self.file_path is None:
            return
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        for n in data.get("nodes", []):
            self._nodes[n["content"]] = Node(
                content=n["content"],
                type=n["type"],
                embedding=n["embedding"],
                created_at=datetime.fromisoformat(n["created_at"]),
                last_accessed=datetime.fromisoformat(n["last_accessed"]),
            )
        for e in data.get("edges", []):
            self._edges.add((e["from_content"], e["to_content"]))

    async def close(self) -> None:
        if self.file_path is None:
            return
        nodes = []
        for node in self._nodes.values():
            nodes.append({
                "content": node.content,
                "type": node.type,
                "embedding": node.embedding,
                "created_at": node.created_at.isoformat(),
                "last_accessed": node.last_accessed.isoformat(),
            })
        edges = []
        for from_c, to_c in self._edges:
            edges.append({"from_content": from_c, "to_content": to_c})
        with open(self.file_path, "w") as f:
            json.dump({"nodes": nodes, "edges": edges}, f)

    async def insert_nodes(self, nodes: list[Node]) -> None:
        now = datetime.now()
        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"Node '{node.content}' has no embedding")

        # Deduplicate within batch (keep first occurrence)
        seen: dict[str, Node] = {}
        unique: list[Node] = []
        for node in nodes:
            if node.content not in seen:
                seen[node.content] = node
                unique.append(node)

        for node in unique:
            if node.content in self._nodes:
                # Update last_accessed for existing nodes
                self._nodes[node.content].last_accessed = now
            else:
                node.last_accessed = now
                self._nodes[node.content] = node

    async def insert_edges(self, edges: list[Edge]) -> None:
        for edge in edges:
            self._edges.add((edge.from_content, edge.to_content))

    async def knn_search(
        self, embedding: list[float], top_k: int
    ) -> list[dict[str, str]]:
        if not self._nodes:
            return []

        contents = []
        types = []
        embeddings = []
        for node in self._nodes.values():
            contents.append(node.content)
            types.append(node.type)
            embeddings.append(node.embedding)

        query_vec = np.asarray(embedding, dtype=np.float32)
        mat = np.asarray(embeddings, dtype=np.float32)

        # Cosine similarity: dot(q, m) / (||q|| * ||m||)
        query_norm = np.linalg.norm(query_vec)
        mat_norms = np.linalg.norm(mat, axis=1)
        # Avoid division by zero
        denom = query_norm * mat_norms
        denom = np.where(denom == 0, 1.0, denom)
        similarities = mat @ query_vec / denom

        k = min(top_k, len(contents))
        if k < len(contents):
            top_indices = np.argpartition(-similarities, k)[:k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
        else:
            top_indices = np.argsort(-similarities)

        now = datetime.now()
        results = []
        for idx in top_indices:
            content = contents[idx]
            self._nodes[content].last_accessed = now
            results.append({"content": content, "type": types[idx]})
        return results

    async def hybrid_search(
        self, embedding: list[float], query_text: str, top_k: int,
        rrf_k: int = 60, keyword_only: bool = False,
    ) -> list[dict[str, str]]:
        if not self._nodes:
            return []

        contents = []
        types = []
        embeddings = []
        for node in self._nodes.values():
            contents.append(node.content)
            types.append(node.type)
            embeddings.append(node.embedding)

        query_lower = query_text.lower()

        if keyword_only:
            # Rank by substring match presence, then by position (earlier = better)
            scored = []
            for i, content in enumerate(contents):
                pos = content.lower().find(query_lower)
                if pos >= 0:
                    scored.append((i, pos))
            # Sort by match position (earlier matches first)
            scored.sort(key=lambda x: x[1])
            now = datetime.now()
            results = []
            for idx, _ in scored[:top_k]:
                self._nodes[contents[idx]].last_accessed = now
                results.append({"content": contents[idx], "type": types[idx]})
            return results

        # Hybrid: embedding + keyword with RRF
        query_vec = np.asarray(embedding, dtype=np.float32)
        mat = np.asarray(embeddings, dtype=np.float32)

        query_norm = np.linalg.norm(query_vec)
        mat_norms = np.linalg.norm(mat, axis=1)
        denom = query_norm * mat_norms
        denom = np.where(denom == 0, 1.0, denom)
        similarities = mat @ query_vec / denom

        # Embedding ranks (1-based)
        emb_order = np.argsort(-similarities)
        emb_rank = np.empty_like(emb_order)
        emb_rank[emb_order] = np.arange(1, len(emb_order) + 1)

        # Keyword ranks: substring match scored by position, non-matches get high rank
        kw_scores = []
        for content in contents:
            pos = content.lower().find(query_lower)
            if pos >= 0:
                kw_scores.append(pos)
            else:
                kw_scores.append(float("inf"))

        kw_order = sorted(range(len(kw_scores)), key=lambda i: kw_scores[i])
        kw_rank = [0] * len(kw_scores)
        for rank_pos, idx in enumerate(kw_order, 1):
            kw_rank[idx] = rank_pos

        # RRF combination
        rrf_scores = []
        for i in range(len(contents)):
            emb_score = 1.0 / (rrf_k + emb_rank[i])
            kw_score = 1.0 / (rrf_k + kw_rank[i]) if kw_scores[i] != float("inf") else 0.0
            rrf_scores.append(emb_score + kw_score)

        rrf_arr = np.asarray(rrf_scores, dtype=np.float32)
        k = min(top_k, len(contents))
        if k < len(contents):
            top_indices = np.argpartition(-rrf_arr, k)[:k]
            top_indices = top_indices[np.argsort(-rrf_arr[top_indices])]
        else:
            top_indices = np.argsort(-rrf_arr)

        now = datetime.now()
        results = []
        for idx in top_indices:
            self._nodes[contents[idx]].last_accessed = now
            results.append({"content": contents[idx], "type": types[idx]})
        return results

    async def get_neighbors(self, content: str) -> list[dict[str, str]]:
        neighbors: dict[str, str] = {}
        for from_c, to_c in self._edges:
            if from_c == content and to_c in self._nodes:
                neighbors[to_c] = self._nodes[to_c].type
            elif to_c == content and from_c in self._nodes:
                neighbors[from_c] = self._nodes[from_c].type
        return [{"content": c, "type": t} for c, t in neighbors.items()]

    async def delete_stale_nodes(self, days: int) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        stale = [c for c, n in self._nodes.items() if n.last_accessed < cutoff]
        for content in stale:
            del self._nodes[content]
        # Remove edges referencing deleted nodes
        self._edges = {
            (f, t) for f, t in self._edges
            if f in self._nodes and t in self._nodes
        }
        return len(stale)

    async def delete_nodes(self, contents: list[str]) -> int:
        deleted = 0
        for content in contents:
            if content in self._nodes:
                del self._nodes[content]
                deleted += 1
        if deleted:
            # Remove edges referencing deleted nodes
            self._edges = {
                (f, t) for f, t in self._edges
                if f in self._nodes and t in self._nodes
            }
        return deleted

    async def get_created_at(self, contents: list[str]) -> dict[str, str]:
        return {
            c: self._nodes[c].created_at.isoformat()
            for c in contents
            if c in self._nodes
        }

    async def get_all_nodes(self) -> list[Node]:
        return list(self._nodes.values())

    async def get_all_edges(self) -> list[Edge]:
        return [
            Edge(from_content=f, to_content=t)
            for f, t in self._edges
        ]
