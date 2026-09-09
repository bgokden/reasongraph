from __future__ import annotations

import re

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
        # (from_content, to_content) -> edge label (None for untyped edges)
        self._edges: dict[tuple[str, str], str | None] = {}

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
                scopes=set(n.get("scopes", [])),
                invalid_at=(datetime.fromisoformat(n["invalid_at"])
                            if n.get("invalid_at") else None),
            )
        for e in data.get("edges", []):
            self._edges[(e["from_content"], e["to_content"])] = e.get("label")

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
                "scopes": sorted(node.scopes),
                "invalid_at": node.invalid_at.isoformat() if node.invalid_at else None,
            })
        edges = []
        for (from_c, to_c), label in self._edges.items():
            edges.append({"from_content": from_c, "to_content": to_c, "label": label})
        with open(self.file_path, "w") as f:
            json.dump({"nodes": nodes, "edges": edges}, f)

    async def insert_nodes(self, nodes: list[Node]) -> None:
        now = datetime.now()
        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"Node '{node.content}' has no embedding")

        # Process in order; the first occurrence of a content stores a COPY of
        # the caller's node (so we never mutate the caller's objects), and later
        # occurrences (in-batch or already stored) union their scopes into it.
        for node in nodes:
            content = node.content
            if content in self._nodes:
                existing = self._nodes[content]
                existing.last_accessed = now
                existing.scopes |= node.scopes
                # Re-asserting a fact revives it: a previously retired fact
                # (soft-superseded) becomes current again.
                existing.invalid_at = None
            else:
                self._nodes[content] = Node(
                    content=content,
                    type=node.type,
                    embedding=node.embedding,
                    created_at=node.created_at,
                    last_accessed=now,
                    scopes=set(node.scopes),
                )

    def _candidates(self, scopes: set[str] | None) -> list[Node]:
        """Nodes eligible as search seeds, filtered by scope when given."""
        if scopes:
            return [n for n in self._nodes.values() if n.scopes & scopes]
        return list(self._nodes.values())

    async def insert_edges(self, edges: list[Edge]) -> None:
        for edge in edges:
            self._edges[(edge.from_content, edge.to_content)] = edge.label

    async def knn_search(
        self, embedding: list[float], top_k: int,
        scopes: set[str] | None = None,
    ) -> list[dict[str, str]]:
        candidates = self._candidates(scopes)
        if not candidates:
            return []

        contents = []
        types = []
        embeddings = []
        for node in candidates:
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
            results.append({"content": content, "type": types[idx],
                            "score": float(similarities[idx])})
        return results

    async def hybrid_search(
        self, embedding: list[float], query_text: str, top_k: int,
        rrf_k: int = 60, keyword_only: bool = False,
        scopes: set[str] | None = None,
    ) -> list[dict[str, str]]:
        candidates = self._candidates(scopes)
        if not candidates:
            return []

        contents = []
        types = []
        embeddings = []
        for node in candidates:
            contents.append(node.content)
            types.append(node.type)
            embeddings.append(node.embedding)

        # Lexical channel: which words of the question this fact actually contains.
        # Postgres does this with pg_trgm's strict word similarity and an index; here the same
        # idea in Python, weighted so a match on a rare word ("Northwind") counts and a match on
        # a common one ("the") does not. A fact sharing no question word scores zero.
        import math

        def _grams(word: str) -> set[str]:
            padded = f"  {word} "
            return {padded[i:i + 3] for i in range(len(padded) - 2)}

        def _words(text: str) -> list[str]:
            return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1]

        q_words = _words(query_text)
        doc_words = [set(_words(c)) for c in contents]
        n_docs = max(1, len(contents))

        def _matches(qw: str, words: set[str]) -> bool:
            if qw in words:
                return True
            qg = _grams(qw)
            return any(len(qg & _grams(w)) / len(qg | _grams(w)) >= 0.45 for w in words)   # same bar as pg_trgm.strict_word_similarity_threshold

        kw_scores_raw = [0.0] * len(contents)
        for qw in dict.fromkeys(q_words):
            hits = [i for i, words in enumerate(doc_words) if _matches(qw, words)]
            if not hits:
                continue
            idf = math.log(1.0 + n_docs / len(hits))   # a word in every fact adds nothing
            for i in hits:
                kw_scores_raw[i] += idf

        if keyword_only:
            scored = [(i, sc) for i, sc in enumerate(kw_scores_raw) if sc > 0]
            scored.sort(key=lambda x: -x[1])
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

        # Keyword ranks: word similarity, descending; facts below the threshold do not rank
        kw_scores = [sc if sc > 0 else float("-inf") for sc in kw_scores_raw]
        kw_order = sorted(range(len(kw_scores)), key=lambda i: -kw_scores[i])
        kw_rank = [0] * len(kw_scores)
        for rank_pos, idx in enumerate(kw_order, 1):
            kw_rank[idx] = rank_pos

        # RRF combination
        rrf_scores = []
        for i in range(len(contents)):
            emb_score = 1.0 / (rrf_k + emb_rank[i])
            kw_score = 1.0 / (rrf_k + kw_rank[i]) if kw_scores[i] != float("-inf") else 0.0
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

    async def get_neighbors(
        self, content: str, scopes: set[str] | None = None
    ) -> list[dict[str, str]]:
        # neighbor content -> (type, label, direction). A labeled edge wins over
        # an untyped one when both connect the same pair, so causal links surface.
        # When ``scopes`` is given, only neighbors carrying at least one of those
        # scopes are returned, confining traversal to a tenant (isolation mode).
        neighbors: dict[str, tuple[str, str | None, str]] = {}
        for (from_c, to_c), label in self._edges.items():
            if from_c == content and to_c in self._nodes:
                other, direction = to_c, "out"
            elif to_c == content and from_c in self._nodes:
                other, direction = from_c, "in"
            else:
                continue
            if scopes is not None and not (self._nodes[other].scopes & scopes):
                continue
            existing = neighbors.get(other)
            if existing is None or (label is not None and existing[1] is None):
                neighbors[other] = (self._nodes[other].type, label, direction)
        return [
            {"content": c, "type": t, "label": label, "direction": direction}
            for c, (t, label, direction) in neighbors.items()
        ]

    async def delete_stale_nodes(self, days: int) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        stale = [c for c, n in self._nodes.items() if n.last_accessed < cutoff]
        for content in stale:
            del self._nodes[content]
        # Remove edges referencing deleted nodes
        self._edges = {
            (f, t): label for (f, t), label in self._edges.items()
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
                (f, t): label for (f, t), label in self._edges.items()
                if f in self._nodes and t in self._nodes
            }
        return deleted

    async def get_created_at(self, contents: list[str]) -> dict[str, str]:
        return {
            c: self._nodes[c].created_at.isoformat()
            for c in contents
            if c in self._nodes
        }

    async def set_invalid(self, contents: list[str], when: datetime) -> None:
        for content in contents:
            node = self._nodes.get(content)
            if node is not None:
                node.invalid_at = when

    async def get_validity(self, contents: list[str]) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for c in contents:
            node = self._nodes.get(c)
            if node is not None:
                out[c] = node.invalid_at.isoformat() if node.invalid_at else None
        return out

    async def list_scopes(self, prefix: str | None = None) -> list[str]:
        out: set[str] = set()
        for n in self._nodes.values():
            out |= {s for s in n.scopes if prefix is None or s.startswith(prefix)}
        return sorted(out)

    async def count_nodes(self, node_type: str | None = None, scopes: set[str] | None = None) -> int:
        return sum(1 for n in self._nodes.values()
                   if (node_type is None or n.type == node_type) and (not scopes or (n.scopes & scopes)))

    async def get_node_types(self, contents: list[str]) -> dict[str, str]:
        return {c: self._nodes[c].type for c in contents if c in self._nodes}

    async def _rank_neighbors(self, neighbors, query_embedding, limit):
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        scored = []
        for n in neighbors:
            node = self._nodes.get(n["content"])
            if node is None:
                continue
            v = np.asarray(node.embedding, dtype=np.float32)
            scored.append((float(v @ q / ((np.linalg.norm(v) or 1.0) * qn)), n))
        scored.sort(key=lambda x: -x[0])
        return [n for _, n in scored[:limit]]

    async def count_edges(self) -> int:
        return len(self._edges)

    async def entities_starting_with(self, word: str, limit: int = 20) -> list[str]:
        w = word.lower()
        out = []
        for n in self._nodes.values():
            if n.type != "entity":
                continue
            c = n.content.lower()
            if c == w or c.startswith(w + " "):
                out.append(n.content)
                if len(out) >= limit:
                    break
        return out

    async def nodes_in_scopes(self, scopes: set[str]) -> list[str]:
        return [n.content for n in self._nodes.values() if n.scopes & set(scopes)]

    async def remove_scopes(self, contents: list[str], scopes: set[str]) -> int:
        removed = 0
        for c in contents:
            node = self._nodes.get(c)
            if node is not None:
                before = len(node.scopes)
                node.scopes -= set(scopes)
                removed += before - len(node.scopes)
        return removed

    async def get_scopes(self, contents: list[str]) -> dict[str, set[str]]:
        return {
            c: set(self._nodes[c].scopes)
            for c in contents
            if c in self._nodes
        }

    async def get_causal_relations(self, contents: list[str]) -> dict[str, list[dict]]:
        wanted = set(contents)
        result: dict[str, list[dict]] = {}
        for (cause, effect), label in self._edges.items():
            if label != "causes":
                continue
            # Facts linked to BOTH the cause span and the effect span.
            for fact in wanted:
                if (cause, fact) in self._edges and (effect, fact) in self._edges:
                    result.setdefault(fact, []).append({"cause": cause, "effect": effect})
        return result

    async def get_all_nodes(self, scopes: set[str] | None = None) -> list[Node]:
        return self._candidates(scopes)

    async def get_all_edges(self) -> list[Edge]:
        return [
            Edge(from_content=f, to_content=t, label=label)
            for (f, t), label in self._edges.items()
        ]
