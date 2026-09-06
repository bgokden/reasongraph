from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Callable, Protocol, Union, runtime_checkable

from sentence_transformers import SentenceTransformer, CrossEncoder


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into vectors.

    ``encode`` must accept either a single ``str`` (returning one vector) or a
    ``list[str]`` (returning one vector per text). Returned vectors may be
    Python lists, numpy arrays, or torch tensors -- they are normalized to
    ``list[float]``. This matches ``SentenceTransformer.encode``, so any such
    model works, as does a small adapter wrapping another embedding library.
    """

    def encode(self, texts: Any) -> Any: ...


# What ReasonGraph / EmbeddingManager accept for the embedder: a model name,
# an object with an encode() method, a plain callable, or None (default model).
EmbedderLike = Union[str, Embedder, Callable[[Any], Any], None]


@runtime_checkable
class Reranker(Protocol):
    """Anything that scores query-document pairs.

    ``predict`` takes a list of ``(query, document)`` tuples (all sharing the
    query) and returns one relevance score per pair. This matches
    ``sentence_transformers.CrossEncoder.predict``, so a CrossEncoder works, as
    does a small ONNX adapter exposing the same method.
    """

    def predict(self, pairs: Any) -> Any: ...


# What EmbeddingManager accepts for the reranker: a model name, an object with a
# predict() method (CrossEncoder or adapter), or None (default model).
RerankerLike = Union[str, Reranker, None]


class EmbeddingManager:
    """Manages embedding generation and cross-encoder reranking."""

    DEFAULT_EMBED_MODEL = "all-MiniLM-L12-v2"
    DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        embed_model: EmbedderLike = None,
        rerank_model: RerankerLike = None,
    ) -> None:
        # str is checked before the encode() duck-check because str itself has
        # an encode() method (text -> bytes), which is not what we want.
        if embed_model is None or isinstance(embed_model, str):
            self._embed: Any = SentenceTransformer(
                embed_model or self.DEFAULT_EMBED_MODEL
            )
            self._encode = self._embed.encode
        elif hasattr(embed_model, "encode"):
            self._embed = embed_model
            self._encode = embed_model.encode
        elif callable(embed_model):
            self._embed = embed_model
            self._encode = embed_model
        else:
            raise TypeError(
                "embed_model must be None, a model name (str), an object with "
                "an encode() method, or a callable that encodes text"
            )

        self._rerank: CrossEncoder | None = None
        self._rerank_name = rerank_model

    @staticmethod
    def _to_vector(vec: Any) -> list[float]:
        """Normalize a single vector (numpy/torch/list) to list[float]."""
        tolist = getattr(vec, "tolist", None)
        if callable(tolist):
            return tolist()
        return list(vec)

    _CACHE_SIZE = 256

    def encode(self, text: str) -> list[float]:
        """Encode a single text string to a float vector.

        A small LRU cache: a write path encodes the same text several times
        (dedup check, node insert, conflict candidates) and agents often re-ask
        the same question; caching avoids the repeated model call."""
        cache = self.__dict__.setdefault("_encode_cache", {})
        vec = cache.get(text)
        if vec is None:
            vec = self._to_vector(self._encode(text))
            if len(cache) >= self._CACHE_SIZE:
                cache.pop(next(iter(cache)))
            cache[text] = vec
        else:
            cache.pop(text); cache[text] = vec   # refresh LRU order
        return list(vec)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts at once."""
        result = self._encode(texts)
        tolist = getattr(result, "tolist", None)
        if callable(tolist):
            return result.tolist()
        return [self._to_vector(v) for v in result]

    def rerank(
        self,
        query: str,
        results: list[dict[str, str]],
        top_k: int,
        recency_weight: float = 0.0,
    ) -> list[dict[str, str]]:
        """Rerank results using a cross-encoder. Lazy-loads the model on first call.

        When ``recency_weight`` > 0, the cross-encoder relevance score is blended
        with a recency score derived from each result's ``created_at`` (when
        present), so newer facts outrank older contradicting ones. The blend is
        ``(1 - recency_weight) * relevance + recency_weight * recency``, both
        normalized to [0, 1] within the candidate set. ``recency_weight`` = 0
        (the default) leaves ranking behavior unchanged.
        """
        if not results:
            return []

        # Deduplicate by content
        seen = {}
        for r in results:
            if r["content"] not in seen:
                seen[r["content"]] = r
        unique = list(seen.values())

        if len(unique) <= 1:
            return unique[:top_k]

        if self._rerank is None:
            if self._rerank_name is not None and not isinstance(self._rerank_name, str):
                # Any object exposing predict(pairs) -> scores (CrossEncoder,
                # a fastembed adapter, etc.).
                self._rerank = self._rerank_name
            else:
                self._rerank = CrossEncoder(self._rerank_name or self.DEFAULT_RERANK_MODEL)

        pairs = [(query, r["content"]) for r in unique]
        scores = self._rerank.predict(pairs)

        if recency_weight > 0:
            blended = self._apply_recency(list(scores), unique, recency_weight)
            order = sorted(range(len(unique)), key=lambda i: blended[i], reverse=True)
            ranked = [unique[i] for i in order]
        else:
            ranked = [r for _, r in sorted(zip(scores, unique), reverse=True)]
        return ranked[:top_k]

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Embedding cosine similarity of each text to the query, in [-1, 1].

        Used to attach a comparable, bounded relevance score to structured query
        results (for thresholding / display). Reuses the embedder rather than the
        cross-encoder so the number is bounded and cheap. Returns one score per
        input text, in order.
        """
        if not texts:
            return []
        q = self.encode(query)
        q_norm = math.sqrt(sum(x * x for x in q)) or 1.0
        out: list[float] = []
        for vec in self.encode_batch(texts):
            v_norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            dot = sum(a * b for a, b in zip(q, vec))
            out.append(dot / (q_norm * v_norm))
        return out

    @staticmethod
    def _apply_recency(
        scores: list[float], items: list[dict[str, str]], weight: float
    ) -> list[float]:
        """Blend relevance scores with recency, both min-max normalized to [0, 1]."""
        smin, smax = min(scores), max(scores)
        srange = smax - smin
        relevance = [(s - smin) / srange if srange else 0.5 for s in scores]

        times: list[float | None] = []
        for it in items:
            raw = it.get("created_at")
            try:
                times.append(datetime.fromisoformat(raw).timestamp() if raw else None)
            except (TypeError, ValueError):
                times.append(None)

        valid = [t for t in times if t is not None]
        tmin, tmax = (min(valid), max(valid)) if valid else (0.0, 0.0)
        trange = tmax - tmin

        recency = []
        for t in times:
            if t is None:
                recency.append(0.0)  # unknown age -> treat as oldest
            elif trange:
                recency.append((t - tmin) / trange)
            else:
                recency.append(0.5)  # all same time -> neutral

        return [
            (1 - weight) * rel + weight * rec
            for rel, rec in zip(relevance, recency)
        ]
