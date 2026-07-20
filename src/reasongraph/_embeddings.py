from __future__ import annotations

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


class EmbeddingManager:
    """Manages embedding generation and cross-encoder reranking."""

    DEFAULT_EMBED_MODEL = "all-MiniLM-L12-v2"
    DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        embed_model: EmbedderLike = None,
        rerank_model: str | CrossEncoder | None = None,
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

    def encode(self, text: str) -> list[float]:
        """Encode a single text string to a float vector."""
        return self._to_vector(self._encode(text))

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
    ) -> list[dict[str, str]]:
        """Rerank results using a cross-encoder. Lazy-loads the model on first call."""
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
            model_name = (
                self._rerank_name
                if isinstance(self._rerank_name, str)
                else self.DEFAULT_RERANK_MODEL
            )
            if isinstance(self._rerank_name, CrossEncoder):
                self._rerank = self._rerank_name
            else:
                self._rerank = CrossEncoder(model_name)

        pairs = [(query, r["content"]) for r in unique]
        scores = self._rerank.predict(pairs)
        ranked = [r for _, r in sorted(zip(scores, unique), reverse=True)]
        return ranked[:top_k]
