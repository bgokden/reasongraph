from __future__ import annotations

from sentence_transformers import SentenceTransformer, CrossEncoder


class EmbeddingManager:
    """Manages embedding generation and cross-encoder reranking."""

    DEFAULT_EMBED_MODEL = "all-MiniLM-L12-v2"
    DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        embed_model: str | SentenceTransformer | None = None,
        rerank_model: str | CrossEncoder | None = None,
    ) -> None:
        if isinstance(embed_model, SentenceTransformer):
            self._embed = embed_model
        else:
            self._embed = SentenceTransformer(embed_model or self.DEFAULT_EMBED_MODEL)

        self._rerank: CrossEncoder | None = None
        self._rerank_name = rerank_model

    def encode(self, text: str) -> list[float]:
        """Encode a single text string to a float vector."""
        return self._embed.encode(text).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts at once."""
        return self._embed.encode(texts).tolist()

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
