"""Pure-ONNX embedder/reranker adapters backed by fastembed.

fastembed ships pre-exported, CPU-optimized ONNX sentence-transformer and
cross-encoder models (including quantized variants), so these give faster cold
start and lower RAM than the PyTorch defaults without an export step. They slot
into the existing pluggable seams: ``FastEmbedEmbedder`` has an ``encode``
method (accepted by ``ReasonGraph(embed_model=...)``) and ``FastEmbedReranker``
exposes ``predict(pairs)`` (accepted by ``rerank_model=...``).

fastembed is an optional dependency, imported lazily on first use.
"""

from __future__ import annotations

from typing import Any


class FastEmbedEmbedder:
    """ONNX sentence embedder via fastembed.

    Args:
        model_name: A fastembed-supported embedding model
            (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"`` or the
            multilingual ``"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"``).
        **kwargs: Passed through to ``fastembed.TextEmbedding`` (e.g. ``threads``,
            ``providers``).
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None, **kwargs: Any) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed not installed. Install with: pip install fastembed"
            )
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = TextEmbedding(model_name=self.model_name, **kwargs)

    def encode(self, texts):
        """Encode a single string or a list of strings to plain float lists."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = [v.tolist() for v in self._model.embed(items)]
        return vecs[0] if single else vecs


class FastEmbedReranker:
    """ONNX cross-encoder reranker via fastembed.

    Exposes a ``predict(pairs)`` method compatible with the reranking call in
    ``EmbeddingManager`` (``pairs`` is a list of ``(query, document)`` tuples,
    all sharing the query). Returns one relevance score per pair.

    Args:
        model_name: A fastembed-supported cross-encoder
            (e.g. ``"Xenova/ms-marco-MiniLM-L-6-v2"`` -- the ONNX build of the
            default reranker, so scores match the PyTorch model).
        **kwargs: Passed through to ``fastembed`` ``TextCrossEncoder``.
    """

    DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None, **kwargs: Any) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            raise ImportError(
                "fastembed cross-encoder not available. Install/upgrade with: "
                "pip install -U fastembed"
            )
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = TextCrossEncoder(model_name=self.model_name, **kwargs)

    def predict(self, pairs):
        """Score (query, document) pairs. All pairs share the same query."""
        if not pairs:
            return []
        query = pairs[0][0]
        documents = [doc for _, doc in pairs]
        return list(self._model.rerank(query, documents))
