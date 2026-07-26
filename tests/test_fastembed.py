"""Adapter tests for the pure-ONNX fastembed embedder / reranker.

Skipped when fastembed is not installed or a model cannot be fetched (offline),
so the suite still runs anywhere. When available they exercise the real ONNX
path and its integration with ReasonGraph's pluggable seams.
"""

import pytest

pytest.importorskip("fastembed")

from reasongraph._fastembed import FastEmbedEmbedder, FastEmbedReranker


@pytest.fixture(scope="module")
def embedder():
    try:
        return FastEmbedEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:  # noqa: BLE001 - offline / download failure
        pytest.skip(f"fastembed embedder unavailable: {e}")


def test_encode_single_and_batch(embedder):
    v = embedder.encode("hello world")
    assert isinstance(v, list) and isinstance(v[0], float) and len(v) == 384

    batch = embedder.encode(["a", "b"])
    assert isinstance(batch, list) and len(batch) == 2
    assert all(len(row) == 384 for row in batch)


def test_plugs_into_reasongraph(embedder):
    from reasongraph import ReasonGraph
    from reasongraph.backends._memory import MemoryBackend

    g = ReasonGraph(backend=MemoryBackend(), embed_model=embedder)
    g.embeddings.rerank = lambda q, results, top_k, recency_weight=0.0: results[:top_k]
    g.initialize_sync()
    try:
        g.add_text_sync("Socrates was a philosopher.", extractor=lambda t: ["Socrates"])
        results = g.query_sync("Who was Socrates?", top_k=3, hops=1)
        assert "Socrates was a philosopher." in results
    finally:
        g.close_sync()


def test_reranker_predict():
    try:
        rr = FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"fastembed reranker unavailable: {e}")

    scores = rr.predict([
        ("what is the capital of France?", "Paris is the capital of France."),
        ("what is the capital of France?", "Bananas are yellow."),
    ])
    assert len(scores) == 2
    assert scores[0] > scores[1]  # the relevant document scores higher
    assert rr.predict([]) == []
