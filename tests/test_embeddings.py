from datetime import datetime

import numpy as np
import pytest

from reasongraph._embeddings import EmbeddingManager, Embedder

def _stable_hash(text):
    """Process-independent 64-bit hash (Python's hash() is randomized per run,
    which made fake embeddings and therefore test outcomes flaky)."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")



def _fake_vec(text: str) -> list[float]:
    """Deterministic 384-dim vector without loading a real model."""
    h = _stable_hash(text)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def test_callable_embedder():
    """A plain callable that encodes str or list[str] is accepted."""
    def embed(x):
        return [_fake_vec(t) for t in x] if isinstance(x, list) else _fake_vec(x)

    em = EmbeddingManager(embed_model=embed)

    v = em.encode("hello")
    assert isinstance(v, list) and len(v) == 384 and isinstance(v[0], float)

    batch = em.encode_batch(["a", "b"])
    assert isinstance(batch, list) and len(batch) == 2
    assert all(len(row) == 384 for row in batch)


def test_object_with_encode_returning_numpy():
    """An object whose encode() returns numpy is normalized to plain lists."""
    class NumpyEmbedder:
        def encode(self, x):
            if isinstance(x, list):
                return np.asarray([_fake_vec(t) for t in x], dtype=np.float32)
            return np.asarray(_fake_vec(x), dtype=np.float32)

    em = EmbeddingManager(embed_model=NumpyEmbedder())

    v = em.encode("hi")
    assert type(v) is list and type(v[0]) is float  # numpy fully unwrapped

    batch = em.encode_batch(["a", "b"])
    assert type(batch) is list and type(batch[0]) is list and type(batch[0][0]) is float


def test_object_with_encode_returning_lists():
    """An encoder returning plain python lists needs no conversion."""
    class ListEmbedder:
        def encode(self, x):
            return [_fake_vec(t) for t in x] if isinstance(x, list) else _fake_vec(x)

    em = EmbeddingManager(embed_model=ListEmbedder())
    assert em.encode("x") == _fake_vec("x")
    assert em.encode_batch(["x"]) == [_fake_vec("x")]


def test_invalid_embedder_type_raises():
    with pytest.raises(TypeError):
        EmbeddingManager(embed_model=123)


def test_callable_satisfies_embedder_protocol_object():
    """An object with encode() is recognized by the runtime-checkable Protocol."""
    class Enc:
        def encode(self, x):
            return _fake_vec(x)

    assert isinstance(Enc(), Embedder)


def test_apply_recency_boosts_newer_when_relevance_tied():
    """With equal relevance, the newer item scores higher."""
    scores = [0.0, 0.0]
    items = [
        {"content": "old", "created_at": datetime(2020, 1, 1).isoformat()},
        {"content": "new", "created_at": datetime(2026, 1, 1).isoformat()},
    ]
    blended = EmbeddingManager._apply_recency(scores, items, weight=1.0)
    assert blended[1] > blended[0]


def test_apply_recency_partial_weight_blends():
    """A strongly relevant old item can still beat a weakly relevant new one."""
    scores = [5.0, 0.0]  # old is far more relevant
    items = [
        {"content": "old", "created_at": datetime(2020, 1, 1).isoformat()},
        {"content": "new", "created_at": datetime(2026, 1, 1).isoformat()},
    ]
    blended = EmbeddingManager._apply_recency(scores, items, weight=0.3)
    assert blended[0] > blended[1]


def test_apply_recency_missing_created_at_treated_as_oldest():
    scores = [0.0, 0.0]
    items = [
        {"content": "no-date"},
        {"content": "dated", "created_at": datetime(2026, 1, 1).isoformat()},
    ]
    blended = EmbeddingManager._apply_recency(scores, items, weight=1.0)
    assert blended[1] > blended[0]
