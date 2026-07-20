import numpy as np
import pytest

from reasongraph._embeddings import EmbeddingManager, Embedder


def _fake_vec(text: str) -> list[float]:
    """Deterministic 384-dim vector without loading a real model."""
    h = hash(text)
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
