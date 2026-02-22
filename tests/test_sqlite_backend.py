import pytest

from reasongraph._types import Node, Edge
from reasongraph.backends._sqlite import SqliteBackend, _trigrams, _trigram_similarity


@pytest.fixture
async def backend():
    b = SqliteBackend(":memory:")
    await b.initialize()
    yield b
    await b.close()


def _make_node(content: str, node_type: str = "text") -> Node:
    """Create a node with a simple deterministic embedding."""
    # Use hash-based pseudo-embedding for testing (avoids loading real model)
    h = hash(content)
    embedding = [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]
    return Node(content=content, type=node_type, embedding=embedding)


@pytest.mark.asyncio
async def test_insert_and_get_nodes(backend):
    nodes = [_make_node("Alpha"), _make_node("Beta")]
    await backend.insert_nodes(nodes)

    all_nodes = await backend.get_all_nodes()
    assert len(all_nodes) == 2
    contents = {n.content for n in all_nodes}
    assert contents == {"Alpha", "Beta"}


@pytest.mark.asyncio
async def test_insert_and_get_edges(backend):
    nodes = [_make_node("A"), _make_node("B"), _make_node("C")]
    await backend.insert_nodes(nodes)

    edges = [Edge(from_content="A", to_content="B"), Edge(from_content="B", to_content="C")]
    await backend.insert_edges(edges)

    all_edges = await backend.get_all_edges()
    assert len(all_edges) == 2


@pytest.mark.asyncio
async def test_knn_search(backend):
    nodes = [
        _make_node("hello world"),
        _make_node("goodbye world"),
        _make_node("something else entirely"),
    ]
    await backend.insert_nodes(nodes)

    # Search using the embedding of the first node — it should come back first
    results = await backend.knn_search(nodes[0].embedding, top_k=2)
    assert len(results) == 2
    assert results[0]["content"] == "hello world"


@pytest.mark.asyncio
async def test_get_neighbors(backend):
    nodes = [_make_node("X"), _make_node("Y"), _make_node("Z")]
    await backend.insert_nodes(nodes)

    edges = [Edge(from_content="X", to_content="Y"), Edge(from_content="Z", to_content="X")]
    await backend.insert_edges(edges)

    neighbors = await backend.get_neighbors("X")
    neighbor_contents = {n["content"] for n in neighbors}
    assert neighbor_contents == {"Y", "Z"}


@pytest.mark.asyncio
async def test_upsert_node(backend):
    node = _make_node("duplicate")
    await backend.insert_nodes([node])
    await backend.insert_nodes([node])  # Should not raise

    all_nodes = await backend.get_all_nodes()
    assert len(all_nodes) == 1


@pytest.mark.asyncio
async def test_duplicate_edge_ignored(backend):
    nodes = [_make_node("P"), _make_node("Q")]
    await backend.insert_nodes(nodes)

    edge = Edge(from_content="P", to_content="Q")
    await backend.insert_edges([edge])
    await backend.insert_edges([edge])

    all_edges = await backend.get_all_edges()
    assert len(all_edges) == 1


@pytest.mark.asyncio
async def test_node_without_embedding_raises(backend):
    node = Node(content="no embedding", type="text", embedding=None)
    with pytest.raises(ValueError, match="no embedding"):
        await backend.insert_nodes([node])


# -- Trigram utilities --

def test_trigrams_basic():
    trgms = _trigrams("cat")
    # With padding "  cat  " -> {"  c", " ca", "cat", "at ", "t  "}
    assert "cat" in trgms
    assert " ca" in trgms
    assert "at " in trgms


def test_trigram_similarity_identical():
    assert _trigram_similarity("flooding", "flooding") == 1.0


def test_trigram_similarity_partial():
    sim = _trigram_similarity("flood", "flooding")
    assert sim > 0.3  # Significant overlap expected


def test_trigram_similarity_unrelated():
    sim = _trigram_similarity("apple", "quantum")
    assert sim < 0.1


def test_trigram_similarity_empty():
    assert _trigram_similarity("", "") == 0.0


# -- Hybrid search --

@pytest.mark.asyncio
async def test_hybrid_search_boosts_keyword_match(backend):
    # Two nodes with identical fake embeddings but different text
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content="The cat sat on the mat", type="text", embedding=emb),
        Node(content="Quantum physics is complex", type="text", embedding=emb),
    ])

    # Hybrid search for "cat" — trigram match should boost the cat node via RRF
    results = await backend.hybrid_search(emb, "cat", top_k=2)
    assert results[0]["content"] == "The cat sat on the mat"


@pytest.mark.asyncio
async def test_hybrid_search_pure_keyword(backend):
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content="flooding in the village", type="text", embedding=emb),
        Node(content="sunny day at the beach", type="text", embedding=emb),
    ])

    # keyword_only=True ranks by trigram similarity only
    results = await backend.hybrid_search(emb, "flood", top_k=2, keyword_only=True)
    assert results[0]["content"] == "flooding in the village"


@pytest.mark.asyncio
async def test_hybrid_search_returns_correct_count(backend):
    nodes = [_make_node(f"node {i}") for i in range(10)]
    await backend.insert_nodes(nodes)

    results = await backend.hybrid_search(nodes[0].embedding, "node", top_k=3)
    assert len(results) == 3
