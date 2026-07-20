import pytest

from reasongraph._types import Node, Edge
from reasongraph.backends._sqlite import SqliteBackend, _escape_fts5


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

    # Search using the embedding of the first node -- it should come back first
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


# -- Hybrid search --

@pytest.mark.asyncio
async def test_hybrid_search_boosts_keyword_match(backend):
    # Two nodes with identical fake embeddings but different text
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content="The cat sat on the mat", type="text", embedding=emb),
        Node(content="Quantum physics is complex", type="text", embedding=emb),
    ])

    # Hybrid search for "cat" -- trigram match should boost the cat node via RRF
    results = await backend.hybrid_search(emb, "cat", top_k=2)
    assert results[0]["content"] == "The cat sat on the mat"


@pytest.mark.asyncio
async def test_hybrid_search_pure_keyword(backend):
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content="flooding in the village", type="text", embedding=emb),
        Node(content="sunny day at the beach", type="text", embedding=emb),
    ])

    # keyword_only=True ranks by FTS5 trigram match
    results = await backend.hybrid_search(emb, "flood", top_k=2, keyword_only=True)
    assert results[0]["content"] == "flooding in the village"


@pytest.mark.asyncio
async def test_hybrid_search_returns_correct_count(backend):
    nodes = [_make_node(f"node {i}") for i in range(10)]
    await backend.insert_nodes(nodes)

    results = await backend.hybrid_search(nodes[0].embedding, "node", top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_hybrid_search_short_query_keyword_fallback(backend):
    """Queries shorter than 3 chars can't use FTS5 trigram; verify LIKE fallback."""
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content="an ox is strong", type="text", embedding=emb),
        Node(content="a cat is quick", type="text", embedding=emb),
    ])

    results = await backend.hybrid_search(emb, "ox", top_k=2, keyword_only=True)
    assert any("ox" in r["content"] for r in results)


@pytest.mark.asyncio
async def test_hybrid_search_short_query_hybrid_fallback(backend):
    """Hybrid mode with < 3 char query falls back to embedding-only search."""
    nodes = [_make_node("hello"), _make_node("world")]
    await backend.insert_nodes(nodes)

    results = await backend.hybrid_search(nodes[0].embedding, "hi", top_k=2)
    assert len(results) == 2
    # Should return embedding-ranked results (same as knn_search)
    assert results[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_fts5_special_characters(backend):
    """FTS5 special characters in queries should be safely escaped."""
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content='value is "important" here', type="text", embedding=emb),
        Node(content="nothing special", type="text", embedding=emb),
    ])

    # Query with double quotes -- should not break FTS5 MATCH
    results = await backend.hybrid_search(emb, '"important"', top_k=2, keyword_only=True)
    assert results[0]["content"] == 'value is "important" here'


def test_escape_fts5_basic():
    assert _escape_fts5("hello") == '"hello"'


def test_escape_fts5_quotes():
    assert _escape_fts5('say "hi"') == '"say ""hi"""'


@pytest.mark.asyncio
async def test_delete_nodes(backend):
    await backend.insert_nodes([_make_node("keep"), _make_node("remove")])

    deleted = await backend.delete_nodes(["remove"])
    assert deleted == 1
    contents = {n.content for n in await backend.get_all_nodes()}
    assert contents == {"keep"}


@pytest.mark.asyncio
async def test_delete_nodes_purges_vec_and_fts(backend):
    """Deleted nodes must vanish from the vec_nodes and fts_nodes shadow tables."""
    keep = _make_node("keep this around")
    remove = _make_node("remove this now")
    await backend.insert_nodes([keep, remove])

    await backend.delete_nodes(["remove this now"])

    # Vector search must not surface the deleted node, even queried with its own embedding
    knn = await backend.knn_search(remove.embedding, top_k=5)
    assert all(r["content"] != "remove this now" for r in knn)

    # Trigram keyword search must not surface it either
    kw = await backend.hybrid_search(
        remove.embedding, "remove this now", top_k=5, keyword_only=True
    )
    assert all(r["content"] != "remove this now" for r in kw)


@pytest.mark.asyncio
async def test_delete_nodes_removes_incident_edges(backend):
    await backend.insert_nodes([
        _make_node("the fact", "text"),
        _make_node("Amsterdam", "entity"),
    ])
    await backend.insert_edges([Edge(from_content="Amsterdam", to_content="the fact")])

    await backend.delete_nodes(["the fact"])
    assert await backend.get_all_edges() == []
    # The shared entity node survives; only its edge to the deleted text is gone
    contents = {n.content for n in await backend.get_all_nodes()}
    assert contents == {"Amsterdam"}


@pytest.mark.asyncio
async def test_delete_nodes_missing_content_is_noop(backend):
    await backend.insert_nodes([_make_node("present")])

    deleted = await backend.delete_nodes(["absent"])
    assert deleted == 0
    assert len(await backend.get_all_nodes()) == 1


@pytest.mark.asyncio
async def test_delete_nodes_empty_list(backend):
    await backend.insert_nodes([_make_node("present")])

    assert await backend.delete_nodes([]) == 0
    assert len(await backend.get_all_nodes()) == 1


@pytest.mark.asyncio
async def test_get_created_at(backend):
    from datetime import datetime

    await backend.insert_nodes([_make_node("a"), _make_node("b")])

    got = await backend.get_created_at(["a", "b", "missing"])
    assert set(got.keys()) == {"a", "b"}  # missing content omitted
    datetime.fromisoformat(got["a"])  # values are parseable ISO strings
    assert await backend.get_created_at([]) == {}


def _scoped_node(content: str, scopes: set[str], node_type: str = "text") -> Node:
    node = _make_node(content, node_type)
    node.scopes = set(scopes)
    return node


@pytest.mark.asyncio
async def test_scopes_multi_label_union_and_seed_filter(backend):
    await backend.insert_nodes([_scoped_node("shared fact", {"user-1", "topic-econ"})])
    # Re-adding the same content under a new scope unions the tags
    await backend.insert_nodes([_scoped_node("shared fact", {"session-9"})])
    await backend.insert_nodes([_scoped_node("other fact", {"user-2"})])

    by_content = {n.content: n for n in await backend.get_all_nodes()}
    assert by_content["shared fact"].scopes == {"user-1", "topic-econ", "session-9"}

    # Scoped KNN (vec_distance_cosine path) only seeds from that scope
    emb = _make_node("other fact").embedding
    res = await backend.knn_search(emb, top_k=10, scopes={"user-1"})
    assert all(r["content"] != "other fact" for r in res)
    assert "shared fact" in {r["content"] for r in res}

    # Scoped keyword and hybrid search also stay within the scope
    kw = await backend.hybrid_search(emb, "other fact", 10, keyword_only=True, scopes={"user-1"})
    assert all(r["content"] != "other fact" for r in kw)
    hy = await backend.hybrid_search(emb, "shared", 10, scopes={"user-1"})
    assert all(r["content"] != "other fact" for r in hy)

    # Scoped get_all_nodes
    assert {n.content for n in await backend.get_all_nodes(scopes={"topic-econ"})} == {"shared fact"}

    # Deleting a node cascades its scope rows
    await backend.delete_nodes(["shared fact"])
    assert await backend.get_all_nodes(scopes={"user-1"}) == []


@pytest.mark.asyncio
async def test_insert_nodes_does_not_mutate_caller_scopes(backend):
    a = _scoped_node("dup", {"s-a"})
    b = _scoped_node("dup", {"s-b"})
    await backend.insert_nodes([a, b])

    # The caller's Node objects are left untouched
    assert a.scopes == {"s-a"}
    assert b.scopes == {"s-b"}
    # The stored node holds the union
    stored = {n.content: n for n in await backend.get_all_nodes()}["dup"]
    assert stored.scopes == {"s-a", "s-b"}

    # Re-inserting the same content under a new scope also must not mutate the caller
    c = _scoped_node("dup", {"s-c"})
    await backend.insert_nodes([c])
    assert c.scopes == {"s-c"}
    stored = {n.content: n for n in await backend.get_all_nodes()}["dup"]
    assert stored.scopes == {"s-a", "s-b", "s-c"}
