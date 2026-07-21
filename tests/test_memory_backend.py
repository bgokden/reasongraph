import json
import os
import tempfile

import pytest

from reasongraph._types import Node, Edge
from reasongraph.backends._memory import MemoryBackend


@pytest.fixture
async def backend():
    b = MemoryBackend()
    await b.initialize()
    yield b
    await b.close()


def _make_node(content: str, node_type: str = "text") -> Node:
    """Create a node with a simple deterministic embedding."""
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
    await backend.insert_nodes([node])

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
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content="The cat sat on the mat", type="text", embedding=emb),
        Node(content="Quantum physics is complex", type="text", embedding=emb),
    ])

    results = await backend.hybrid_search(emb, "cat", top_k=2)
    assert results[0]["content"] == "The cat sat on the mat"


@pytest.mark.asyncio
async def test_hybrid_search_pure_keyword(backend):
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content="flooding in the village", type="text", embedding=emb),
        Node(content="sunny day at the beach", type="text", embedding=emb),
    ])

    results = await backend.hybrid_search(emb, "flood", top_k=2, keyword_only=True)
    assert results[0]["content"] == "flooding in the village"


@pytest.mark.asyncio
async def test_hybrid_search_returns_correct_count(backend):
    nodes = [_make_node(f"node {i}") for i in range(10)]
    await backend.insert_nodes(nodes)

    results = await backend.hybrid_search(nodes[0].embedding, "node", top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_hybrid_search_short_query_keyword(backend):
    """Short keyword queries still work via substring matching."""
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content="an ox is strong", type="text", embedding=emb),
        Node(content="a cat is quick", type="text", embedding=emb),
    ])

    results = await backend.hybrid_search(emb, "ox", top_k=2, keyword_only=True)
    assert any("ox" in r["content"] for r in results)


# -- File persistence --

@pytest.mark.asyncio
async def test_file_persistence():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        # Write data
        b1 = MemoryBackend(file_path=path)
        await b1.initialize()
        await b1.insert_nodes([_make_node("persisted node")])
        await b1.insert_edges([Edge(from_content="persisted node", to_content="persisted node")])
        await b1.close()

        # Read it back
        b2 = MemoryBackend(file_path=path)
        await b2.initialize()
        nodes = await b2.get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0].content == "persisted node"
        edges = await b2.get_all_edges()
        assert len(edges) == 1
        await b2.close()
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_file_persistence_missing_file():
    """Initialize with a non-existent file path should not raise."""
    b = MemoryBackend(file_path="/tmp/nonexistent_reasongraph_test.json")
    await b.initialize()
    nodes = await b.get_all_nodes()
    assert len(nodes) == 0
    # Don't close (would create the file); just verify it loaded fine


@pytest.mark.asyncio
async def test_knn_search_empty(backend):
    """KNN search on empty backend returns empty list."""
    results = await backend.knn_search([0.5] * 384, top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_hybrid_search_empty(backend):
    """Hybrid search on empty backend returns empty list."""
    results = await backend.hybrid_search([0.5] * 384, "test", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_delete_stale_nodes(backend):
    from datetime import datetime, timedelta

    node = _make_node("old node")
    await backend.insert_nodes([node])

    # Manually set last_accessed to 60 days ago
    backend._nodes["old node"].last_accessed = datetime.now() - timedelta(days=60)

    deleted = await backend.delete_stale_nodes(30)
    assert deleted == 1
    assert len(await backend.get_all_nodes()) == 0


@pytest.mark.asyncio
async def test_delete_stale_removes_orphan_edges(backend):
    from datetime import datetime, timedelta

    nodes = [_make_node("fresh"), _make_node("stale")]
    await backend.insert_nodes(nodes)
    await backend.insert_edges([Edge(from_content="fresh", to_content="stale")])

    backend._nodes["stale"].last_accessed = datetime.now() - timedelta(days=60)

    await backend.delete_stale_nodes(30)
    edges = await backend.get_all_edges()
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_batch_dedup_within_insert(backend):
    """Inserting the same content twice in one batch should deduplicate."""
    node1 = _make_node("same")
    node2 = _make_node("same")
    await backend.insert_nodes([node1, node2])

    all_nodes = await backend.get_all_nodes()
    assert len(all_nodes) == 1


@pytest.mark.asyncio
async def test_delete_nodes(backend):
    await backend.insert_nodes([_make_node("keep"), _make_node("remove")])

    deleted = await backend.delete_nodes(["remove"])
    assert deleted == 1
    contents = {n.content for n in await backend.get_all_nodes()}
    assert contents == {"keep"}


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
async def test_delete_nodes_batch(backend):
    await backend.insert_nodes([_make_node("a"), _make_node("b"), _make_node("c")])

    deleted = await backend.delete_nodes(["a", "c", "missing"])
    assert deleted == 2
    contents = {n.content for n in await backend.get_all_nodes()}
    assert contents == {"b"}


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

    # A scoped KNN only seeds from nodes carrying that scope
    emb = _make_node("other fact").embedding
    res = await backend.knn_search(emb, top_k=10, scopes={"user-1"})
    assert all(r["content"] != "other fact" for r in res)
    assert "shared fact" in {r["content"] for r in res}

    # get_all_nodes can filter by scope too
    assert {n.content for n in await backend.get_all_nodes(scopes={"topic-econ"})} == {"shared fact"}
    assert {n.content for n in await backend.get_all_nodes(scopes={"user-2"})} == {"other fact"}


@pytest.mark.asyncio
async def test_get_scopes_bounded_lookup(backend):
    await backend.insert_nodes([
        _scoped_node("fact one", {"user-1", "topic-econ"}),
        _scoped_node("fact two", {"user-2"}),
        _make_node("unscoped fact"),  # no scopes
    ])

    got = await backend.get_scopes(["fact one", "fact two", "unscoped fact", "missing"])
    assert got["fact one"] == {"user-1", "topic-econ"}
    assert got["fact two"] == {"user-2"}
    # Unscoped nodes map to an empty set; missing contents are omitted entirely.
    assert got.get("unscoped fact", set()) == set()
    assert "missing" not in got
    assert await backend.get_scopes([]) == {}


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
