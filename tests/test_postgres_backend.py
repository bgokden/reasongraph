"""Integration tests for PostgresBackend against a real local PostgreSQL.

Skipped automatically when psycopg / pgvector / a reachable server with the
``vector`` and ``pg_trgm`` extensions are not available, so the suite still
runs anywhere. When a local server is present it exercises the real SQL.
"""

import os

import pytest

from reasongraph._types import Node, Edge

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("pgvector")

from reasongraph.backends._postgres import PostgresBackend

# Local default: a Unix-socket superuser connection. CI sets REASONGRAPH_TEST_PG_ADMIN
# to the service container; the test database is created next to the admin one.
ADMIN_DSN = os.environ.get("REASONGRAPH_TEST_PG_ADMIN", "postgresql:///postgres")
TEST_DB = "reasongraph_pytest"


def _test_dsn() -> str:
    """ADMIN_DSN with the database name swapped for TEST_DB."""
    base, _, _db = ADMIN_DSN.rpartition("/")
    return f"{base}/{TEST_DB}"


def _make_node(content: str, node_type: str = "text", scopes: set[str] | None = None) -> Node:
    h = hash(content)
    embedding = [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]
    node = Node(content=content, type=node_type, embedding=embedding)
    if scopes:
        node.scopes = set(scopes)
    return node


async def _admin():
    return await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True)


@pytest.fixture
async def backend():
    try:
        conn = await _admin()
    except Exception as e:  # noqa: BLE001 - any connect failure means no server
        pytest.skip(f"no local postgres: {e}")

    try:
        await conn.execute(
            "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
        )
        if await (await conn.execute(
            "SELECT count(*) FROM pg_available_extensions WHERE name IN ('vector','pg_trgm')"
        )).fetchone() != (2,):
            await conn.close()
            pytest.skip("vector/pg_trgm extensions not available")
        # Drop any lingering connections, then recreate a clean test database
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
            (TEST_DB,),
        )
        await conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        await conn.execute(f"CREATE DATABASE {TEST_DB}")
    except Exception as e:  # noqa: BLE001
        await conn.close()
        pytest.skip(f"cannot provision test db: {e}")
    await conn.close()

    b = PostgresBackend(_test_dsn())
    try:
        await b.initialize()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot initialize postgres backend: {e}")
    yield b
    await b.close()

    conn = await _admin()
    try:
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
            (TEST_DB,),
        )
        await conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_insert_and_query(backend):
    await backend.insert_nodes([
        _make_node("hello world"),
        _make_node("goodbye world"),
        _make_node("something else entirely"),
    ])
    all_nodes = await backend.get_all_nodes()
    assert {n.content for n in all_nodes} == {"hello world", "goodbye world", "something else entirely"}

    knn = await backend.knn_search(_make_node("hello world").embedding, top_k=2)
    assert knn[0]["content"] == "hello world"


@pytest.mark.asyncio
async def test_vector_index_created(backend):
    # HNSW cosine index accelerates knn_search/hybrid_search (pgvector >= 0.5.0,
    # which the test server has). Guards against the index silently going missing.
    pool = await backend._get_pool()
    async with pool.connection() as conn:
        row = await (await conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'nodes_embedding_hnsw'"
        )).fetchone()
    assert row is not None, "HNSW vector index should be created on init"
    assert "hnsw" in row[0].lower() and "vector_cosine_ops" in row[0].lower()


@pytest.mark.asyncio
async def test_edges_neighbors_and_cascade_delete(backend):
    await backend.insert_nodes([_make_node("A"), _make_node("B", "entity"), _make_node("C")])
    await backend.insert_edges([
        Edge(from_content="B", to_content="A"),
        Edge(from_content="B", to_content="C"),
    ])
    # duplicate edge ignored (ON CONFLICT DO NOTHING)
    await backend.insert_edges([Edge(from_content="B", to_content="A")])
    assert len(await backend.get_all_edges()) == 2

    neighbors = {n["content"] for n in await backend.get_neighbors("B")}
    assert neighbors == {"A", "C"}

    # Scoped traversal (isolation): only neighbors carrying the scope are returned.
    await backend.insert_nodes([_make_node("A", scopes={"t1"}), _make_node("C", scopes={"t2"})])
    scoped = {n["content"] for n in await backend.get_neighbors("B", scopes={"t1"})}
    assert scoped == {"A"}  # C is out of scope t1

    # deleting B cascades its incident edges
    assert await backend.delete_nodes(["B"]) == 1
    assert await backend.get_all_edges() == []


@pytest.mark.asyncio
async def test_temporal_validity(backend):
    from datetime import datetime

    await backend.insert_nodes([_make_node("current"), _make_node("retired")])
    assert await backend.get_validity(["current", "retired"]) == {"current": None, "retired": None}

    when = datetime(2022, 1, 1)
    await backend.set_invalid(["retired"], when)
    validity = await backend.get_validity(["current", "retired", "missing"])
    assert validity["current"] is None
    assert validity["retired"] == when.isoformat()
    assert "missing" not in validity
    # invalid_at round-trips through get_all_nodes
    by_content = {n.content: n for n in await backend.get_all_nodes()}
    assert by_content["retired"].invalid_at == when
    assert by_content["current"].invalid_at is None

    # re-asserting a retired fact revives it (upsert clears invalid_at)
    await backend.insert_nodes([_make_node("retired")])
    assert (await backend.get_validity(["retired"]))["retired"] is None


@pytest.mark.asyncio
async def test_hybrid_search(backend):
    emb = [0.5] * 384
    await backend.insert_nodes([
        Node(content="the cat sat on the mat", type="text", embedding=emb),
        Node(content="quantum physics is complex", type="text", embedding=emb),
    ])
    kw = await backend.hybrid_search(emb, "cat", top_k=2, keyword_only=True)
    assert kw[0]["content"] == "the cat sat on the mat"

    hy = await backend.hybrid_search(emb, "cat", top_k=2)
    assert hy[0]["content"] == "the cat sat on the mat"


@pytest.mark.asyncio
async def test_scopes(backend):
    await backend.insert_nodes([_make_node("shared fact", scopes={"user-1", "topic-econ"})])
    await backend.insert_nodes([_make_node("shared fact", scopes={"session-9"})])  # union
    await backend.insert_nodes([_make_node("other fact", scopes={"user-2"})])

    by_content = {n.content: n for n in await backend.get_all_nodes()}
    assert by_content["shared fact"].scopes == {"user-1", "topic-econ", "session-9"}

    emb = _make_node("other fact").embedding
    res = await backend.knn_search(emb, top_k=10, scopes={"user-1"})
    assert all(r["content"] != "other fact" for r in res)
    assert "shared fact" in {r["content"] for r in res}

    kw = await backend.hybrid_search(emb, "other", 10, keyword_only=True, scopes={"user-1"})
    assert all(r["content"] != "other fact" for r in kw)

    # Scoped RRF hybrid (keyword_only=False) exercises the two-scope-placeholder CTE
    hy = await backend.hybrid_search(emb, "shared", 10, scopes={"user-1"})
    assert all(r["content"] != "other fact" for r in hy)
    assert "shared fact" in {r["content"] for r in hy}

    assert {n.content for n in await backend.get_all_nodes(scopes={"topic-econ"})} == {"shared fact"}

    # Bounded scope lookup for a known set of contents
    got = await backend.get_scopes(["shared fact", "other fact", "missing"])
    assert got["shared fact"] == {"user-1", "topic-econ", "session-9"}
    assert got["other fact"] == {"user-2"}
    assert "missing" not in got
    assert await backend.get_scopes([]) == {}

    # deleting a node cascades its node_scopes rows
    await backend.delete_nodes(["shared fact"])
    assert await backend.get_all_nodes(scopes={"user-1"}) == []


@pytest.mark.asyncio
async def test_causal_edges_and_relations(backend):
    await backend.insert_nodes([
        _make_node("Heavy rainfall caused flooding.", "text"),
        _make_node("heavy rainfall", "entity"),
        _make_node("flooding", "entity"),
    ])
    await backend.insert_edges([
        Edge(from_content="heavy rainfall", to_content="flooding", label="causes"),
        Edge(from_content="heavy rainfall", to_content="Heavy rainfall caused flooding."),
        Edge(from_content="flooding", to_content="Heavy rainfall caused flooding."),
    ])

    # Edge.label round-trips
    labels = {(e.from_content, e.to_content): e.label for e in await backend.get_all_edges()}
    assert labels[("heavy rainfall", "flooding")] == "causes"
    assert labels[("flooding", "Heavy rainfall caused flooding.")] is None

    # get_neighbors exposes label + direction
    neigh = {n["content"]: n for n in await backend.get_neighbors("heavy rainfall")}
    assert neigh["flooding"]["label"] == "causes"
    assert neigh["flooding"]["direction"] == "out"

    # get_causal_relations returns the fact's directed pair
    rels = await backend.get_causal_relations(["Heavy rainfall caused flooding."])
    assert rels["Heavy rainfall caused flooding."] == [{"cause": "heavy rainfall", "effect": "flooding"}]
    assert await backend.get_causal_relations([]) == {}
