"""Counterfactual ``what_if`` over the directed 'causes' DAG (model-free).

Prunes one fact hypothetically (no graph mutation) and re-walks causal
reachability to report which downstream spans collapse (lose all support) vs
survive via an alternate path. Uses the same fake hash embeddings + fake causal
extractor as the trace tests, parametrized over Memory/SQLite (and Postgres when
a local server is present) to prove three-backend parity.
"""

import datetime

import pytest

from reasongraph.graph import ReasonGraph
from reasongraph.backends._memory import MemoryBackend
from reasongraph.backends._sqlite import SqliteBackend


def _fake_encode(text):
    h = hash(text)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def _fake_embed(x):
    return [_fake_encode(t) for t in x] if isinstance(x, list) else _fake_encode(x)


def _no_rerank(query, results, top_k, recency_weight=0.0):
    return results[:top_k]


LINEAR = {
    "Heavy rainfall caused flooding.": {"cause": "rainfall", "effect": "flooding"},
    "Flooding caused power outages.": {"cause": "flooding", "effect": "power outages"},
    "Power outages caused hospital disruptions.": {"cause": "power outages", "effect": "hospital disruptions"},
}

# Two branches (a, b) both reach span x -> pruning one branch leaves x reachable.
DIAMOND = {
    "Root caused branch A.": {"cause": "root", "effect": "a"},
    "Root caused branch B.": {"cause": "root", "effect": "b"},
    "Branch A caused merge X.": {"cause": "a", "effect": "x"},
    "Branch B caused merge X.": {"cause": "b", "effect": "x"},
}


def _make_causal(rels):
    def _fake_causal(texts):
        return [
            {"causal": bool(rels.get(t)), "relations": [rels[t]] if rels.get(t) else []}
            for t in texts
        ]
    return _fake_causal


async def _build(backend, rels, *, texts=None, scopes=None):
    g = ReasonGraph(backend=backend, embed_model=_fake_embed, causal_extractor=_make_causal(rels))
    g.embeddings.rerank = _no_rerank
    await g.initialize()
    await g.add_texts(list(texts if texts is not None else rels), extractor=lambda t: [], scopes=scopes)
    return g


# Fresh backend per test so parametrized cases never share accumulated state.
BACKENDS = [
    pytest.param(lambda: MemoryBackend(), id="memory"),
    pytest.param(lambda: SqliteBackend(":memory:"), id="sqlite"),
]


@pytest.mark.parametrize("make_backend", BACKENDS)
@pytest.mark.asyncio
async def test_what_if_linear_collapse(make_backend):
    g = await _build(make_backend(), LINEAR)
    try:
        out = await g.what_if(
            "Flooding caused power outages.", origin="Heavy rainfall caused flooding.",
        )
        assert out["pruned"] == "Flooding caused power outages."
        assert out["origin"] == "Heavy rainfall caused flooding."
        assert out["pruned_edges"] == [{"cause": "flooding", "effect": "power outages"}]
        spans = {c["span"] for c in out["collapsed"]}
        # everything downstream of the cut loses its only path
        assert spans == {"power outages", "hospital disruptions"}
        # the upstream span is unaffected
        assert "flooding" not in spans
        for c in out["collapsed"]:
            assert c["fact"] in LINEAR         # cited to a real asserting fact
            assert isinstance(c["scopes"], list)
            assert isinstance(c["depth"], int)
        assert out["survived"] == []           # nothing rescued: linear chain
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_what_if_origin_defaults_to_pruned():
    g = await _build(MemoryBackend(), LINEAR)
    try:
        out = await g.what_if("Heavy rainfall caused flooding.")
        assert out["origin"] == "Heavy rainfall caused flooding."
        spans = {c["span"] for c in out["collapsed"]}
        assert spans == {"flooding", "power outages", "hospital disruptions"}
        assert out["survived"] == []
    finally:
        await g.close()


@pytest.mark.parametrize("make_backend", BACKENDS)
@pytest.mark.asyncio
async def test_what_if_redundant_edge_is_noop(make_backend):
    # A second fact asserts the SAME (flooding, power outages) edge, so pruning
    # one leaves the edge supported: nothing is solely-pruned, nothing collapses.
    rels = dict(LINEAR)
    rels["Flood water shorted the substation causing power outages."] = {
        "cause": "flooding", "effect": "power outages",
    }
    g = await _build(make_backend(), rels)
    try:
        out = await g.what_if(
            "Flooding caused power outages.", origin="Heavy rainfall caused flooding.",
        )
        assert out["pruned_edges"] == []
        assert out["collapsed"] == []
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_what_if_diamond_alternate_path_rescues():
    # x is reachable via a and via b; pruning the a->x fact still leaves x reachable
    # via b -> x SURVIVES (rescued), and nothing collapses.
    g = await _build(MemoryBackend(), DIAMOND)
    try:
        out = await g.what_if("Branch A caused merge X.", origin="Root caused branch A.")
        assert out["pruned_edges"] == [{"cause": "a", "effect": "x"}]
        assert out["survived"] == ["x"]
        assert {c["span"] for c in out["collapsed"]} == set()
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_what_if_non_causal_fact_is_noop():
    g = await _build(
        MemoryBackend(), LINEAR, texts=list(LINEAR) + ["The sky is blue."],
    )
    try:
        out = await g.what_if("The sky is blue.")
        assert out["pruned"] == "The sky is blue."
        assert out["pruned_edges"] == []
        assert out["collapsed"] == []
        assert out["survived"] == []
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_what_if_empty_graph():
    g = ReasonGraph(
        backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_make_causal({}),
    )
    g.embeddings.rerank = _no_rerank
    await g.initialize()
    try:
        out = await g.what_if("anything")
        assert out == {
            "pruned": None, "origin": None, "pruned_edges": [],
            "collapsed": [], "survived": [],
        }
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_what_if_direction_causes_mirror():
    g = await _build(MemoryBackend(), LINEAR)
    try:
        out = await g.what_if(
            "Flooding caused power outages.",
            origin="Power outages caused hospital disruptions.",
            direction="causes",
        )
        # backward walk: pruning the flooding->power outages edge orphans upstream 'flooding'
        assert out["pruned_edges"] == [{"cause": "flooding", "effect": "power outages"}]
        assert "flooding" in {c["span"] for c in out["collapsed"]}
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_what_if_superseded_supporter_does_not_rescue():
    rels = dict(LINEAR)
    rels["Flood water shorted the substation causing power outages."] = {
        "cause": "flooding", "effect": "power outages",
    }
    g = await _build(MemoryBackend(), rels)
    try:
        g.conflict_resolver = object()  # any non-None resolver enables retirement filtering
        await g.backend.set_invalid(
            ["Flood water shorted the substation causing power outages."],
            datetime.datetime.now(),
        )
        # the live supporter is pruned; the retired duplicate must NOT rescue the effect
        out = await g.what_if(
            "Flooding caused power outages.", origin="Heavy rainfall caused flooding.",
        )
        assert "power outages" in {c["span"] for c in out["collapsed"]}
        # include_superseded lets the retired duplicate count as support -> nothing collapses
        keep = await g.what_if(
            "Flooding caused power outages.", origin="Heavy rainfall caused flooding.",
            include_superseded=True,
        )
        assert keep["collapsed"] == []
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_what_if_max_depth_bounds_symmetrically():
    g = await _build(MemoryBackend(), LINEAR)
    try:
        # power outages is beyond depth 1 in the baseline walk; the symmetric bound
        # must not falsely report it collapsed.
        out = await g.what_if(
            "Flooding caused power outages.", origin="Heavy rainfall caused flooding.",
            max_depth=1,
        )
        assert {c["span"] for c in out["collapsed"]} == set()
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_what_if_isolate_restricts_walk_to_scope():
    rels = {
        "Rainfall caused flooding.": {"cause": "rainfall", "effect": "flooding"},
        "Flooding caused evacuation.": {"cause": "flooding", "effect": "evacuation"},
    }
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed,
                    causal_extractor=_make_causal(rels))
    g.embeddings.rerank = _no_rerank
    await g.initialize()
    # 'flooding' is shared across scopes; 'evacuation' lives only in user-b.
    await g.add_texts(["Rainfall caused flooding."], extractor=lambda t: [], scopes={"user-a"})
    await g.add_texts(["Flooding caused evacuation."], extractor=lambda t: [], scopes={"user-b"})
    try:
        # Shared traversal crosses the bridge -> evacuation collapses too.
        shared = await g.what_if("Rainfall caused flooding.", scopes={"user-a"})
        assert "evacuation" in {c["span"] for c in shared["collapsed"]}
        # Isolated traversal stays in user-a -> the cross-scope span is unreachable.
        isolated = await g.what_if("Rainfall caused flooding.", scopes={"user-a"}, isolate=True)
        collapsed = {c["span"] for c in isolated["collapsed"]}
        assert "flooding" in collapsed
        assert "evacuation" not in collapsed
    finally:
        await g.close()


def test_what_if_sync_twin():
    g = ReasonGraph(
        backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_make_causal(LINEAR),
    )
    g.embeddings.rerank = _no_rerank
    g.initialize_sync()
    g.add_texts_sync(list(LINEAR), extractor=lambda t: [])
    try:
        out = g.what_if_sync(
            "Flooding caused power outages.", origin="Heavy rainfall caused flooding.",
        )
        assert out["pruned_edges"] == [{"cause": "flooding", "effect": "power outages"}]
        assert {c["span"] for c in out["collapsed"]} == {"power outages", "hospital disruptions"}
    finally:
        g.close_sync()


@pytest.fixture
async def pg_backend():
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("psycopg_pool")
    pytest.importorskip("pgvector")
    from reasongraph.backends._postgres import PostgresBackend

    admin_dsn = "postgresql:///postgres"
    test_db = "reasongraph_whatif_pytest"
    try:
        conn = await psycopg.AsyncConnection.connect(admin_dsn, autocommit=True)
    except Exception as e:  # noqa: BLE001 - any connect failure means no server
        pytest.skip(f"no local postgres: {e}")
    try:
        if await (await conn.execute(
            "SELECT count(*) FROM pg_available_extensions WHERE name IN ('vector','pg_trgm')"
        )).fetchone() != (2,):
            await conn.close()
            pytest.skip("vector/pg_trgm extensions not available")
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
            (test_db,),
        )
        await conn.execute(f"DROP DATABASE IF EXISTS {test_db}")
        await conn.execute(f"CREATE DATABASE {test_db}")
    except Exception as e:  # noqa: BLE001
        await conn.close()
        pytest.skip(f"cannot provision test db: {e}")
    await conn.close()

    backend = PostgresBackend(f"postgresql:///{test_db}")  # left uninitialized; _build inits once
    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_what_if_postgres_parity(pg_backend):
    g = await _build(pg_backend, LINEAR)
    out = await g.what_if(
        "Flooding caused power outages.", origin="Heavy rainfall caused flooding.",
    )
    assert out["pruned_edges"] == [{"cause": "flooding", "effect": "power outages"}]
    assert {c["span"] for c in out["collapsed"]} == {"power outages", "hospital disruptions"}
    # citation parity: assert membership, not a specific representative string
    for c in out["collapsed"]:
        assert c["fact"] in LINEAR
    # backend owned by the fixture; don't g.close() it here
