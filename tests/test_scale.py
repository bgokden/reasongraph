"""A large graph must still answer, and must not scan.

Twenty thousand facts, a few hundred recurring names, one hub entity on five thousand
facts, and the console's checkout chain buried inside. Fake embeddings keep this fast
and deterministic: what is measured is the graph, not the models.
"""

import random
import time

import pytest

from reasongraph import ReasonGraph
from reasongraph.backends._memory import MemoryBackend
from test_causal import _fake_embed, _no_rerank

N_FACTS = 20_000
HUB = "Apple"
CHAIN = {
    "The checkout service stores shopping carts in Redis since the March release.": ["checkout service", "Redis"],
    "Redis runs on the same node as Elasticsearch.": ["Redis", "Elasticsearch"],
    "Because Elasticsearch rebuilds its index at 09:00 every day, the node's CPU is saturated for twenty minutes each morning.": ["Elasticsearch"],
}
RELS = {"Because Elasticsearch rebuilds its index at 09:00 every day, the node's CPU is saturated for twenty minutes each morning.":
            {"cause": "Elasticsearch rebuilds its index at 09:00 every day", "effect": "the node's CPU is saturated"}}


def _causal(texts):
    return [{"causal": t in RELS, "relations": [RELS[t]] if t in RELS else []} for t in texts]


@pytest.fixture(scope="module")
def big_graph():
    rng = random.Random(7)
    names = [f"Company {i}" for i in range(300)] + [f"Person {i}" for i in range(200)]
    ents: dict[str, list[str]] = {}
    texts = []
    for i in range(N_FACTS):
        e = [rng.choice(names)]
        if i % 4 == 0:
            e.append(HUB)                              # one hub on a quarter of all facts
        t = f"Note {i}: {e[0]} reported figure {rng.randint(1, 9999)} in period {i % 97}."
        texts.append(t); ents[t] = e
    for t, e in CHAIN.items():
        texts.append(t); ents[t] = e
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_causal)
    g.embeddings.rerank = _no_rerank
    g.initialize_sync()
    t0 = time.perf_counter()
    for i in range(0, len(texts), 1000):
        g.add_texts_sync(texts[i:i + 1000], extractor=lambda t: ents[t], scopes={"tenant"})
    g._ingest_seconds = time.perf_counter() - t0
    yield g
    g.close_sync()


def test_ingest_of_twenty_thousand_facts_is_bounded(big_graph):
    assert big_graph._ingest_seconds < 120, big_graph._ingest_seconds


@pytest.mark.asyncio
async def test_hub_expansion_is_capped_and_nearest(big_graph):
    g = big_graph
    q = g.embeddings.encode_query("Redis runs on the same node as Elasticsearch.")
    all_n = await g.backend.get_neighbors(HUB)
    assert len(all_n) >= 5000
    capped = await g.backend.nearest_neighbors(HUB, q, 64)
    assert len(capped) == 64
    # a walk through the hub sees at most max_degree neighbours
    seen = []
    real = g.backend.nearest_neighbors
    async def spy(content, emb, limit, scopes=None):
        r = await real(content, emb, limit, scopes); seen.append((content, len(r))); return r
    g.backend.nearest_neighbors = spy
    try:
        await g.discover("Apple reported figure 42 in period 3.", top_k=3, hops=2, max_results=8)
    finally:
        g.backend.nearest_neighbors = real
    assert seen and max(n for _, n in seen) <= g.max_degree


@pytest.mark.asyncio
async def test_chain_is_still_found_inside_the_large_graph(big_graph):
    g = big_graph
    t0 = time.perf_counter()
    res = await g.discover("The checkout service stores shopping carts in Redis since the March release.",
                           top_k=3, hops=3, max_results=8)
    elapsed = time.perf_counter() - t0
    got = {r["content"] for r in res}
    assert "Redis runs on the same node as Elasticsearch." in got, got
    assert elapsed < 5.0, elapsed


@pytest.mark.asyncio
async def test_counts_and_scopes_do_not_load_the_graph(big_graph):
    g = big_graph
    assert await g.backend.count_nodes("text") == N_FACTS + len(CHAIN)
    assert await g.backend.list_scopes("ten") == ["tenant"]
    assert (await g.backend.get_node_types([HUB, "Redis runs on the same node as Elasticsearch."])) == {HUB: "entity", "Redis runs on the same node as Elasticsearch.": "text"}
