"""Causal chain tracing over the directed 'causes' DAG (model-free)."""

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


# Spans chain because each effect exactly matches the next cause.
_RELS = {
    "Heavy rainfall caused flooding.": {"cause": "rainfall", "effect": "flooding"},
    "Flooding caused power outages.": {"cause": "flooding", "effect": "power outages"},
    "Power outages caused hospital disruptions.": {"cause": "power outages", "effect": "hospital disruptions"},
}


def _fake_causal(texts):
    return [
        {"causal": bool(_RELS.get(t)), "relations": [_RELS[t]] if _RELS.get(t) else []}
        for t in texts
    ]


async def _build(backend, scopes=None):
    g = ReasonGraph(backend=backend, embed_model=_fake_embed, causal_extractor=_fake_causal)
    g.embeddings.rerank = _no_rerank
    await g.initialize()
    await g.add_texts(list(_RELS), extractor=lambda t: [], scopes=scopes)
    return g


@pytest.mark.parametrize("backend", [MemoryBackend(), SqliteBackend(":memory:")])
@pytest.mark.asyncio
async def test_trace_effects_causes_root_and_chain(backend):
    g = await _build(backend)
    try:
        eff = await g.trace_effects("Heavy rainfall caused flooding.")
        assert eff["origin"] == "Heavy rainfall caused flooding."
        hops = {(h["cause"], h["effect"]) for h in eff["chain"]}
        assert hops == {
            ("rainfall", "flooding"),
            ("flooding", "power outages"),
            ("power outages", "hospital disruptions"),
        }
        assert eff["terminals"] == ["hospital disruptions"]
        # every hop resolves to the fact that asserted it
        assert all(h["fact"] in _RELS for h in eff["chain"])
        # depth increases along the chain
        assert {h["effect"]: h["depth"] for h in eff["chain"]}["hospital disruptions"] == 2

        causes = await g.trace_causes("Power outages caused hospital disruptions.")
        assert causes["terminals"] == ["rainfall"]
        assert await g.root_causes("Power outages caused hospital disruptions.") == ["rainfall"]

        # directed reachability
        fwd = await g.causal_chain(
            "Heavy rainfall caused flooding.", "Power outages caused hospital disruptions."
        )
        assert fwd is not None and len(fwd) >= 1
        assert await g.causal_chain(
            "Power outages caused hospital disruptions.", "Heavy rainfall caused flooding."
        ) is None
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_trace_max_depth_bounds_walk():
    g = await _build(MemoryBackend())
    try:
        shallow = await g.trace_effects("Heavy rainfall caused flooding.", max_depth=1)
        # only the first hop is reachable within one level
        assert ("power outages", "hospital disruptions") not in {
            (h["cause"], h["effect"]) for h in shallow["chain"]
        }
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_trace_drops_superseded_hop():
    g = await _build(MemoryBackend())
    try:
        # retire the middle fact; its hop drops from the trace unless requested
        g.conflict_resolver = object()  # any non-None resolver enables the filter
        await g.backend.set_invalid(["Flooding caused power outages."], __import__("datetime").datetime.now())
        eff = await g.trace_effects("Heavy rainfall caused flooding.")
        facts = {h["fact"] for h in eff["chain"]}
        assert "Flooding caused power outages." not in facts
        full = await g.trace_effects("Heavy rainfall caused flooding.", include_superseded=True)
        assert "Flooding caused power outages." in {h["fact"] for h in full["chain"]}
    finally:
        await g.close()
