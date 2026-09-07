"""Causal chain tracing over the directed 'causes' DAG (model-free)."""

import pytest

from reasongraph.graph import ReasonGraph
from reasongraph.backends._memory import MemoryBackend
from reasongraph.backends._sqlite import SqliteBackend

def _stable_hash(text):
    """Process-independent 64-bit hash (Python's hash() is randomized per run,
    which made fake embeddings and therefore test outcomes flaky)."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")



def _fake_encode(text):
    h = _stable_hash(text)
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


# --- causal_chain bridges: plain root fact -> next fact's cause span mentions its entity ---
_BRIDGE_RELS = {
    "When the CPU temperature rises, the system throttles performance.":
        {"cause": "the CPU temperature rises", "effect": "the system throttles performance"},
    "Throttling performance causes application latency to increase.":
        {"cause": "Throttling performance", "effect": "application latency to increase"},
}
_BRIDGE_ENTS = {
    "The server's CPU temperature exceeded 85°C.": ["CPU temperature", "server"],
    "The CPU temperature was monitored by a sensor.": ["CPU temperature", "sensor"],
    "When the CPU temperature rises, the system throttles performance.": ["CPU temperature"],
    "Throttling performance causes application latency to increase.": ["application latency"],
}


def _bridge_causal(texts):
    return [{"causal": t in _BRIDGE_RELS, "relations": [_BRIDGE_RELS[t]] if t in _BRIDGE_RELS else []}
            for t in texts]


@pytest.mark.asyncio
async def test_causal_chain_bridges_plain_root_fact_via_entity():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_bridge_causal,
                    span_link_threshold=0.99)
    g.embeddings.rerank = _no_rerank
    g.bridge_min_score = -1.0   # fake embeddings: rely on the shared-word rule only
    await g.initialize()
    await g.add_texts(list(_BRIDGE_ENTS), extractor=lambda t: _BRIDGE_ENTS[t])
    try:
        # the root fact has no causal relation of its own; its entity "CPU temperature" is
        # mentioned by the next fact's cause span, which the effect of links onward.
        chain = await g.causal_chain("The server's CPU temperature exceeded 85°C.",
                                     "Throttling performance causes application latency to increase.")
        assert chain is not None
        assert [(h["cause"], h["effect"]) for h in chain][0] == (
            "the CPU temperature rises", "the system throttles performance")
        assert chain[-1]["effect"] == "application latency to increase"
        # direction still matters: the last fact does not lead back to the root
        assert await g.causal_chain("Throttling performance causes application latency to increase.",
                                    "The server's CPU temperature exceeded 85°C.") is None
        # a distractor that shares the entity but has no causal role is not a bridge target
        assert await g.causal_chain("The server's CPU temperature exceeded 85°C.",
                                    "The CPU temperature was monitored by a sensor.") is None
    finally:
        await g.close()
