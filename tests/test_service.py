import pytest

from reasongraph.service import MemoryService
from reasongraph.backends._memory import MemoryBackend


def _fake_encode(text):
    h = hash(text)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def _fake_embed(x):
    return [_fake_encode(t) for t in x] if isinstance(x, list) else _fake_encode(x)


def _zeus_extractor(text):
    return ["Zeus"] if "Zeus" in text else []


def _fake_synth(query, context):
    facts = [c["content"] for c in context]
    return f"About {query}: " + "; ".join(facts)


@pytest.fixture
async def svc():
    s = MemoryService(
        backend=MemoryBackend(),
        embed_model=_fake_embed,
        extractor=_zeus_extractor,
        synthesizer=_fake_synth,
        causal_extractor=False,  # no real causal model in unit tests
    )
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_deferred_extraction_enriches_in_background():
    s = MemoryService(
        backend=MemoryBackend(), embed_model=_fake_embed,
        extractor=_zeus_extractor, causal_extractor=False,
        defer_extraction=True,
    )
    await s.initialize()
    try:
        res = await s.push("agent-1", "Zeus is king of the gods.")
        # push returns immediately with no entities (extraction is deferred)
        assert res["deferred"] is True and res["entities"] == []
        # the fact is queryable at once
        assert "Zeus is king of the gods." in await s.query("Zeus", session="agent-1")
        # after the background worker drains, the entity bridge exists
        await s._enrich_queue.join()
        st = await s.stats()
        assert st["facts"] == 1 and st["entities"] == 1
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_push_stats_sessions(svc):
    await svc.push("agent-1", "Zeus is king of the gods.")
    st = await svc.stats()
    assert st["facts"] == 1
    assert st["sessions"] == 1
    assert await svc.list_sessions() == ["agent-1"]


@pytest.mark.asyncio
async def test_cross_session_discovery(svc):
    # Two agents, separate sessions, bridged by the shared Zeus entity.
    await svc.push("agent-1", "Zeus threw lightning bolts.")
    await svc.push("agent-2", "Zeus lived on Mount Olympus.")

    found = await svc.discover("Zeus", session="agent-1", hops=3)
    by_content = {f["content"]: f for f in found}
    assert "Zeus threw lightning bolts." in by_content
    assert "Zeus lived on Mount Olympus." in by_content

    other = by_content["Zeus lived on Mount Olympus."]
    assert other["cross_session"] is True  # discovered from agent-2's session
    assert any(step.get("entity") == "Zeus" for step in other["path"])


@pytest.mark.asyncio
async def test_answer_synthesizes(svc):
    await svc.push("agent-1", "Zeus threw lightning bolts.")
    text = await svc.answer("Zeus", session="agent-1")
    assert isinstance(text, str)
    assert "Zeus threw lightning bolts." in text


@pytest.mark.asyncio
async def test_supersede_removes_old(svc):
    await svc.push("agent-1", "Zeus is mortal.")
    await svc.supersede("agent-1", "Zeus is mortal.", "Zeus is immortal.")
    facts = await svc.query("Zeus", session="agent-1")
    assert "Zeus is mortal." not in facts
    assert "Zeus is immortal." in facts


# -- Multi-domain, multi-session discovery at scale --

# A small economy/supply-chain/energy/health/policy world. Facts interlock only
# through the named entities they share; discovery walks those bridges.
_MULTIDOMAIN = {
    "markets-bot": [
        "Nvidia market cap crossed three trillion dollars on AI chip demand.",
        "TSMC reported record revenue from AI accelerators.",
    ],
    "supply-bot": [
        "TSMC manufactures the advanced chips that Nvidia designs.",
        "TSMC is building a new fabrication plant in Arizona.",
        "ASML supplies EUV lithography machines to TSMC.",
    ],
    "energy-bot": [
        "Arizona declared a water emergency amid a record drought.",
        "Taiwan expanded desalination to protect its chip fabs.",
    ],
    "health-bot": [
        "Drought in Arizona worsened dust storms and respiratory illness.",
    ],
    "policy-bot": [
        "Export controls restricted Nvidia AI chips from China.",
    ],
}

_MULTIDOMAIN_ENTITIES = ["TSMC", "Nvidia", "Apple", "Arizona", "Taiwan", "ASML", "China"]


def _multidomain_extractor(text):
    return [e for e in _MULTIDOMAIN_ENTITIES if e in text]


@pytest.fixture
async def world():
    s = MemoryService(
        backend=MemoryBackend(),
        embed_model=_fake_embed,
        extractor=_multidomain_extractor,
        synthesizer=_fake_synth,
        causal_extractor=False,  # no real causal model in unit tests
    )
    await s.initialize()
    for session, facts in _MULTIDOMAIN.items():
        for fact in facts:
            await s.push(session, fact)
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_multidomain_stats(world):
    st = await world.stats()
    assert st["sessions"] == 5
    assert st["facts"] == 9  # only text facts, entities excluded
    assert set(await world.list_sessions()) == {
        "markets-bot", "supply-bot", "energy-bot", "health-bot", "policy-bot",
    }


@pytest.mark.asyncio
async def test_markets_query_reaches_supply_and_policy(world):
    # A markets query about Nvidia crosses into supply and policy sessions via
    # the shared Nvidia entity, and two hops out to Arizona via TSMC.
    found = await world.discover("Nvidia AI chips", session="markets-bot", hops=5)
    by_content = {f["content"]: f for f in found}

    supply = by_content["TSMC manufactures the advanced chips that Nvidia designs."]
    assert supply["cross_session"] is True
    assert supply["scopes"] == ["supply-bot"]
    # Reached through a shared markets<->supply bridge (this fact carries both
    # Nvidia and TSMC entities; either is a valid bridge, so accept either).
    assert any(step.get("entity") in ("Nvidia", "TSMC") for step in supply["path"])

    policy = by_content["Export controls restricted Nvidia AI chips from China."]
    assert policy["cross_session"] is True
    assert any(step.get("entity") == "Nvidia" for step in policy["path"])

    # Two-hop reach: Nvidia -> (TSMC makes chips) -> TSMC -> Arizona fab.
    fab = by_content["TSMC is building a new fabrication plant in Arizona."]
    assert fab["cross_session"] is True
    entities_on_path = [step["entity"] for step in fab["path"] if "entity" in step]
    assert "TSMC" in entities_on_path


@pytest.mark.asyncio
async def test_energy_query_reaches_supply_and_health(world):
    # An energy query about Arizona reaches the fab a supply bot tracks and the
    # health impact a health bot recorded -- both via the shared Arizona entity.
    found = await world.discover("Arizona drought water", session="energy-bot", hops=4)
    by_content = {f["content"]: f for f in found}

    fab = by_content["TSMC is building a new fabrication plant in Arizona."]
    assert fab["cross_session"] is True
    assert any(step.get("entity") == "Arizona" for step in fab["path"])

    health = by_content["Drought in Arizona worsened dust storms and respiratory illness."]
    assert health["cross_session"] is True
    assert health["scopes"] == ["health-bot"]

    # The energy bot's own fact is not a cross-session discovery.
    own = by_content["Arizona declared a water emergency amid a record drought."]
    assert own["cross_session"] is False


@pytest.mark.asyncio
async def test_answer_folds_in_cross_session_facts(world):
    text = await world.answer("Nvidia AI chips", session="markets-bot", hops=5)
    # The synthesized answer names facts pulled from other sessions.
    assert "TSMC manufactures the advanced chips that Nvidia designs." in text
    assert "Export controls restricted Nvidia AI chips from China." in text
