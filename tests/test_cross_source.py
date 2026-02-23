"""Cross-source discovery: two independent sources connected through shared entities.

Unlike standard retrieval tests that check "did you find the right document?",
these tests verify that ReasonGraph can discover connections between sources
that never reference each other directly. The graph bridges them through
shared entities (organizations, locations) and causal relations extracted
by GLiNER2.

Scenario: A tech industry report about TSMC's semiconductor plant and an
environmental report about Arizona's water crisis share entities (Arizona,
Phoenix, Apple) but neither mentions the other's topic. Queries that span
both domains should return results from both sources.
"""

import pytest

from reasongraph import ReasonGraph

SOURCE_A_TECH = [
    "TSMC announced plans to build a $40 billion semiconductor fabrication plant in Phoenix, Arizona.",
    "The Phoenix fab requires 10 million gallons of purified water daily to cool wafers during the chip etching process.",
    "TSMC signed a long-term supply agreement with Apple to manufacture next-generation M-series processors at the Arizona facility.",
    "Construction delays at the Phoenix site pushed first production to late 2025, raising concerns among TSMC's major customers.",
]

SOURCE_B_WATER = [
    "Arizona declared a water emergency after Lake Mead dropped to its lowest level since the 1930s, threatening water supply for millions.",
    "The Arizona Department of Water Resources ordered mandatory water cuts for all industrial users in Maricopa County, where Phoenix is located.",
    "Intel paused expansion of its Chandler, Arizona chip plant citing water availability concerns and rising operational costs.",
    "Apple warned investors that component shortages from its Asian and North American suppliers could impact iPhone production timelines through 2026.",
]


@pytest.fixture
async def graph():
    g = ReasonGraph()
    await g.initialize()
    await g.add_texts(SOURCE_A_TECH)
    await g.add_texts(SOURCE_B_WATER)
    yield g
    await g.close()


def _split_by_source(results: list[str]) -> tuple[list[str], list[str]]:
    """Split query results into which source they came from."""
    from_a = [r for r in results if r in SOURCE_A_TECH]
    from_b = [r for r in results if r in SOURCE_B_WATER]
    return from_a, from_b


# -- Cross-source discovery tests --

@pytest.mark.asyncio
async def test_water_crisis_discovers_semiconductor_manufacturing(graph):
    """A water crisis query should reach semiconductor facts via Arizona/Phoenix entities."""
    results = await graph.query(
        "How does the Arizona water crisis affect semiconductor manufacturing?"
    )

    from_a, from_b = _split_by_source(results)
    assert len(from_a) >= 1, f"Expected tech source results, got only: {results}"
    assert len(from_b) >= 1, f"Expected water source results, got only: {results}"


@pytest.mark.asyncio
async def test_apple_supply_chain_bridges_sources(graph):
    """An Apple supply chain query should discover TSMC agreement via shared entity."""
    results = await graph.query("What supply chain risks does Apple face?")

    from_a, from_b = _split_by_source(results)
    assert len(from_a) >= 1, f"Expected tech source results, got only: {results}"
    assert len(from_b) >= 1, f"Expected water source results, got only: {results}"

    # The supply agreement should be discovered
    assert any("supply agreement" in r or "Apple" in r for r in from_a), \
        f"Expected TSMC-Apple supply agreement in Source A results: {from_a}"


@pytest.mark.asyncio
async def test_lake_mead_reaches_chip_production(graph):
    """A Lake Mead query should traverse to chip production via Arizona."""
    results = await graph.query(
        "What is the connection between Lake Mead water levels and chip production?"
    )

    from_a, from_b = _split_by_source(results)
    assert len(from_a) >= 1, f"Expected tech source results, got only: {results}"
    assert len(from_b) >= 1, f"Expected water source results, got only: {results}"


@pytest.mark.asyncio
async def test_tsmc_query_reaches_water_crisis(graph):
    """A TSMC production query should discover water restrictions via Phoenix."""
    results = await graph.query("What threatens TSMC production in Arizona?")

    from_a, from_b = _split_by_source(results)
    assert len(from_a) >= 1, f"Expected tech source results, got only: {results}"
    assert len(from_b) >= 1, f"Expected water source results, got only: {results}"


# -- Graph structure tests --

@pytest.mark.asyncio
async def test_shared_entities_create_bridges(graph):
    """Arizona, Phoenix, and Apple should appear as entity nodes connecting both sources."""
    nodes = await graph.get_all_nodes()
    entity_contents = {n.content for n in nodes if n.type == "entity"}

    assert "Arizona" in entity_contents, \
        f"Arizona entity not found in: {entity_contents}"
    assert "Phoenix" in entity_contents, \
        f"Phoenix entity not found in: {entity_contents}"
    assert "Apple" in entity_contents, \
        f"Apple entity not found in: {entity_contents}"


@pytest.mark.asyncio
async def test_bridge_entity_connects_both_sources(graph):
    """An entity shared between sources should have neighbors from both."""
    # Apple appears in both Source A (TSMC supply agreement) and Source B (investor warning)
    neighbors = await graph.backend.get_neighbors("Apple")
    neighbor_texts = {n["content"] for n in neighbors if n["type"] == "text"}

    from_a = neighbor_texts & set(SOURCE_A_TECH)
    from_b = neighbor_texts & set(SOURCE_B_WATER)

    assert len(from_a) >= 1, f"Apple should connect to Source A, got: {neighbor_texts}"
    assert len(from_b) >= 1, f"Apple should connect to Source B, got: {neighbor_texts}"


@pytest.mark.asyncio
async def test_causal_relations_extracted(graph):
    """GLiNER2 should extract causal relations that enrich the graph."""
    nodes = await graph.get_all_nodes()
    entity_contents = {n.content for n in nodes if n.type == "entity"}

    # These causal spans should exist as entity nodes
    # "Lake Mead dropped -> water emergency" and "component shortages -> iPhone production timelines"
    causal_entities = {
        "Lake Mead", "water emergency",
        "component shortages", "iPhone production timelines",
        "Construction delays", "first production",
    }
    found = causal_entities & entity_contents
    assert len(found) >= 3, \
        f"Expected at least 3 causal entities, found {len(found)}: {found}"
