"""Cross-source discovery: two independent sources connected through shared entities.

Unlike standard retrieval tests that check "did you find the right document?",
these tests verify that ReasonGraph can discover connections between sources
that never reference each other directly. The graph bridges them through
shared entities (organizations, locations) and causal relations extracted
by GLiNER2.

Scenario: A tech industry report and a water crisis report share entities
(Phoenix, Apex Electronics) but neither mentions the other's topic. Queries
that span both domains should return results from both sources.
"""

import pytest

from reasongraph import ReasonGraph

SOURCE_A_TECH = [
    "Meridian Technologies opened a semiconductor fabrication plant in Phoenix, Arizona in 2023.",
    "Dr. Sarah Chen, chief engineer at Meridian Technologies, developed a new chip architecture requiring enormous water usage for cooling.",
    "The Phoenix fabrication plant consumes 10 million gallons of water daily for semiconductor manufacturing.",
    "Meridian Technologies signed a five-year supply contract with Apex Electronics to deliver next-generation processors.",
]

SOURCE_B_WATER = [
    "Phoenix, Arizona declared a water emergency in 2024 due to declining Colorado River levels.",
    "The Arizona Department of Water Resources imposed mandatory 40% water cuts on industrial users in the Phoenix metropolitan area.",
    "Large-scale manufacturing facilities in Phoenix face production shutdowns under the new water restrictions.",
    "Apex Electronics warned investors that supply chain disruptions from its key suppliers could delay product launches through 2026.",
]

ALL_SOURCES = set(SOURCE_A_TECH + SOURCE_B_WATER)


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
async def test_water_crisis_discovers_chip_manufacturing(graph):
    """A water crisis query should reach chip manufacturing facts via Phoenix entity."""
    results = await graph.query("How might the water crisis affect chip manufacturing?")

    from_a, from_b = _split_by_source(results)
    assert len(from_a) >= 1, f"Expected tech source results, got only: {results}"
    assert len(from_b) >= 1, f"Expected water source results, got only: {results}"


@pytest.mark.asyncio
async def test_supplier_risk_bridges_to_manufacturing(graph):
    """An Apex supply chain query should discover Meridian's plant via shared entity."""
    results = await graph.query("What risks does Apex Electronics face from its suppliers?")

    from_a, from_b = _split_by_source(results)
    assert len(from_a) >= 1, f"Expected tech source results, got only: {results}"
    assert len(from_b) >= 1, f"Expected water source results, got only: {results}"

    # The supply contract should be discovered
    assert any("supply contract" in r for r in results), \
        f"Expected supply contract in results: {results}"


@pytest.mark.asyncio
async def test_river_levels_reach_tech_industry(graph):
    """A Colorado River query should traverse to tech industry via Phoenix."""
    results = await graph.query(
        "What is the impact of declining Colorado River levels on the tech industry?"
    )

    from_a, from_b = _split_by_source(results)
    assert len(from_a) >= 1, f"Expected tech source results, got only: {results}"
    assert len(from_b) >= 1, f"Expected water source results, got only: {results}"


@pytest.mark.asyncio
async def test_semiconductor_query_reaches_water_crisis(graph):
    """A semiconductor production query should discover water restrictions."""
    results = await graph.query("What threatens semiconductor production in Arizona?")

    from_a, from_b = _split_by_source(results)
    assert len(from_a) >= 1, f"Expected tech source results, got only: {results}"
    assert len(from_b) >= 1, f"Expected water source results, got only: {results}"


# -- Graph structure tests --

@pytest.mark.asyncio
async def test_shared_entities_create_bridges(graph):
    """Phoenix and Apex Electronics should appear as entity nodes connecting both sources."""
    nodes = await graph.get_all_nodes()
    entity_contents = {n.content for n in nodes if n.type == "entity"}

    # These entities should exist and serve as bridges
    assert "Phoenix" in entity_contents or "Phoenix, Arizona" in entity_contents, \
        f"Phoenix entity not found in: {entity_contents}"
    assert "Apex Electronics" in entity_contents, \
        f"Apex Electronics entity not found in: {entity_contents}"


@pytest.mark.asyncio
async def test_both_sources_connected_through_entity(graph):
    """An entity shared between sources should have neighbors from both."""
    # Apex Electronics appears in both source A and source B
    neighbors = await graph.backend.get_neighbors("Apex Electronics")
    neighbor_texts = {n["content"] for n in neighbors if n["type"] == "text"}

    from_a = neighbor_texts & set(SOURCE_A_TECH)
    from_b = neighbor_texts & set(SOURCE_B_WATER)

    assert len(from_a) >= 1, f"Apex should connect to Source A, got: {neighbor_texts}"
    assert len(from_b) >= 1, f"Apex should connect to Source B, got: {neighbor_texts}"


@pytest.mark.asyncio
async def test_causal_relations_extracted(graph):
    """GLiNER2 should extract causal relations that enrich the graph."""
    nodes = await graph.get_all_nodes()
    entity_contents = {n.content for n in nodes if n.type == "entity"}

    # Causal entities created by GLiNER2 extraction
    causal_entities = {
        "water restrictions", "production shutdowns",
        "supply chain disruptions", "delay product launches",
        "declining Colorado River levels", "water emergency",
    }
    found = causal_entities & entity_contents
    assert len(found) >= 3, \
        f"Expected at least 3 causal entities, found {len(found)}: {found}"
