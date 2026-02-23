"""Cross-source discovery demo.

Feeds two independent sources into ReasonGraph and shows how queries
can discover connections between them through shared entities and
causal relations -- something flat embedding search cannot do.

Source A: Tech industry report about TSMC's semiconductor plant in Arizona
Source B: Environmental report about Arizona's water crisis

Neither source references the other's topic. ReasonGraph bridges them
through shared entities (Arizona, Phoenix, Apple) extracted by GLiNER2.

Usage:
    uv run python examples/cross_source_discovery.py
"""

import asyncio

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

QUERIES = [
    "How does the Arizona water crisis affect semiconductor manufacturing?",
    "What supply chain risks does Apple face?",
    "What is the connection between Lake Mead water levels and chip production?",
    "What threatens TSMC production in Arizona?",
]


async def main():
    async with ReasonGraph() as graph:
        print("Ingesting Source A (tech industry report)...")
        await graph.add_texts(SOURCE_A_TECH)
        print("Ingesting Source B (environmental report)...")
        await graph.add_texts(SOURCE_B_WATER)

        nodes = await graph.get_all_nodes()
        edges = await graph.get_all_edges()
        text_nodes = [n for n in nodes if n.type == "text"]
        entity_nodes = [n for n in nodes if n.type == "entity"]

        print(f"\nGraph: {len(text_nodes)} text, {len(entity_nodes)} entity nodes, {len(edges)} edges")

        # Show what GLiNER2 extracted
        print(f"\nExtracted entities:")
        for entity in sorted(entity_nodes, key=lambda x: x.content):
            neighbors = await graph.backend.get_neighbors(entity.content)
            text_neighbors = [n["content"] for n in neighbors if n["type"] == "text"]
            sources = []
            if any(t in SOURCE_A_TECH for t in text_neighbors):
                sources.append("A")
            if any(t in SOURCE_B_WATER for t in text_neighbors):
                sources.append("B")
            label = ",".join(sources)
            print(f"  {entity.content}  [{label}]")

        print(f"\nBridge entities (appear in both sources):")
        for entity in sorted(entity_nodes, key=lambda x: x.content):
            neighbors = await graph.backend.get_neighbors(entity.content)
            text_neighbors = [n["content"] for n in neighbors if n["type"] == "text"]
            from_a = [t for t in text_neighbors if t in SOURCE_A_TECH]
            from_b = [t for t in text_neighbors if t in SOURCE_B_WATER]
            if from_a and from_b:
                print(f"  * {entity.content}  ({len(from_a)} from A, {len(from_b)} from B)")

        for query in QUERIES:
            print(f"\n{'=' * 80}")
            print(f"  Query: {query}")
            print(f"{'=' * 80}")
            results = await graph.query(query, top_k=5, hops=4, rerank_top_k=4)
            from_a = 0
            from_b = 0
            for i, text in enumerate(results, 1):
                if text in SOURCE_A_TECH:
                    source = "A"
                    from_a += 1
                elif text in SOURCE_B_WATER:
                    source = "B"
                    from_b += 1
                else:
                    source = "?"
                print(f"  {i}. [Source {source}] {text}")
            print(f"\n  Cross-source: {from_a} from Tech (A), {from_b} from Water (B)")


if __name__ == "__main__":
    asyncio.run(main())
