"""Cross-source discovery demo.

Feeds two independent sources into ReasonGraph and shows how queries
can discover connections between them through shared entities and
causal relations -- something flat embedding search cannot do.

Source A: Tech industry report about a semiconductor plant
Source B: Environmental report about a water crisis

Neither source references the other. ReasonGraph bridges them through
shared entities (Phoenix, Apex Electronics) extracted by GLiNER2.

Usage:
    uv run python examples/cross_source_discovery.py
"""

import asyncio

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

QUERIES = [
    "How might the water crisis affect chip manufacturing?",
    "What risks does Apex Electronics face from its suppliers?",
    "What is the impact of declining Colorado River levels on the tech industry?",
    "What threatens semiconductor production in Arizona?",
]


async def main():
    async with ReasonGraph() as graph:
        print("Ingesting Source A (tech industry report)...")
        await graph.add_texts(SOURCE_A_TECH)
        print("Ingesting Source B (water crisis report)...")
        await graph.add_texts(SOURCE_B_WATER)

        nodes = await graph.get_all_nodes()
        edges = await graph.get_all_edges()
        text_nodes = [n for n in nodes if n.type == "text"]
        entity_nodes = [n for n in nodes if n.type == "entity"]

        print(f"\nGraph: {len(text_nodes)} text, {len(entity_nodes)} entity nodes, {len(edges)} edges")
        print(f"\nBridge entities (shared between sources):")
        for entity in sorted(entity_nodes, key=lambda x: x.content):
            neighbors = await graph.backend.get_neighbors(entity.content)
            text_neighbors = [n["content"] for n in neighbors if n["type"] == "text"]
            from_a = [t for t in text_neighbors if t in SOURCE_A_TECH]
            from_b = [t for t in text_neighbors if t in SOURCE_B_WATER]
            if from_a and from_b:
                print(f"  * {entity.content}  (connects {len(from_a)} from A, {len(from_b)} from B)")

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
