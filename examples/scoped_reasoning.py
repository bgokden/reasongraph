"""Multi-label scopes with cross-scope reasoning.

Scopes are free-text tags on facts (e.g. "user-alice", "topic-economy"), not
partitions. A query scoped to one tag draws its *seeds* from that tag, but
multi-hop traversal follows shared entities across every scope -- so a user's
personal situation connects to macro-economic facts tagged under a different
scope. A single fact can also carry several scopes at once.
"""

import asyncio

from reasongraph import ReasonGraph

# Personal facts about one user.
alice_facts = [
    "Alice is a software engineer saving to buy her first home in Phoenix, Arizona.",
    "Alice worries that rising home prices in Phoenix could push her budget too far.",
]

# General macro-economics facts, written independently of Alice.
economy_facts = [
    "The Federal Reserve raised interest rates to 5.5 percent to cool inflation.",
    "Higher Federal Reserve rates increase mortgage costs and slow home sales in Arizona.",
]

# A fact that legitimately belongs to BOTH scopes at once (multi-label).
shared_fact = [
    "Phoenix housing demand cooled as mortgage rates climbed through 2025.",
]


def scopes_of(node_lookup, content):
    tags = node_lookup.get(content, set())
    return "".join(f" [{t}]" for t in sorted(tags)) if tags else ""


async def main():
    async with ReasonGraph() as graph:
        await graph.add_texts(alice_facts, scopes=["user-alice"])
        await graph.add_texts(economy_facts, scopes=["topic-economy"])
        # One fact tagged with two scopes simultaneously
        await graph.add_texts(shared_fact, scopes=["user-alice", "topic-economy"])

        node_lookup = {n.content: n.scopes for n in await graph.get_all_nodes()}

        print("Facts tagged with more than one scope (multi-label):")
        for content, tags in node_lookup.items():
            if len(tags) > 1:
                print(f"  {sorted(tags)}  {content}")

        question = "Will it get harder for Alice to afford a home?"

        print("\n" + "=" * 78)
        print(f"  Query (seeds scoped to user-alice): {question}")
        print("=" * 78)
        results = await graph.query(question, scopes=["user-alice"], hops=4)
        for i, text in enumerate(results, 1):
            origin = "user-alice" if text in alice_facts + shared_fact else "topic-economy"
            print(f"  {i}. [{origin}]{scopes_of(node_lookup, text)} {text}")

        crossed = any(t in economy_facts for t in results)
        print(f"\n  Reached macro-economy facts from a user-scoped query: {crossed}")

        print("\n" + "=" * 78)
        print("  Same query, but seeds scoped to topic-economy")
        print("=" * 78)
        econ = await graph.query(question, scopes=["topic-economy"], hops=4)
        for i, text in enumerate(econ, 1):
            print(f"  {i}. {text}")


if __name__ == "__main__":
    asyncio.run(main())
