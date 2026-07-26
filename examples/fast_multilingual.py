"""Fast, multilingual, cross-lingual reasoning with the pure-ONNX stack.

Wires the session's recommended fast configuration:
  - extractor: gliner_small-v2.5 (multilingual, high recall, ~12 ms/call in ONNX)
  - embedder:  paraphrase-multilingual-MiniLM-L12-v2 via fastembed (pure ONNX)

Three facts about the same person are added in three languages. None references
the others, and the query is in English. The multilingual extractor pulls the
shared entity out of every language, so multi-hop traversal bridges all three --
cross-lingual reasoning, not just retrieval.

Run:  uv run python examples/fast_multilingual.py
Deps: pip install reasongraph[fastembed,gliner-onnx,sqlite]
"""

import asyncio

from reasongraph import ReasonGraph, GlinerExtractor, FastEmbedEmbedder

FACTS = [
    "Angela Merkel served as the Chancellor of Germany for sixteen years.",   # EN
    "Angela Merkel wurde 1954 in Hamburg geboren.",                           # DE: born in Hamburg
    "Angela Merkel estudió física en la Universidad de Leipzig.",             # ES: studied physics
]


async def main():
    graph = ReasonGraph(
        embed_model=FastEmbedEmbedder(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    )
    extractor = GlinerExtractor()  # defaults to gliner_small-v2.5

    async with graph:
        for fact in FACTS:
            entities = await graph.add_text(fact, extractor=extractor)
            print(f"  extracted {entities}")
            print(f"    <- {fact}")

        print("\nQuery (English): 'Tell me about Angela Merkel'")
        results = await graph.query("Tell me about Angela Merkel", hops=3)
        for i, text in enumerate(results, 1):
            print(f"  {i}. {text}")

        reached = sum(1 for f in FACTS if f in results)
        print(f"\n  Reached {reached}/{len(FACTS)} facts across 3 languages "
              f"from one English query.")


if __name__ == "__main__":
    asyncio.run(main())
