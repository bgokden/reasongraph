"""The dedup and span-linking thresholds, checked with the production embedder
(fastembed multilingual MiniLM) instead of a fake one. Skipped when fastembed is
not installed. Entities and causal spans are still supplied by small fakes so the
test measures similarity behaviour, not extractor quality."""

import pytest

pytest.importorskip("fastembed")

from reasongraph import FastEmbedEmbedder
from reasongraph.backends._memory import MemoryBackend
from reasongraph.service import MemoryService

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture(scope="module")
def embedder():
    return FastEmbedEmbedder(MODEL)


def _no_entities(_text):
    return []


def _causal(texts):
    out = []
    for t in texts:
        if " => " in t:
            cause, effect = [p.strip() for p in t.split(" => ", 1)]
            out.append({"causal": True, "relations": [{"cause": cause, "effect": effect}]})
        else:
            out.append({"causal": False, "relations": []})
    return out


async def test_dedup_090_merges_paraphrases_but_not_distinct_facts(embedder):
    svc = MemoryService(backend=MemoryBackend(), embed_model=embedder, extractor=_no_entities,
                        causal_extractor=False, dedup_threshold=0.90)
    async with svc:
        await svc.push("a", "Alice works from the Berlin office.")
        r = await svc.push_many("b", ["Alice is working out of the Berlin office.",   # paraphrase
                                      "Redis and Elasticsearch run on the same node."])
        assert r["entities"] == [[], []]

        async def stored():
            return {n.content for n in await svc.graph.get_all_nodes() if n.type == "text"}

        assert "Alice is working out of the Berlin office." not in await stored()   # merged
        assert (await svc.graph.backend.get_scopes(["Alice works from the Berlin office."]))[
            "Alice works from the Berlin office."] == {"a", "b"}
        # distinct facts about the same subject are kept as separate nodes
        distinct = ["Bob works from the Berlin office.", "Alice moved to the Amsterdam office.",
                    "Redis runs on a dedicated node."]
        for t in distinct:
            await svc.push("c", t)
        assert set(distinct) | {"Redis and Elasticsearch run on the same node."} <= await stored()


async def test_span_link_085_joins_differently_worded_causal_spans(embedder):
    facts = ["Because of the heavy rain, the river flooded the old town. => the heavy rain => the river flooded the old town",
             "the old town flooded => the main road was closed for two days"]
    for threshold, expect in ((None, False), (0.85, True)):
        svc = MemoryService(backend=MemoryBackend(), embed_model=embedder, extractor=_no_entities,
                            causal_extractor=_causal)
        svc.graph.span_link_threshold = threshold
        async with svc:
            await svc.push("s", "the heavy rain => the river flooded the old town")
            await svc.push("s", "the old town flooded => the main road was closed for two days")
            res = await svc.causal_chain("the heavy rain", "the main road was closed")
            assert bool(res["chain"]) is expect, (threshold, res)
            if expect:
                # the tie does not create chains between unrelated spans
                await svc.push("s", "Elasticsearch rebuilds its index => the node's CPU is saturated")
                assert not (await svc.causal_chain("the heavy rain", "CPU is saturated"))["chain"]
