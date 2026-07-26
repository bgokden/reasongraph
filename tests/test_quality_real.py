"""Real-embedding quality gate.

The rest of the suite uses hash-based fake embeddings, so it validates plumbing
but nothing about real retrieval quality -- a real regression (bad embedder swap,
broken reranking, broken dedup) would pass silently. These tests load the REAL
default embedder/reranker (on CPU, per conftest) and assert end-to-end behaviour
that only holds if semantic similarity actually works. Thresholds are derived from
the model's own scores where possible, so the assertions are robust, not tuned.

Kept small (two model-loading tests) to bound runtime.
"""

import pytest

from reasongraph.graph import ReasonGraph
from reasongraph.backends._sqlite import SqliteBackend

# No-op entity extractor: these tests exercise embedding retrieval, not NER, so we
# avoid loading a second model. Facts are queried directly (single hop).
def _no_entities(text):
    return []


async def _graph():
    g = ReasonGraph(backend=SqliteBackend(":memory:"), causal_extractor=False)
    await g.initialize()
    return g


@pytest.mark.asyncio
async def test_real_embedding_retrieval_scores_and_dedup():
    g = await _graph()
    try:
        facts = [
            "The Eiffel Tower is located in Paris, France.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "The central bank raised interest rates to curb inflation.",
        ]
        await g.add_texts(facts, extractor=_no_entities)

        # Retrieval: the geography question surfaces the Eiffel Tower fact on top.
        detailed = await g.query_detailed(
            "Where is the Eiffel Tower?", top_k=3, hops=1,
        )
        assert detailed
        assert detailed[0]["content"] == facts[0]
        assert isinstance(detailed[0]["score"], float)

        # score(): a paraphrase is more similar than an unrelated sentence.
        paraphrase = "The Eiffel Tower stands in the French capital, Paris."
        sim_para = g.embeddings.score(facts[0], [paraphrase])[0]
        sim_unrel = g.embeddings.score(facts[0], [facts[1]])[0]
        assert sim_para > sim_unrel

        # Semantic dedup: with a threshold just under the measured paraphrase
        # similarity, the paraphrase is recognised as a duplicate and dropped.
        ents = await g.add_text(
            paraphrase, extractor=_no_entities, dedup_threshold=sim_para - 0.05,
        )
        assert ents == []
        contents = {n.content for n in await g.get_all_nodes()}
        assert paraphrase not in contents
        assert facts[0] in contents
    finally:
        await g.close()


def test_real_nli_resolver_detects_contradiction():
    # Loads the real NLI cross-encoder (cached in CI). Guards that the shipped
    # resolver actually separates a contradiction from an unrelated statement --
    # the whole reason NLI is used over cosine similarity.
    from reasongraph import NLIConflictResolver

    resolver = NLIConflictResolver()
    # Both candidates are topically similar (the realistic knn case): one is a
    # same-subject contradiction, the other a different-subject fact that stands.
    out = resolver.contradictions(
        "Alice lives in Berlin.",
        ["Alice lives in Munich.", "Bob lives in Munich."],
    )
    assert "Alice lives in Munich." in out
    assert "Bob lives in Munich." not in out


@pytest.mark.asyncio
async def test_real_trace_effects_single_hop():
    # Default causal extractor (real gliner-relex hybrid) + the causal walk end to end.
    g = ReasonGraph(backend=SqliteBackend(":memory:"))
    await g.initialize()
    try:
        await g.add_text("Heavy rainfall caused severe flooding.")
        traced = await g.trace_effects("Heavy rainfall caused severe flooding.")
        assert traced["origin"] == "Heavy rainfall caused severe flooding."
        assert traced["chain"], "real extractor should yield at least one causal hop"
        assert traced["chain"][0]["fact"] == "Heavy rainfall caused severe flooding."
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_real_supersede_removes_old_fact():
    g = await _graph()
    try:
        await g.add_text("Alice lives in Munich.", extractor=_no_entities)
        await g.supersede(
            "Alice lives in Munich.", "Alice lives in Berlin.", extractor=_no_entities,
        )
        answers = await g.query("Where does Alice live?", top_k=3, hops=1)
        assert "Alice lives in Berlin." in answers
        assert "Alice lives in Munich." not in answers
    finally:
        await g.close()
