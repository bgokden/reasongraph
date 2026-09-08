"""The console's four starter scenarios, end to end, with the production models.

Guards what a fresh-graph eval and the fake-embedding tests cannot: that the memory loop,
run with the real embedder, reranker, entity tagger and causal model, still hands every
fact of a multi-source chain to the chat model. Skipped unless the models are installed.
"""

import os

import pytest

from reasongraph import MemoryLoop, ReasonGraph
from reasongraph.backends._memory import MemoryBackend

pytest.importorskip("gliner")
pytest.importorskip("transformers")

SCENARIOS = {
    "coding": ("Why is the checkout service slow every morning?", {
        "shop": ["The checkout service stores shopping carts in Redis since the March release."],
        "infra": ["Redis runs on the same node as Elasticsearch."],
        "search": ["Because Elasticsearch rebuilds its index at 09:00 every day, the node's CPU is saturated for twenty minutes each morning."]}),
    "support": ("Why did Maria Lopez cancel?", {
        "support-bot": ["Maria Lopez reported that Bulk Export fails for files over 50 MB."],
        "product-notes": ["Bulk Export's size limit was lowered to 50 MB in the May release to cut storage costs."],
        "billing-bot": ["Maria Lopez downgraded to the Free plan in June."]}),
    "travel": ("Anything I should know about Friday's trip?", {
        "calendar": ["On Friday Berk flies from Amsterdam to San Francisco with a connection in Frankfurt."],
        "news": ["Frankfurt airport ground staff announced a strike for Friday, which cancels most connecting flights."],
        "preferences": ["Berk prefers direct flights when a connection is at risk."]}),
    "analyst": ("How does the Arizona water crisis affect Apple?", {
        "tech-report": ["TSMC is building a $40B chip plant in Phoenix, Arizona, that needs 10M gallons of water a day."],
        "environment": ["Arizona ordered water cuts for industrial users after Lake Mead hit a record low."],
        "markets": ["Apple sources its M-series chips from TSMC and warned investors about component shortages."]}),
}
# unrelated facts every real tenant accumulates; none of them may crowd a chain out
NOISE = ["Alice works from the Berlin office.", "Dana leads the platform team.", "The company pays quarterly dividends.",
         "Bob now reports to Dana.", "Maria Lopez cancelled her subscription.", "Apple opened a new office in Rotterdam."]


@pytest.fixture(scope="module")
def graph():
    from reasongraph._extraction import GlinerExtractor
    os.environ.setdefault("REASONGRAPH_CAUSAL_MODEL", "Berk/causal-span-pointer-v2")
    os.environ.setdefault("REASONGRAPH_CAUSAL_TOKEN_GATE", "hf://Berk/causal-span-pointer-v2/token_gate")
    os.environ.setdefault("REASONGRAPH_CAUSAL_TOKEN_GATE_THRESHOLD", "0.1")
    g = ReasonGraph(backend=MemoryBackend(), embed_model="paraphrase-multilingual-MiniLM-L12-v2",
                    rerank_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    g.initialize_sync()
    # the hosted service's label set (CLOUD_ENTITY_LABELS default); "software" and "service"
    # are what make "Redis" and "Bulk Export" bridge sessions
    extractor = GlinerExtractor(labels=["person", "organization", "location", "event", "product", "software", "service"])
    for _, (_, sessions) in SCENARIOS.items():
        for session, texts in sessions.items():
            g.add_texts_sync(texts, extractor=extractor, scopes={session})
    g.add_texts_sync(NOISE, extractor=extractor, scopes={"misc"})
    yield g
    g.close_sync()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_chat_context_holds_the_whole_chain(graph, name):
    question, sessions = SCENARIOS[name]
    expected = [t for texts in sessions.values() for t in texts]
    block = MemoryLoop(graph, session="chat", max_facts=8).recall_sync(question)
    got = [f["content"] for f in block.facts]
    missing = [t for t in expected if t not in got]
    assert not missing, f"{name}: missing {missing}; context was {got}"
