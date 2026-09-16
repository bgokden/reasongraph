"""LangChain's standard retriever tests against an in-memory ReasonGraph."""

from __future__ import annotations

import pytest
from langchain_tests.integration_tests import RetrieversIntegrationTests
from reasongraph import ReasonGraph

from langchain_reasongraph import ReasonGraphRetriever

FACTS = [
    "The checkout service stores shopping carts in Redis since the March release.",
    "Redis runs on the same node as Elasticsearch.",
    "Because Elasticsearch rebuilds its index at 09:00 every day, the node's CPU is saturated for twenty minutes each morning.",
    "Checkout latency doubles every morning around nine.",
    "The payments team owns the checkout service.",
    "The search team owns Elasticsearch.",
    "Marketing sends the newsletter at 09:30.",
]


@pytest.fixture(scope="module")
def graph() -> ReasonGraph:
    g = ReasonGraph()
    g.initialize_sync()
    g.add_texts_sync(FACTS)
    return g


class TestReasonGraphRetriever(RetrieversIntegrationTests):
    @pytest.fixture(autouse=True)
    def _graph(self, graph: ReasonGraph) -> None:
        self._g = graph

    @property
    def retriever_constructor(self) -> type[ReasonGraphRetriever]:
        return ReasonGraphRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"target": self._g, "k": 3}

    @property
    def retriever_query_example(self) -> str:
        return "Why is the checkout service slow every morning?"


def test_documents_carry_the_graph_metadata(graph: ReasonGraph) -> None:
    # whether `via`/`causes` are filled depends on the extractors installed (the bare
    # [langchain] extra has none); the shape is the contract
    docs = ReasonGraphRetriever(target=graph, k=4).invoke("Why is the checkout service slow every morning?")
    assert len(docs) == 4 and all(d.page_content in FACTS for d in docs)
    assert all({"sources", "via", "causes", "cross_session"} <= set(d.metadata) for d in docs)
