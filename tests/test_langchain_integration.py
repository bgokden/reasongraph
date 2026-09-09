import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from reasongraph import ReasonGraph
from reasongraph.backends._memory import MemoryBackend
from reasongraph.integrations.langchain import ReasonGraphRetriever, memory_tools, with_memory
from test_causal import _fake_embed, _no_rerank

_RELS = {"Heavy rain caused the river to flood.": {"cause": "heavy rain", "effect": "the river to flood"},
         "The flood closed the main road.": {"cause": "the flood", "effect": "closed the main road"}}
_causal = lambda texts: [{"causal": t in _RELS, "relations": [_RELS[t]] if t in _RELS else []} for t in texts]
_ents = lambda t: [w for w in ("river", "road") if w in t]


def _graph():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_causal)
    g.embeddings.rerank = _no_rerank
    g.initialize_sync()
    g.add_texts_sync(list(_RELS) + ["The warehouse is in Rotterdam."], extractor=_ents, scopes={"notes"})
    return g


def test_retriever_returns_connected_facts_as_documents():
    g = _graph()
    try:
        docs = ReasonGraphRetriever(target=g, max_facts=8).invoke("Why is the main road closed?")
        texts = [d.page_content for d in docs]
        assert "The flood closed the main road." in texts
        d = next(d for d in docs if d.page_content == "The flood closed the main road.")
        assert d.metadata["sources"] == ["notes"] and d.metadata["causes"][0]["effect"] == "closed the main road"
    finally:
        g.close_sync()


def test_with_memory_injects_recall_and_remembers_the_exchange():
    g = _graph()
    seen = {}
    def fake_model(msgs):
        seen["msgs"] = msgs
        return AIMessage(content="Because the river flooded after heavy rain.")
    try:
        chain = with_memory(RunnableLambda(fake_model), g, session="chat")
        out = chain.invoke([SystemMessage(content="Be brief."), HumanMessage(content="Why is the main road closed?")])
        assert out.content.startswith("Because the river flooded")
        msgs = seen["msgs"]
        assert isinstance(msgs[0], SystemMessage) and msgs[0].content == "Be brief."
        assert isinstance(msgs[1], SystemMessage) and "The flood closed the main road." in msgs[1].content
        assert isinstance(msgs[-1], HumanMessage)
        assert "Why is the main road closed?" in g.query_sync("main road closed?", top_k=5, scopes={"chat"})
    finally:
        g.close_sync()


def test_memory_tools_round_trip():
    g = _graph()
    try:
        tools = {t.name: t for t in memory_tools(g, session="agent")}
        assert set(tools) == {"remember", "recall", "discover"}
        assert tools["remember"].invoke({"text": "The Utrecht office opened in May."}).startswith("Remembered")
        # fake embeddings only match identical text; the real embedder ranks by meaning
        assert "Utrecht" in tools["recall"].invoke({"query": "The Utrecht office opened in May."})
        assert "closed the main road" in tools["discover"].invoke({"question": "Why is the main road closed?"})
    finally:
        g.close_sync()
