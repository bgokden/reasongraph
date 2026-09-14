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


def test_memory_class_recalls_facts_and_keeps_a_buffer_when_no_budget_is_set():
    from reasongraph.integrations.langchain import ReasonGraphMemory
    g = _graph()
    try:
        mem = ReasonGraphMemory(target=g, session="chat")
        assert mem.memory_variables == ["memory", "history"]

        first = mem.load_memory_variables({"input": "Why is the main road closed?"})
        assert "The flood closed the main road." in first["memory"]   # recalled, with its source
        assert "[notes" in first["memory"]
        assert first["history"] == ""                                  # nothing said yet

        mem.save_context({"input": "Why is the main road closed?"}, {"output": "Because the river flooded."})
        second = mem.load_memory_variables({"input": "And the warehouse?"})
        assert second["history"] == "Human: Why is the main road closed?\nAI: Because the river flooded."
        assert "Rotterdam" in second["memory"]

        mem.return_messages = True
        msgs = mem.load_memory_variables({"input": "x"})["history"]
        assert [type(m) for m in msgs] == [HumanMessage, AIMessage]

        # what the agent said is itself a fact now, in the chat session
        assert "Because the river flooded." in g.query_sync("the river flooded", top_k=5, scopes={"chat"})
    finally:
        g.close_sync()


def test_memory_class_folds_old_turns_but_never_forgets_them():
    from reasongraph.integrations.langchain import ReasonGraphMemory
    g = _graph()
    try:
        summaries = []

        def summarize(msgs):
            summaries.append([m["content"] for m in msgs])
            return "The user's ferry to Texel leaves at nine."

        mem = ReasonGraphMemory(target=g, session="chat", max_history_tokens=40, keep_tail_tokens=25,
                                summarizer=summarize)
        mem.save_context({"input": "My ferry to Texel leaves at nine tomorrow, remind me to pack the tent."},
                         {"output": "Noted, the ferry to Texel at nine and the tent."})
        mem.save_context({"input": "What is the weather like on the island in May?"},
                         {"output": "Usually mild, around fifteen degrees with wind."})
        mem.save_context({"input": "Is the road still closed?"}, {"output": "Yes, the flood closed it."})

        seen = mem.load_memory_variables({"input": "When does my ferry leave?"})
        assert "Is the road still closed?" in seen["history"]           # newest turn verbatim
        assert "pack the tent" not in seen["history"]                   # oldest turn folded out
        assert summaries and "pack the tent" in summaries[0][0]         # ...into the summary
        assert mem.summary and "Texel" in mem.summary
        assert seen["history"].startswith("Summary:")
        assert "ferry to Texel leaves at nine" in seen["memory"]        # and still recalled as a fact

        mem.clear()
        assert mem.load_memory_variables({"input": "ferry"})["history"] == ""
        assert not any("Texel" in t for t in g.query_sync("ferry to Texel", top_k=5))
        assert "The warehouse is in Rotterdam." in g.query_sync("The warehouse is in Rotterdam.", top_k=5)   # other sessions untouched
    finally:
        g.close_sync()


def test_chat_message_history_pairs_turns_and_folds():
    from reasongraph.integrations.langchain import ReasonGraphChatMessageHistory
    g = _graph()
    try:
        h = ReasonGraphChatMessageHistory(g, session="chat", max_history_tokens=20, keep_tail_tokens=10,
                                          summarizer=lambda msgs: "earlier: a ferry and a tent")
        h.add_messages([HumanMessage(content="My ferry to Texel leaves at nine, remind me to pack the tent."),
                        AIMessage(content="Noted.")])
        h.add_messages([HumanMessage(content="Is the road closed?"), AIMessage(content="Yes, by the flood.")])
        msgs = h.messages
        assert isinstance(msgs[0], SystemMessage) and "ferry" in msgs[0].content
        assert msgs[-1].content == "Yes, by the flood."
        assert any("Texel" in t for t in g.query_sync("My ferry to Texel leaves at nine", top_k=5, scopes={"chat"}))
    finally:
        g.close_sync()
