import pytest

from reasongraph import ReasonGraph, MemoryLoop
from reasongraph.backends._memory import MemoryBackend
from test_causal import _fake_embed, _no_rerank

_RELS = {"Heavy rain caused the river to flood.": {"cause": "heavy rain", "effect": "the river to flood"},
         "The flood closed the main road.": {"cause": "the flood", "effect": "closed the main road"},
         # the root span paraphrases the fact: no recalled fact contains "coastal storms"
         "Storms hit the coast, so rain was heavy.": {"cause": "coastal storms", "effect": "heavy rain"}}
def _causal(texts):
    return [{"causal": t in _RELS, "relations": [_RELS[t]] if t in _RELS else []} for t in texts]
def _ents(t):
    return [w for w in ("river", "road", "Rotterdam") if w in t]


@pytest.mark.asyncio
async def test_memory_loop_recalls_injects_and_observes():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_causal)
    g.embeddings.rerank = _no_rerank
    await g.initialize()
    try:
        await g.add_texts(list(_RELS) + ["The warehouse is in Rotterdam."], extractor=_ents, scopes={"notes"})
        loop = MemoryLoop(g, session="chat", max_facts=5)
        history = [{"role": "user", "content": "Why is the main road closed?"}]
        msgs, block = await loop.messages(history, system="You are a helpful assistant.")
        assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("You are")
        assert msgs[1]["role"] == "system" and "remember" in msgs[1]["content"].lower()
        assert "The flood closed the main road." in msgs[1]["content"] and "[notes" in msgs[1]["content"]
        assert msgs[-1] == history[-1]
        assert not block.empty and any("closed the main road" in f["content"] for f in block.facts)
        # why-questions walk back to the root and spell it out for the model
        assert "coastal storms" in block.roots and "Root cause(s)" in block.text
        assert any("Heavy rain caused" in f["content"] for f in block.facts)   # hop facts pulled in

        # a root cause nobody's fact states yet triggers one targeted follow-up query
        asked = []
        real = g.query_detailed
        async def spy(q, **kw):
            asked.append(q); return await real(q, **kw)
        g.query_detailed = spy
        block3 = await MemoryLoop(g, session="chat", max_facts=8).recall("Why is the main road closed?")
        assert "coastal storms" in block3.roots and "coastal storms" in asked   # asked for the loose end
        g.query_detailed = real
        off = await MemoryLoop(g, session="chat", max_facts=8, extend_query=False).recall("Why is the main road closed?")
        assert off.roots == block3.roots

        seen = {}
        def model(messages):
            seen["messages"] = messages
            return "The road is closed because the river flooded after heavy rain."
        reply, block2 = await loop.chat(model, history)
        assert reply.startswith("The road is closed") and seen["messages"][0]["role"] == "system"
        stored = await g.query("river flooded after heavy rain", top_k=3, scopes={"chat"})
        assert any("because the river flooded" in r for r in stored)   # the reply was remembered
        assert "Why is the main road closed?" in await g.query("main road closed?", top_k=5, scopes={"chat"})

        quiet = MemoryLoop(g, session="chat2", observe_user=False, redact=lambda t: None if "secret" in t else t)
        assert await quiet.observe("my secret is x", "fine") == ["fine"]
        assert await quiet.observe("hello", "the secret answer") == []
    finally:
        await g.close()


def test_memory_loop_sync_wrappers_and_empty_memory():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=False)
    g.embeddings.rerank = _no_rerank
    g.initialize_sync()
    loop = MemoryLoop(g)
    msgs, block = loop.messages_sync([{"role": "user", "content": "anything?"}])
    assert block.empty and msgs == [{"role": "user", "content": "anything?"}]
    reply, _ = loop.chat_sync(lambda m: "ok", [{"role": "user", "content": "Remember that the meeting is on Tuesday."}])
    assert reply == "ok"
    assert loop.recall_sync("meeting Tuesday").facts
    g.close_sync()


@pytest.mark.asyncio
async def test_memory_loop_min_score_keeps_unrelated_filler_out():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=False)
    g.embeddings.rerank = _no_rerank
    await g.initialize()
    try:
        await g.add_texts(["The printer is a shared model.", "The office is on the third floor."], extractor=lambda t: [], scopes={"notes"})
        strict = MemoryLoop(g, min_score=0.99)      # nothing is that similar with fake embeddings
        assert (await strict.recall("Why did the warehouse lose power?")).facts == []
        loose = MemoryLoop(g, min_score=-1.0)
        assert len((await loose.recall("Why did the warehouse lose power?")).facts) == 2
    finally:
        await g.close()
