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


class _FakeCrossEncoder:
    """Unrelated content far below zero, everything else above: the shape of the real one."""
    def predict(self, pairs):
        return [-9.0 if "Berlin" in text else 4.0 for _, text in pairs]


@pytest.mark.asyncio
async def test_memory_loop_rerank_cutoff_drops_same_topic_filler():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=False,
                    rerank_model=_FakeCrossEncoder())
    g.embeddings.rerank = _no_rerank
    await g.initialize()
    try:
        await g.add_texts(["The Rotterdam warehouse lost power on Tuesday.", "Alice works from the Berlin office."])
        q = "Why did the Rotterdam warehouse lose power?"
        loose = await MemoryLoop(g, min_score=-1.0).recall(q)
        assert any("Berlin" in f["content"] for f in loose.facts)         # cosine alone lets it through
        strict = await MemoryLoop(g, min_score=-1.0, rerank_min=-4.0).recall(q)
        assert not any("Berlin" in f["content"] for f in strict.facts)
        assert any("Rotterdam" in f["content"] for f in strict.facts)
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_follow_up_query_readds_a_fact_the_question_cutoff_dropped():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_causal)
    g.embeddings.rerank = _no_rerank
    await g.initialize()
    hop = "Storms hit the coast, so rain was heavy."
    # the hop fact reads unrelated to the question but close to the chain's root span
    g.embeddings.score = lambda q, texts: [0.9 if (t == hop) == ("coastal storms" in q) else 0.1 for t in texts]
    try:
        await g.add_texts(list(_RELS), extractor=_ents)
        block = await MemoryLoop(g, max_facts=8, min_score=0.5).recall("Why is the main road closed?")
        assert "coastal storms" in block.roots
        assert any(f["content"] == hop for f in block.facts)
    finally:
        await g.close()


@pytest.mark.asyncio
async def test_own_conversation_turns_do_not_crowd_out_facts():
    """The loop stores the conversation; asking again must still surface the real facts,
    not the question itself and the earlier 'I don't know' answer."""
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_causal)
    g.embeddings.rerank = _no_rerank
    await g.initialize()
    q = "Why is the main road closed?"
    hedge = "I don't know why the main road is closed."
    # fake scores: the question and the hedge score highest, the real facts lower
    g.embeddings.score = lambda query, texts: [1.0 if t == q else 0.9 if t == hedge else 0.5 for t in texts]
    try:
        await g.add_texts(list(_RELS), extractor=_ents, scopes={"notes"})
        loop = MemoryLoop(g, session="chat", max_facts=3)
        await loop.observe(q, hedge)                       # a first round that found nothing useful
        block = await loop.recall(q)
        got = [f["content"] for f in block.facts]
        assert q not in got                                # never inject the question itself
        assert "The flood closed the main road." in got    # the real fact wins a slot
        assert got.index("The flood closed the main road.") < (got.index(hedge) if hedge in got else 99)
    finally:
        await g.close()


def test_own_turn_detection_ignores_the_tenant_tag():
    loop = MemoryLoop.__new__(MemoryLoop); loop.session = "chat"
    assert loop._is_own_turn({"scopes": ["chat"]}, "q")
    assert loop._is_own_turn({"scopes": ["acme/chat", "acme"]}, "q")          # hosted: session tag + tenant tag
    assert not loop._is_own_turn({"scopes": ["acme/chat", "acme", "acme/notes"]}, "q")   # also held by a real session
    assert not loop._is_own_turn({"scopes": ["notes"]}, "q")
    assert not loop._is_own_turn({"scopes": []}, "q")
