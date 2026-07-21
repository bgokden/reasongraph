from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from reasongraph._extraction import NERExtractor, GLiNER2Extractor, ChatExtractor
from reasongraph.graph import ReasonGraph
from reasongraph.backends._sqlite import SqliteBackend
from reasongraph.backends._memory import MemoryBackend


def _fake_encode(text):
    """Deterministic fake embedding for testing without loading real models."""
    h = hash(text)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def _fake_encode_batch(texts):
    return [_fake_encode(t) for t in texts]


def _fake_rerank(query, results, top_k, recency_weight=0.0):
    return results[:top_k]


@pytest.fixture
async def graph():
    g = ReasonGraph(backend=SqliteBackend(":memory:"))
    # Mock the embedding manager to avoid loading real models
    g.embeddings.encode = _fake_encode
    g.embeddings.encode_batch = _fake_encode_batch
    g.embeddings.rerank = _fake_rerank
    await g.initialize()
    yield g
    await g.close()


@pytest.mark.asyncio
async def test_add_and_query_nodes(graph):
    await graph.add_nodes([
        ("The sky is blue.", "text"),
        ("Water is wet.", "text"),
        ("sky", "entity"),
    ])
    await graph.add_edges([
        ("sky", "The sky is blue."),
    ])

    results = await graph.query("What color is the sky?", top_k=3, hops=1)
    assert isinstance(results, list)
    # Should return text-type nodes only
    for r in results:
        assert isinstance(r, str)


@pytest.mark.asyncio
async def test_context_manager():
    g = ReasonGraph(backend=SqliteBackend(":memory:"))
    g.embeddings.encode = _fake_encode
    g.embeddings.encode_batch = _fake_encode_batch
    g.embeddings.rerank = _fake_rerank

    async with g:
        await g.add_nodes([("test node", "text")])
        nodes = await g.get_all_nodes()
        assert len(nodes) == 1


@pytest.mark.asyncio
async def test_load_dataset(graph):
    await graph.load_dataset("syllogisms")
    nodes = await graph.get_all_nodes()
    assert len(nodes) > 0
    edges = await graph.get_all_edges()
    assert len(edges) > 0


@pytest.mark.asyncio
async def test_query_returns_text_only(graph):
    await graph.add_nodes([
        ("Fact about gravity.", "text"),
        ("gravity", "entity"),
    ])
    await graph.add_edges([("gravity", "Fact about gravity.")])

    results = await graph.query("gravity", top_k=5, hops=1)
    # Entity nodes should be filtered out
    assert "gravity" not in results


@pytest.mark.asyncio
async def test_get_all_nodes_and_edges(graph):
    await graph.add_nodes([("A", "text"), ("B", "text")])
    await graph.add_edges([("A", "B")])

    nodes = await graph.get_all_nodes()
    edges = await graph.get_all_edges()
    assert len(nodes) == 2
    assert len(edges) == 1


@pytest.mark.asyncio
async def test_query_hybrid_mode(graph):
    await graph.add_nodes([
        ("Flooding destroyed the village.", "text"),
        ("The sun was shining brightly.", "text"),
        ("flooding", "entity"),
    ])
    await graph.add_edges([("flooding", "Flooding destroyed the village.")])

    results = await graph.query(
        "flood damage", top_k=3, hops=1, search_mode="hybrid"
    )
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_query_keyword_mode(graph):
    await graph.add_nodes([
        ("Cats are independent animals.", "text"),
        ("Dogs are loyal companions.", "text"),
    ])

    results = await graph.query("cat", top_k=2, hops=1, search_mode="keyword")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_query_invalid_search_mode(graph):
    with pytest.raises(ValueError, match="search_mode"):
        await graph.query("test", search_mode="invalid")


# -- add_text with entity extraction --

def _fake_extractor(text: str) -> list[str]:
    """Fake extractor that returns hardcoded entities for testing."""
    entities = {
        "Heavy rainfall caused flooding in Bangladesh.": ["Bangladesh"],
        "Socrates was a philosopher in Athens.": ["Socrates", "Athens"],
        "The Amazon rainforest spans multiple countries.": ["Amazon"],
    }
    return entities.get(text, [])


@pytest.mark.asyncio
async def test_add_text_creates_nodes_and_edges(graph):
    entities = await graph.add_text(
        "Socrates was a philosopher in Athens.",
        extractor=_fake_extractor,
    )
    assert entities == ["Socrates", "Athens"]

    nodes = await graph.get_all_nodes()
    contents = {n.content for n in nodes}
    assert "Socrates was a philosopher in Athens." in contents
    assert "Socrates" in contents
    assert "Athens" in contents

    # Verify node types
    types = {n.content: n.type for n in nodes}
    assert types["Socrates was a philosopher in Athens."] == "text"
    assert types["Socrates"] == "entity"
    assert types["Athens"] == "entity"

    edges = await graph.get_all_edges()
    edge_pairs = {(e.from_content, e.to_content) for e in edges}
    assert ("Socrates", "Socrates was a philosopher in Athens.") in edge_pairs
    assert ("Athens", "Socrates was a philosopher in Athens.") in edge_pairs


@pytest.mark.asyncio
async def test_add_text_no_entities(graph):
    def no_entities(text):
        return []

    entities = await graph.add_text("Just a simple sentence.", extractor=no_entities)
    assert entities == []

    nodes = await graph.get_all_nodes()
    assert len(nodes) == 1
    assert nodes[0].type == "text"

    edges = await graph.get_all_edges()
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_add_texts_batch(graph):
    texts = [
        "Heavy rainfall caused flooding in Bangladesh.",
        "Socrates was a philosopher in Athens.",
    ]
    all_entities = await graph.add_texts(texts, extractor=_fake_extractor)
    assert all_entities == [["Bangladesh"], ["Socrates", "Athens"]]

    nodes = await graph.get_all_nodes()
    assert len(nodes) == 5  # 2 text + 3 entity


@pytest.mark.asyncio
async def test_add_texts_shared_entities(graph):
    """Entities appearing in multiple texts should be deduplicated (upsert)."""
    def shared_extractor(text):
        return ["shared_entity"]

    await graph.add_texts(["Text A.", "Text B."], extractor=shared_extractor)

    nodes = await graph.get_all_nodes()
    entity_nodes = [n for n in nodes if n.type == "entity"]
    assert len(entity_nodes) == 1
    assert entity_nodes[0].content == "shared_entity"

    edges = await graph.get_all_edges()
    assert len(edges) == 2  # shared_entity -> Text A, shared_entity -> Text B


# -- Real NER extractor tests --

@pytest.fixture(scope="module")
def ner_extractor():
    try:
        ext = NERExtractor()
        ext("")  # force model download
        return ext
    except Exception as e:
        pytest.skip(f"NER model not available: {e}")


def test_ner_extractor_finds_entities(ner_extractor):
    entities = ner_extractor("Barack Obama visited Paris and met with Emmanuel Macron.")
    assert len(entities) >= 2
    # Should find at least the main named entities
    entity_lower = [e.lower() for e in entities]
    assert any("obama" in e for e in entity_lower)
    assert any("paris" in e for e in entity_lower)


def test_ner_extractor_empty_input(ner_extractor):
    entities = ner_extractor("")
    assert entities == []


def test_ner_extractor_no_entities(ner_extractor):
    entities = ner_extractor("The weather is nice today.")
    # This sentence has no named entities
    assert isinstance(entities, list)


@pytest.mark.asyncio
async def test_add_text_with_real_ner(graph, ner_extractor):
    text = "Albert Einstein developed the theory of relativity in Berlin."
    entities = await graph.add_text(text, extractor=ner_extractor)
    assert len(entities) >= 1

    nodes = await graph.get_all_nodes()
    contents = {n.content for n in nodes}
    assert text in contents
    for entity in entities:
        assert entity in contents

    # All entities should be linked to the text
    edges = await graph.get_all_edges()
    edge_pairs = {(e.from_content, e.to_content) for e in edges}
    for entity in entities:
        assert (entity, text) in edge_pairs


# -- Causal extraction tests --

def _fake_causal_extractor(sentences: list[str]) -> list[dict]:
    """Fake causal extractor that returns hardcoded cause-effect pairs."""
    results = []
    for sent in sentences:
        if "caused" in sent.lower() or "leads to" in sent.lower():
            results.append({
                "text": sent,
                "causal": True,
                "relations": [
                    {"cause": "heavy rainfall", "effect": "flooding"},
                ],
            })
        else:
            results.append({"text": sent, "causal": False, "relations": []})
    return results


@pytest.mark.asyncio
async def test_add_texts_with_causal_extractor(graph):
    texts = [
        "Heavy rainfall caused severe flooding.",
        "The sun was shining brightly.",
    ]
    entities = await graph.add_texts(
        texts,
        extractor=_fake_extractor,
        causal_extractor=_fake_causal_extractor,
    )

    nodes = await graph.get_all_nodes()
    contents = {n.content for n in nodes}

    # Source texts should be present
    assert "Heavy rainfall caused severe flooding." in contents
    assert "The sun was shining brightly." in contents

    # Cause and effect spans from the causal extractor
    assert "heavy rainfall" in contents
    assert "flooding" in contents

    edges = await graph.get_all_edges()
    edge_pairs = {(e.from_content, e.to_content) for e in edges}

    # Causal link: cause -> effect
    assert ("heavy rainfall", "flooding") in edge_pairs
    # Both link back to source
    assert ("heavy rainfall", "Heavy rainfall caused severe flooding.") in edge_pairs
    assert ("flooding", "Heavy rainfall caused severe flooding.") in edge_pairs


@pytest.mark.asyncio
async def test_add_texts_non_causal_sentence_no_relations(graph):
    texts = ["The weather is nice today."]
    await graph.add_texts(
        texts,
        extractor=_fake_extractor,
        causal_extractor=_fake_causal_extractor,
    )

    edges = await graph.get_all_edges()
    # No causal edges for non-causal sentence, no NER entities from fake extractor
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_add_texts_causal_without_ner(graph):
    """Causal extraction works even when NER returns no entities."""
    def no_entities(text):
        return []

    texts = ["Deforestation leads to soil erosion."]

    def causal(sents):
        return [{
            "text": sents[0],
            "causal": True,
            "relations": [{"cause": "deforestation", "effect": "soil erosion"}],
        }]

    await graph.add_texts(texts, extractor=no_entities, causal_extractor=causal)

    nodes = await graph.get_all_nodes()
    contents = {n.content for n in nodes}
    assert "deforestation" in contents
    assert "soil erosion" in contents

    edges = await graph.get_all_edges()
    edge_pairs = {(e.from_content, e.to_content) for e in edges}
    assert ("deforestation", "soil erosion") in edge_pairs


# -- Real GLiNER2 extractor tests --

@pytest.fixture(scope="module")
def gliner2_extractor():
    pytest.importorskip("gliner2")
    try:
        ext = GLiNER2Extractor()
        ext("")  # force model download
        return ext
    except Exception as e:
        pytest.skip(f"GLiNER2 model not available: {e}")


def test_chat_extractor_config():
    """ChatExtractor adds conversational entity types on top of the defaults."""
    ext = ChatExtractor()
    assert isinstance(ext, GLiNER2Extractor)
    for t in ("preference", "plan", "topic"):
        assert t in ext.entity_types
    for t in ("person", "location", "event"):
        assert t in ext.entity_types
    assert hasattr(ext, "extract_causal")  # inherits causal extraction


@pytest.fixture(scope="module")
def chat_extractor():
    pytest.importorskip("gliner2")
    try:
        ext = ChatExtractor()
        ext("")  # force model download
        return ext
    except Exception as e:
        pytest.skip(f"GLiNER2 model not available: {e}")


def test_chat_extractor_runs_on_conversational_text(chat_extractor):
    result = chat_extractor("I love hard techno and I plan to visit Berlin with Sarah.")
    assert isinstance(result, list)
    assert all(isinstance(e, str) for e in result)
    assert len(result) >= 1


def test_gliner2_entity_extraction(gliner2_extractor):
    entities = gliner2_extractor("Barack Obama visited Paris and met with Emmanuel Macron.")
    assert isinstance(entities, list)
    assert len(entities) >= 1
    entity_lower = [e.lower() for e in entities]
    assert any("obama" in e for e in entity_lower) or any("paris" in e for e in entity_lower)


def test_gliner2_entity_extraction_empty(gliner2_extractor):
    entities = gliner2_extractor("")
    assert isinstance(entities, list)


def test_gliner2_causal_extraction(gliner2_extractor):
    texts = [
        "Heavy rainfall caused severe flooding in the coastal regions.",
        "The sun was shining brightly.",
    ]
    results = gliner2_extractor.extract_causal(texts)
    assert len(results) == 2

    # First sentence should have causal relations
    assert results[0]["text"] == texts[0]
    assert isinstance(results[0]["causal"], bool)
    assert isinstance(results[0]["relations"], list)

    # Second sentence is not causal
    assert results[1]["text"] == texts[1]

    # Check structure of relations
    for rel in results[0].get("relations", []):
        assert "cause" in rel
        assert "effect" in rel
        assert isinstance(rel["cause"], str)
        assert isinstance(rel["effect"], str)


def test_gliner2_causal_extraction_paragraph(gliner2_extractor):
    texts = [
        "Deforestation in the Amazon has accelerated soil erosion across the region. "
        "Without tree roots to hold the soil, landslides have become more frequent "
        "during heavy rains."
    ]
    results = gliner2_extractor.extract_causal(texts)
    assert len(results) == 1
    # Should find at least one causal relation in this paragraph
    if results[0]["causal"]:
        assert len(results[0]["relations"]) >= 1


@pytest.mark.asyncio
async def test_add_texts_with_real_gliner2(graph, gliner2_extractor):
    texts = [
        "Smoking causes lung cancer and heart disease.",
    ]
    entities = await graph.add_texts(
        texts,
        extractor=gliner2_extractor,
        causal_extractor=gliner2_extractor.extract_causal,
    )

    nodes = await graph.get_all_nodes()
    assert len(nodes) >= 1  # At least the source text node

    # Source text should be present
    contents = {n.content for n in nodes}
    assert texts[0] in contents


# -- delete and supersede --

def _home_extractor(text: str) -> list[str]:
    """Fake extractor: both facts share the 'home' entity, differ on the city."""
    entities = {
        "I live in Amsterdam.": ["Amsterdam", "home"],
        "I live in Rotterdam.": ["Rotterdam", "home"],
    }
    return entities.get(text, [])


@pytest.mark.asyncio
async def test_delete_removes_node_and_incident_edges(graph):
    await graph.add_text(
        "Socrates was a philosopher in Athens.", extractor=_fake_extractor
    )

    deleted = await graph.delete("Socrates was a philosopher in Athens.")
    assert deleted is True

    contents = {n.content for n in await graph.get_all_nodes()}
    assert "Socrates was a philosopher in Athens." not in contents
    # Entity nodes survive as benign orphans; only the text node is removed
    assert "Socrates" in contents and "Athens" in contents
    # Edges incident to the deleted text are gone
    assert await graph.get_all_edges() == []


@pytest.mark.asyncio
async def test_delete_missing_returns_false(graph):
    assert await graph.delete("nothing was ever added") is False


@pytest.mark.asyncio
async def test_supersede_replaces_stale_fact(graph):
    await graph.add_text("I live in Amsterdam.", extractor=_home_extractor)

    entities = await graph.supersede(
        "I live in Amsterdam.", "I live in Rotterdam.", extractor=_home_extractor
    )
    assert entities == ["Rotterdam", "home"]

    contents = {n.content for n in await graph.get_all_nodes()}
    assert "I live in Amsterdam." not in contents
    assert "I live in Rotterdam." in contents

    # The superseded fact can no longer resurface in a query
    results = await graph.query("Where do I live?", top_k=5, hops=2)
    assert "I live in Amsterdam." not in results


@pytest.mark.asyncio
async def test_supersede_preserves_shared_entity(graph):
    await graph.add_text("I live in Amsterdam.", extractor=_home_extractor)
    await graph.supersede(
        "I live in Amsterdam.", "I live in Rotterdam.", extractor=_home_extractor
    )

    # The shared 'home' entity survives and now bridges only to the new text
    neighbor_contents = {
        n["content"] for n in await graph.backend.get_neighbors("home")
    }
    assert "I live in Rotterdam." in neighbor_contents
    assert "I live in Amsterdam." not in neighbor_contents


@pytest.mark.asyncio
async def test_supersede_same_content_keeps_node(graph):
    """A no-op supersede (old == new) must (re)add and keep the fact, not delete it."""
    await graph.add_text("I live in Amsterdam.", extractor=_home_extractor)

    entities = await graph.supersede(
        "I live in Amsterdam.", "I live in Amsterdam.", extractor=_home_extractor
    )
    assert entities == ["Amsterdam", "home"]

    contents = {n.content for n in await graph.get_all_nodes()}
    assert "I live in Amsterdam." in contents


def test_delete_and_supersede_sync():
    """The delete_sync / supersede_sync wrappers work outside an event loop.

    Uses MemoryBackend so repeated asyncio.run() calls (one per _sync call)
    don't hit loop-bound resources.
    """
    g = ReasonGraph(backend=MemoryBackend())
    g.embeddings.encode = _fake_encode
    g.embeddings.encode_batch = _fake_encode_batch
    g.embeddings.rerank = _fake_rerank
    g.initialize_sync()
    try:
        g.add_text_sync("I live in Amsterdam.", extractor=_home_extractor)

        entities = g.supersede_sync(
            "I live in Amsterdam.", "I live in Rotterdam.", extractor=_home_extractor
        )
        assert entities == ["Rotterdam", "home"]

        assert g.delete_sync("I live in Rotterdam.") is True
        assert g.delete_sync("never added") is False
    finally:
        g.close_sync()


@pytest.mark.asyncio
async def test_pluggable_embedder_via_constructor():
    """ReasonGraph accepts a caller-supplied encoder (no second model stack)."""
    def embed(x):
        return [_fake_encode(t) for t in x] if isinstance(x, list) else _fake_encode(x)

    g = ReasonGraph(backend=MemoryBackend(), embed_model=embed)
    async with g:
        await g.add_text(
            "Socrates was a philosopher in Athens.", extractor=_fake_extractor
        )
        nodes = await g.get_all_nodes()
        contents = {n.content for n in nodes}
        assert "Socrates was a philosopher in Athens." in contents
        # Embeddings came from the pluggable encoder, normalized to plain lists
        assert all(isinstance(n.embedding, list) for n in nodes)


# -- auto-forget hook (maybe_forget) --

def _noop_embed(x):
    """Trivial embedder so ReasonGraph construction loads no real model."""
    return [[0.0] for _ in x] if isinstance(x, list) else [0.0]


@pytest.mark.asyncio
async def test_maybe_forget_disabled_by_default():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_noop_embed, forget_after=30)
    async with g:
        await g.add_nodes([("old fact", "text")])
        g.backend._nodes["old fact"].last_accessed = datetime.now() - timedelta(days=60)

        # forget_every is None -> disabled: no sweep even though the node is stale
        assert await g.maybe_forget() == 0
        assert len(await g.get_all_nodes()) == 1


@pytest.mark.asyncio
async def test_maybe_forget_throttles_by_interval():
    g = ReasonGraph(
        backend=MemoryBackend(), embed_model=_noop_embed,
        forget_after=30, forget_every=3600,
    )
    async with g:
        await g.add_nodes([("old fact", "text")])
        g.backend._nodes["old fact"].last_accessed = datetime.now() - timedelta(days=60)

        # First call runs the sweep and removes the stale node
        assert await g.maybe_forget() == 1
        assert len(await g.get_all_nodes()) == 0

        # A fresh stale node, but an immediate second call is throttled
        await g.add_nodes([("another old", "text")])
        g.backend._nodes["another old"].last_accessed = datetime.now() - timedelta(days=60)
        assert await g.maybe_forget() == 0
        assert len(await g.get_all_nodes()) == 1

        # Once the interval has elapsed, the sweep runs again
        g._last_forget = datetime.now() - timedelta(seconds=4000)
        assert await g.maybe_forget() == 1
        assert len(await g.get_all_nodes()) == 0


def test_maybe_forget_sync():
    g = ReasonGraph(
        backend=MemoryBackend(), embed_model=_noop_embed,
        forget_after=30, forget_every=3600,
    )
    g.initialize_sync()
    try:
        g.add_nodes_sync([("old", "text")])
        g.backend._nodes["old"].last_accessed = datetime.now() - timedelta(days=60)

        assert g.maybe_forget_sync() == 1
        assert len(g.backend._nodes) == 0
    finally:
        g.close_sync()


# -- recency-weighted ranking --

@pytest.mark.asyncio
async def test_query_recency_weight_prefers_newer():
    """With relevance tied, recency_weight=1 orders newer facts first."""
    def same_vec(x):
        v = [1.0, 0.0, 0.0, 0.0]
        return [v for _ in x] if isinstance(x, list) else v

    class FlatReranker:
        def predict(self, pairs):
            return [0.0] * len(pairs)

    g = ReasonGraph(backend=MemoryBackend(), embed_model=same_vec)
    async with g:
        await g.add_nodes([("old truth", "text"), ("new truth", "text")])
        g.backend._nodes["old truth"].created_at = datetime(2020, 1, 1)
        g.backend._nodes["new truth"].created_at = datetime(2026, 1, 1)
        g.embeddings._rerank = FlatReranker()  # avoid loading a real cross-encoder

        results = await g.query("truth", top_k=5, hops=1, recency_weight=1.0)
        assert results[:2] == ["new truth", "old truth"]


@pytest.mark.asyncio
async def test_query_invalid_recency_weight_raises():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_noop_embed)
    async with g:
        with pytest.raises(ValueError):
            await g.query("x", recency_weight=1.5)


# -- multi-label scopes (shared graph) --

@pytest.mark.asyncio
async def test_scopes_filter_seeds_but_reasoning_crosses(graph):
    # Two facts under different scopes, bridged by a shared 'Zeus' entity
    await graph.add_text("Fact A about Zeus.", extractor=lambda t: ["Zeus"], scopes=["user-1"])
    await graph.add_text("Fact B about Zeus.", extractor=lambda t: ["Zeus"], scopes=["topic-myth"])

    # Seeds are drawn only from user-1, but traversal follows the shared entity
    # across scopes, so the topic-myth fact is still reached.
    results = await graph.query("Zeus", top_k=5, hops=3, scopes=["user-1"])
    assert "Fact A about Zeus." in results
    assert "Fact B about Zeus." in results


@pytest.mark.asyncio
async def test_discover_returns_cross_session_paths(graph):
    # Two sessions, bridged only by the shared 'Zeus' entity.
    await graph.add_text("Fact A about Zeus.", extractor=lambda t: ["Zeus"], scopes=["session-1"])
    await graph.add_text("Fact B about Zeus.", extractor=lambda t: ["Zeus"], scopes=["session-2"])

    found = await graph.discover("Zeus", top_k=5, hops=3, scopes=["session-1"])
    by_content = {f["content"]: f for f in found}
    assert "Fact A about Zeus." in by_content
    assert "Fact B about Zeus." in by_content

    # The session-2 fact is a cross-session discovery reached via the Zeus entity.
    b = by_content["Fact B about Zeus."]
    assert b["cross_session"] is True
    assert b["scopes"] == ["session-2"]
    assert any(step.get("entity") == "Zeus" for step in b["path"])
    # The path ends at the discovered fact and includes the seed fact.
    assert b["path"][-1]["content"] == "Fact B about Zeus."
    assert any(step.get("content") == "Fact A about Zeus." for step in b["path"])

    # A seed-scope fact is not a cross-session discovery.
    assert by_content["Fact A about Zeus."]["cross_session"] is False


@pytest.mark.asyncio
async def test_answer_uses_pluggable_synthesizer():
    # A fake synthesizer that rephrases retrieved facts into one line.
    def synth(query, context):
        facts = [step["content"] for item in context for step in item["path"] if "content" in step]
        uniq = list(dict.fromkeys(facts))
        return f"Q:{query} | facts:{len(uniq)} | " + " + ".join(uniq)

    g = ReasonGraph(backend=MemoryBackend(), embed_model=_noop_embed, synthesizer=synth)
    async with g:
        await g.add_text("Fact A about Zeus.", extractor=lambda t: ["Zeus"], scopes=["s1"])
        await g.add_text("Fact B about Zeus.", extractor=lambda t: ["Zeus"], scopes=["s2"])

        text = await g.answer("Zeus", scopes=["s1"], hops=3)
        assert isinstance(text, str)
        assert "Fact A about Zeus." in text
        assert "Fact B about Zeus." in text  # cross-session fact folded into the answer


@pytest.mark.asyncio
async def test_answer_without_synthesizer_raises():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_noop_embed)
    async with g:
        with pytest.raises(RuntimeError, match="synthesizer"):
            await g.answer("anything")


@pytest.mark.asyncio
async def test_scopes_isolate_seeds_without_a_bridge(graph):
    # No shared entity => no bridge, so a scoped query cannot reach the other scope
    await graph.add_text("Private A fact.", extractor=lambda t: [], scopes=["user-1"])
    await graph.add_text("Private B fact.", extractor=lambda t: [], scopes=["user-2"])

    results = await graph.query("fact", top_k=5, hops=3, scopes=["user-1"])
    assert "Private A fact." in results
    assert "Private B fact." not in results
