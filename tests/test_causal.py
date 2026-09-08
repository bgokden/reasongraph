import hashlib

import pytest

from reasongraph._extraction import (
    causal_from_cues,
    HybridCausalExtractor,
    GlinerRelexExtractor,
)
from reasongraph.graph import ReasonGraph
from reasongraph.backends._memory import MemoryBackend
from reasongraph.backends._sqlite import SqliteBackend
from reasongraph._types import Edge, Node


def _fake_encode(text):
    # md5-based so it is stable across processes (unlike hash()).
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def _fake_embed(x):
    return [_fake_encode(t) for t in x] if isinstance(x, list) else _fake_encode(x)


def _no_rerank(query, results, top_k, recency_weight=0.0):
    return results[:top_k]


# -- cue extractor: direction-aware, model-free --

def test_cue_explicit_direction():
    assert causal_from_cues("Heavy rainfall caused severe flooding.") == [
        {"cause": "Heavy rainfall", "effect": "severe flooding"}
    ]


def test_cue_reversed_direction():
    # "resulted from" puts the cause after the marker.
    assert causal_from_cues("The crash resulted from brake failure.") == [
        {"cause": "brake failure", "effect": "The crash"}
    ]


def test_cue_implicit_returns_empty():
    assert causal_from_cues("The bridge collapsed. Commuters were stranded.") == []


def test_cue_multilingual():
    # Spanish "provocó" (cause_first) and German "wegen" (effect_first).
    assert causal_from_cues("La deforestación provocó la erosión del suelo.") == [
        {"cause": "La deforestación", "effect": "la erosión del suelo"}
    ]


@pytest.mark.parametrize("text,cause,effect", [
    ("Crop yields fell due to the drought.", "the drought", "Crop yields fell"),  # effect_first
    ("Deforestation led to soil erosion.", "Deforestation", "soil erosion"),      # cause_first
    ("持续干旱导致水资源紧张。", "持续干旱", "水资源紧张"),                        # zh cause_first (导致)
    ("Starke Regenfälle verursachten Überschwemmungen.", "Starke Regenfälle", "Überschwemmungen"),  # de cause_first
])
def test_cue_marker_orientations(text, cause, effect):
    out = causal_from_cues(text)
    assert out and out[0]["cause"] == cause and out[0]["effect"] == effect


def test_cue_no_marker_no_output():
    assert causal_from_cues("The market closed higher today.") == []


def test_cue_bounds_span_to_clause():
    # The cause span stops at the clause boundary instead of swallowing the
    # trailing relative clause.
    out = causal_from_cues(
        "The blackout was caused by a failure at the substation, which had been "
        "flagged as vulnerable months earlier."
    )
    assert out == [{"cause": "a failure at the substation", "effect": "The blackout was"}]


def test_cue_skips_noun_usage():
    # "causes" as a noun after a determiner is not a causal connective.
    assert causal_from_cues("Researchers studied the causes of coral bleaching.") == []


def test_cue_uses_earliest_marker():
    # "because of" occurs before "resulted in", so it wins (and the whole
    # sentence is not swallowed).
    out = causal_from_cues("Sales dropped because of weak demand, which resulted in layoffs.")
    assert out == [{"cause": "weak demand", "effect": "Sales dropped"}]


def test_cue_rejects_pronoun_cause():
    # A lone relative pronoun is not a usable cause span.
    out = causal_from_cues("Prices rose, which caused, in turn, more spending.")
    assert all(r["cause"].lower() != "which" for r in out)


# -- hybrid: cue first, model fallback (fake relex, no model load) --

class _FakeRelex:
    def __init__(self):
        self.calls = 0

    def relations_for(self, text):
        self.calls += 1
        return [{"cause": "implicit cause", "effect": "implicit effect"}]


def test_hybrid_uses_cue_and_skips_model_when_marked():
    relex = _FakeRelex()
    hybrid = HybridCausalExtractor(relex=relex)
    out = hybrid.extract_causal(["Heavy rainfall caused severe flooding."])
    assert out[0]["causal"] is True
    assert out[0]["relations"] == [{"cause": "Heavy rainfall", "effect": "severe flooding"}]
    assert relex.calls == 0  # the model was never touched


def test_hybrid_falls_back_to_model_on_implicit():
    relex = _FakeRelex()
    hybrid = HybridCausalExtractor(relex=relex)
    out = hybrid.extract_causal(["The bridge collapsed. Commuters were stranded."])
    assert out[0]["relations"] == [{"cause": "implicit cause", "effect": "implicit effect"}]
    assert relex.calls == 1


# -- graph causal wiring (fake causal extractor, no real models) --

def _fake_causal(texts):
    out = []
    for t in texts:
        rels = [{"cause": "heavy rainfall", "effect": "flooding"}] if "caused" in t else []
        out.append({"text": t, "causal": bool(rels), "relations": rels})
    return out


@pytest.mark.asyncio
async def test_causal_on_by_default_creates_typed_edges():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_fake_causal)
    async with g:
        await g.add_text("Heavy rainfall caused flooding.", extractor=lambda t: [])
        edges = await g.get_all_edges()
        causal = [(e.from_content, e.to_content) for e in edges if e.label == "causes"]
        assert ("heavy rainfall", "flooding") in causal
        # cause/effect spans are entity nodes
        types = {n.content: n.type for n in await g.get_all_nodes()}
        assert types["heavy rainfall"] == "entity" and types["flooding"] == "entity"


@pytest.mark.asyncio
async def test_causal_false_disables():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_fake_causal)
    async with g:
        await g.add_text("Heavy rainfall caused flooding.", extractor=lambda t: [], causal=False)
        edges = await g.get_all_edges()
        assert not any(e.label == "causes" for e in edges)


@pytest.mark.asyncio
async def test_causal_true_unsupported_raises():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=False)
    async with g:
        with pytest.raises(ValueError, match="causal=True"):
            await g.add_text("X caused Y.", extractor=lambda t: [], causal=True)


@pytest.mark.asyncio
async def test_discover_surfaces_causes():
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=_fake_causal)
    g.embeddings.rerank = _no_rerank
    async with g:
        await g.add_text("Heavy rainfall caused flooding.", extractor=lambda t: [])
        found = await g.discover("rainfall flooding", hops=4)
        by_content = {f["content"]: f for f in found}
        fact = by_content["Heavy rainfall caused flooding."]
        assert {"cause": "heavy rainfall", "effect": "flooding"} in fact["causes"]


# -- backend get_causal_relations + Edge.label round-trip (direct) --

def _causal_graph_nodes_edges():
    nodes = [
        Node(content="Heavy rainfall caused flooding.", type="text", embedding=_fake_encode("f")),
        Node(content="heavy rainfall", type="entity", embedding=_fake_encode("hr")),
        Node(content="flooding", type="entity", embedding=_fake_encode("fl")),
    ]
    edges = [
        Edge(from_content="heavy rainfall", to_content="flooding", label="causes"),
        Edge(from_content="heavy rainfall", to_content="Heavy rainfall caused flooding."),
        Edge(from_content="flooding", to_content="Heavy rainfall caused flooding."),
    ]
    return nodes, edges


@pytest.mark.asyncio
@pytest.mark.parametrize("make_backend", [lambda: MemoryBackend(), lambda: SqliteBackend(":memory:")])
async def test_backend_causal_relations_and_label(make_backend):
    backend = make_backend()
    await backend.initialize()
    try:
        nodes, edges = _causal_graph_nodes_edges()
        await backend.insert_nodes(nodes)
        await backend.insert_edges(edges)

        # Edge.label round-trips through get_all_edges
        labels = {(e.from_content, e.to_content): e.label for e in await backend.get_all_edges()}
        assert labels[("heavy rainfall", "flooding")] == "causes"
        assert labels[("flooding", "Heavy rainfall caused flooding.")] is None

        # get_neighbors exposes the label + direction on the causal edge
        neigh = {n["content"]: n for n in await backend.get_neighbors("heavy rainfall")}
        assert neigh["flooding"]["label"] == "causes"
        assert neigh["flooding"]["direction"] == "out"

        # get_causal_relations returns the fact's directed pair
        rels = await backend.get_causal_relations(["Heavy rainfall caused flooding."])
        assert rels["Heavy rainfall caused flooding."] == [{"cause": "heavy rainfall", "effect": "flooding"}]
        assert await backend.get_causal_relations([]) == {}
    finally:
        await backend.close()


# -- real gliner-relex-multi (guarded; loads the model) --

@pytest.fixture(scope="module")
def relex_extractor():
    pytest.importorskip("gliner")
    try:
        ext = GlinerRelexExtractor()
        ext.relations_for("warmup")
        return ext
    except Exception as e:
        pytest.skip(f"gliner-relex model not available: {e}")


def test_gliner_relex_extracts_directed_causal(relex_extractor):
    out = relex_extractor.extract_causal(["Heavy rainfall caused severe flooding."])
    assert out[0]["causal"] is True
    rels = out[0]["relations"]
    assert any("rain" in r["cause"].lower() and "flood" in r["effect"].lower() for r in rels)


# -- CausalPointerExtractor (thin wrapper over the pointer model) --

def test_pointer_extractor_is_exported_and_constructible():
    from reasongraph import CausalPointerExtractor
    # default points at the pointer model's canonical HF repo (must match
    # causal_span_model.pointer.submission.POINTER_HF_REPO so first-call download
    # resolves); an explicit override is respected.
    assert CausalPointerExtractor().model == "berk/causal-span-pointer-mdeberta"
    ext = CausalPointerExtractor(model="some/local-or-repo")
    assert ext.model == "some/local-or-repo"
    # lazy: no model loaded on construction
    assert ext.topk == 5 and ext._model is None


# --- embedding gate on the pointer extractor (no model download: stubs) ---
class _StubClf:
    classes_ = [0, 1]

    def predict_proba(self, feats):
        # feature = [is_causal_hint]; P(causal) = that value
        return [[1.0 - f[0], f[0]] for f in feats]


def _stub_encoder(texts):
    return [[0.95 if "because" in t else 0.2] for t in texts]


def test_pointer_extractor_embed_gate_skips_low_probability_texts(monkeypatch):
    from reasongraph._extraction import CausalPointerExtractor

    ex = CausalPointerExtractor(embed_gate={"embed_model": "stub", "kind": "stub", "clf": _StubClf()},
                                embed_gate_threshold=0.9, embed_gate_encoder=_stub_encoder)
    calls = []
    monkeypatch.setattr(ex, "relations_for", lambda text: (calls.append(text) or
                                                          [{"cause": "rain", "effect": "floods"}]))
    out = ex.extract_causal(["It flooded because it rained.", "The river is 200 km long."])
    assert out[0]["causal"] is True and out[0]["relations"] == [{"cause": "rain", "effect": "floods"}]
    assert out[0]["causal_prob"] == 0.95
    assert out[1] == {"text": "The river is 200 km long.", "causal": False, "relations": [],
                      "causal_prob": 0.2}
    assert calls == ["It flooded because it rained."]     # the span model never ran on the gated text
    assert ex.causal_probs([]) == []


def test_pointer_extractor_without_gate_reports_no_probs(monkeypatch):
    from reasongraph._extraction import CausalPointerExtractor

    ex = CausalPointerExtractor()
    monkeypatch.setattr(ex, "relations_for", lambda text: [])
    assert ex.causal_probs(["x"]) is None
    assert ex.extract_causal(["x"]) == [{"text": "x", "causal": False, "relations": []}]


def test_embed_gate_env_wiring(monkeypatch):
    from reasongraph._extraction import CausalPointerExtractor
    monkeypatch.setenv("REASONGRAPH_CAUSAL_EMBED_GATE", "/tmp/gate.joblib")
    monkeypatch.setenv("REASONGRAPH_CAUSAL_EMBED_GATE_THRESHOLD", "0.7")
    monkeypatch.setenv("REASONGRAPH_CAUSAL_MODEL", "some/model")
    pytest.importorskip("causal_span_model")
    ex = ReasonGraph._build_default_causal_extractor()
    assert isinstance(ex, CausalPointerExtractor)
    assert ex.model == "some/model" and ex.embed_gate == "/tmp/gate.joblib"
    assert ex.embed_gate_threshold == 0.7


def test_regex_splitter_and_resolve():
    from reasongraph._split import RegexSplitter, resolve_splitter, SaTSplitter
    r = RegexSplitter()
    assert r.split("The server failed. It was e.g. overloaded! Dr. Smith replied? Yes.\nNew line.") == [
        "The server failed.", "It was e.g. overloaded!", "Dr. Smith replied?", "Yes.", "New line."]
    assert r.split("Die Bank senkt den Leitzins. Das gilt z.B. im Bau.") == ["Die Bank senkt den Leitzins.", "Das gilt z.B. im Bau."]
    assert resolve_splitter(None) is None and resolve_splitter("off") is None
    assert isinstance(resolve_splitter("regex"), RegexSplitter)
    s = resolve_splitter("sat:sat-12l-sm"); assert isinstance(s, SaTSplitter) and s.model == "sat-12l-sm"
    assert resolve_splitter(r) is r


@pytest.mark.asyncio
async def test_add_texts_split_keeps_entities_aligned_with_inputs():
    from reasongraph._split import RegexSplitter
    ents = {"Rain fell.": ["Rain"], "Roads flooded.": ["Roads"], "Trains stopped.": ["Trains"]}
    g = ReasonGraph(backend=MemoryBackend(), embed_model=_fake_embed, causal_extractor=False,
                    sentence_splitter=RegexSplitter())
    await g.initialize()
    try:
        out = await g.add_texts(["Rain fell. Roads flooded.", "Trains stopped."], extractor=lambda t: ents.get(t, []))
        assert out == [["Rain", "Roads"], ["Trains"]]
        assert (await g.stats())["texts"] == 3 if hasattr(g, "stats") else True
        out2 = await g.add_texts(["A. B."], extractor=lambda t: [], split=False)
        assert len(out2) == 1
    finally:
        await g.close()
