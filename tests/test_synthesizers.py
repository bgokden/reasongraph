import pytest

from reasongraph._synthesizers import (
    TemplateSynthesizer,
    PromptSynthesizer,
    TransformersSynthesizer,
)
from reasongraph.graph import ReasonGraph
from reasongraph.backends._memory import MemoryBackend

def _stable_hash(text):
    """Process-independent 64-bit hash (Python's hash() is randomized per run,
    which made fake embeddings and therefore test outcomes flaky)."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")



# A discover()-shaped context: one seed-session fact and one cross-session fact
# reached through the shared "Nvidia" entity.
CONTEXT = [
    {
        "content": "Nvidia designs AI chips.",
        "scopes": ["markets"],
        "cross_session": False,
        "path": [{"content": "Nvidia designs AI chips.", "scopes": ["markets"]}],
    },
    {
        "content": "TSMC makes the chips Nvidia designs.",
        "scopes": ["supply"],
        "cross_session": True,
        "path": [
            {"content": "Nvidia designs AI chips.", "scopes": ["markets"]},
            {"entity": "Nvidia"},
            {"content": "TSMC makes the chips Nvidia designs.", "scopes": ["supply"]},
        ],
    },
]


# -- TemplateSynthesizer --

def test_template_names_facts_and_bridges():
    out = TemplateSynthesizer().synthesize("Nvidia", CONTEXT)
    assert "Nvidia designs AI chips." in out
    assert "TSMC makes the chips Nvidia designs." in out
    # The cross-session fact is flagged and the bridging entity is named
    assert "(from another session)" in out
    assert "Nvidia" in out.splitlines()[-1]


def test_template_empty_context():
    assert TemplateSynthesizer().synthesize("ghosts", []) == "No stored facts connect to 'ghosts'."


def test_template_dedupes_and_caps():
    dup = CONTEXT + CONTEXT
    out = TemplateSynthesizer(max_facts=1).synthesize("Nvidia", dup)
    # max_facts caps the fact lines (header + 1 fact + bridge line)
    fact_lines = [line for line in out.splitlines() if line.startswith("- ")]
    assert len(fact_lines) == 1


def test_template_is_callable():
    synth = TemplateSynthesizer()
    assert synth("Nvidia", CONTEXT) == synth.synthesize("Nvidia", CONTEXT)


# -- PromptSynthesizer --

def test_prompt_build_includes_facts_and_connections():
    prompt = PromptSynthesizer(lambda p: p).build_prompt("Nvidia", CONTEXT)
    assert "Question: Nvidia" in prompt
    assert "- Nvidia designs AI chips." in prompt
    assert "- TSMC makes the chips Nvidia designs." in prompt
    assert "Cross-session connections:" in prompt
    assert "connects via Nvidia" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_prompt_calls_generate_with_prompt():
    captured = {}

    def generate(prompt):
        captured["prompt"] = prompt
        return "SYNTHESIZED"

    out = PromptSynthesizer(generate).synthesize("Nvidia", CONTEXT)
    assert out == "SYNTHESIZED"
    assert "Question: Nvidia" in captured["prompt"]


def test_prompt_custom_instruction():
    synth = PromptSynthesizer(lambda p: p, instruction="BE TERSE.")
    assert synth.build_prompt("q", CONTEXT).startswith("BE TERSE.")


def test_prompt_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        PromptSynthesizer("not callable")


@pytest.mark.asyncio
async def test_prompt_supports_async_generate():
    async def generate(prompt):
        return "ASYNC OK"

    result = PromptSynthesizer(generate).synthesize("Nvidia", CONTEXT)
    # An async generate makes synthesize return an awaitable (ReasonGraph.answer awaits it)
    assert await result == "ASYNC OK"


# -- TransformersSynthesizer (parsing only; no model load) --

def test_transformers_extract_reply_chat_list():
    output = [{"generated_text": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "  the answer  "},
    ]}]
    assert TransformersSynthesizer._extract_reply(output) == "the answer"


def test_transformers_extract_reply_plain_string():
    output = [{"generated_text": "plain completion"}]
    assert TransformersSynthesizer._extract_reply(output) == "plain completion"


# -- Integration with ReasonGraph.answer --

def _fake_encode(text):
    h = _stable_hash(text)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def _fake_embed(x):
    return [_fake_encode(t) for t in x] if isinstance(x, list) else _fake_encode(x)


@pytest.mark.asyncio
async def test_prompt_synthesizer_drives_answer():
    captured = {}

    def generate(prompt):
        captured["prompt"] = prompt
        return "FINAL ANSWER"

    g = ReasonGraph(
        backend=MemoryBackend(),
        embed_model=_fake_embed,
        synthesizer=PromptSynthesizer(generate),
        causal_extractor=False,  # no real causal model in unit tests
    )
    async with g:
        await g.add_text("Fact A about Zeus.", extractor=lambda t: ["Zeus"], scopes=["s1"])
        await g.add_text("Fact B about Zeus.", extractor=lambda t: ["Zeus"], scopes=["s2"])

        answer = await g.answer("Zeus", scopes=["s1"], hops=3)
        assert answer == "FINAL ANSWER"
        # The cross-session fact reached the prompt the model saw
        assert "Fact B about Zeus." in captured["prompt"]
