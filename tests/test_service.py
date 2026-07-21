import pytest

from reasongraph.service import MemoryService
from reasongraph.backends._memory import MemoryBackend


def _fake_encode(text):
    h = hash(text)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def _fake_embed(x):
    return [_fake_encode(t) for t in x] if isinstance(x, list) else _fake_encode(x)


def _zeus_extractor(text):
    return ["Zeus"] if "Zeus" in text else []


def _fake_synth(query, context):
    facts = [c["content"] for c in context]
    return f"About {query}: " + "; ".join(facts)


@pytest.fixture
async def svc():
    s = MemoryService(
        backend=MemoryBackend(),
        embed_model=_fake_embed,
        extractor=_zeus_extractor,
        synthesizer=_fake_synth,
    )
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_push_stats_sessions(svc):
    await svc.push("agent-1", "Zeus is king of the gods.")
    st = await svc.stats()
    assert st["facts"] == 1
    assert st["sessions"] == 1
    assert await svc.list_sessions() == ["agent-1"]


@pytest.mark.asyncio
async def test_cross_session_discovery(svc):
    # Two agents, separate sessions, bridged by the shared Zeus entity.
    await svc.push("agent-1", "Zeus threw lightning bolts.")
    await svc.push("agent-2", "Zeus lived on Mount Olympus.")

    found = await svc.discover("Zeus", session="agent-1", hops=3)
    by_content = {f["content"]: f for f in found}
    assert "Zeus threw lightning bolts." in by_content
    assert "Zeus lived on Mount Olympus." in by_content

    other = by_content["Zeus lived on Mount Olympus."]
    assert other["cross_session"] is True  # discovered from agent-2's session
    assert any(step.get("entity") == "Zeus" for step in other["path"])


@pytest.mark.asyncio
async def test_answer_synthesizes(svc):
    await svc.push("agent-1", "Zeus threw lightning bolts.")
    text = await svc.answer("Zeus", session="agent-1")
    assert isinstance(text, str)
    assert "Zeus threw lightning bolts." in text


@pytest.mark.asyncio
async def test_supersede_removes_old(svc):
    await svc.push("agent-1", "Zeus is mortal.")
    await svc.supersede("agent-1", "Zeus is mortal.", "Zeus is immortal.")
    facts = await svc.query("Zeus", session="agent-1")
    assert "Zeus is mortal." not in facts
    assert "Zeus is immortal." in facts
