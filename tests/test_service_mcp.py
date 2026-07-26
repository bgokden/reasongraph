import json

import pytest

pytest.importorskip("mcp")

from reasongraph.service import MemoryService
from reasongraph.service.mcp_server import create_mcp
from reasongraph.backends._memory import MemoryBackend


def _fake_encode(text):
    h = hash(text)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def _fake_embed(x):
    return [_fake_encode(t) for t in x] if isinstance(x, list) else _fake_encode(x)


def _zeus_extractor(text):
    return ["Zeus"] if "Zeus" in text else []


def _fake_synth(query, context):
    return f"About {query}: " + "; ".join(c["content"] for c in context)


def _unwrap(result):
    """Normalize FastMCP call_tool output to the tool's raw return value.

    call_tool returns either a list of content blocks, or a
    (content_blocks, structured_dict) tuple when the tool has structured output.
    """
    if isinstance(result, tuple):
        structured = result[1]
        if isinstance(structured, dict):
            return structured.get("result", structured)
        return structured
    return json.loads(result[0].text)


async def _service():
    svc = MemoryService(
        backend=MemoryBackend(), embed_model=_fake_embed,
        extractor=_zeus_extractor, synthesizer=_fake_synth, causal_extractor=False,
    )
    await svc.initialize()
    return svc


@pytest.mark.asyncio
async def test_mcp_lists_all_tools():
    svc = await _service()
    try:
        mcp = create_mcp(svc)
        names = {t.name for t in await mcp.list_tools()}
        assert names == {
            "push_memory", "query_memory", "query_memory_detailed",
            "discover_connections", "answer", "update_memory", "delete_memory",
            "forget_stale", "list_sessions",
        }
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_mcp_update_and_delete_memory():
    svc = await _service()
    try:
        mcp = create_mcp(svc)
        await mcp.call_tool("push_memory", {"session": "a", "text": "Zeus is mortal."})
        # self-correction: replace the stale fact
        res = _unwrap(await mcp.call_tool("update_memory", {
            "session": "a", "old_text": "Zeus is mortal.", "new_text": "Zeus is immortal.",
        }))
        assert res["superseded"] is True
        facts = _unwrap(await mcp.call_tool("query_memory", {"query": "Zeus", "session": "a"}))
        assert "Zeus is immortal." in facts and "Zeus is mortal." not in facts
        # delete by exact text
        d = _unwrap(await mcp.call_tool("delete_memory", {"text": "Zeus is immortal."}))
        assert d["deleted"] is True
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_mcp_query_detailed_has_scores():
    svc = await _service()
    try:
        mcp = create_mcp(svc)
        await mcp.call_tool("push_memory", {"session": "a", "text": "Zeus rules the sky."})
        rows = _unwrap(await mcp.call_tool(
            "query_memory_detailed", {"query": "Zeus", "session": "a"}
        ))
        assert rows and isinstance(rows[0], dict)
        assert set(rows[0]) == {"content", "score", "created_at", "scopes"}
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_mcp_push_query_and_cross_session_discovery():
    svc = await _service()
    try:
        mcp = create_mcp(svc)
        pushed = _unwrap(await mcp.call_tool(
            "push_memory", {"session": "agent-1", "text": "Zeus threw lightning bolts."}
        ))
        assert pushed["session"] == "agent-1" and "Zeus" in pushed["entities"]
        await mcp.call_tool(
            "push_memory", {"session": "agent-2", "text": "Zeus lived on Mount Olympus."}
        )

        facts = _unwrap(await mcp.call_tool(
            "query_memory", {"query": "Zeus", "session": "agent-1"}
        ))
        assert "Zeus threw lightning bolts." in facts

        connections = _unwrap(await mcp.call_tool(
            "discover_connections", {"query": "Zeus", "session": "agent-1"}
        ))
        by_content = {c["content"]: c for c in connections}
        assert "Zeus lived on Mount Olympus." in by_content
        # agent-2's fact is a cross-session discovery through the shared Zeus entity
        assert by_content["Zeus lived on Mount Olympus."]["cross_session"] is True
    finally:
        await svc.close()


@pytest.mark.asyncio
async def test_mcp_answer_and_list_sessions():
    svc = await _service()
    try:
        mcp = create_mcp(svc)
        await mcp.call_tool("push_memory", {"session": "a", "text": "Zeus is a god."})
        await mcp.call_tool("push_memory", {"session": "b", "text": "Zeus rules Olympus."})

        answer = _unwrap(await mcp.call_tool("answer", {"query": "Zeus", "session": "a"}))
        assert isinstance(answer, str) and "Zeus" in answer

        sessions = _unwrap(await mcp.call_tool("list_sessions", {}))
        assert set(sessions) == {"a", "b"}
    finally:
        await svc.close()
