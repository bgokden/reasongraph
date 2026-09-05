"""walk_scopes: seed from one scope, walk a wider boundary, never beyond it."""

import pytest

from reasongraph import ReasonGraph
from reasongraph.backends._memory import MemoryBackend


def _enc(text):
    h = hash(text)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def _embed(x):
    return [_enc(t) for t in x] if isinstance(x, list) else _enc(x)


def _extract(text):
    return ["Zeus"] if "Zeus" in text else []


@pytest.fixture
async def graph():
    async with ReasonGraph(backend=MemoryBackend(), embed_model=_embed, causal_extractor=False) as g:
        # tenant A, two sessions; tenant B, one session. All bridged by "Zeus".
        await g.add_texts(["Zeus threw lightning bolts."], extractor=_extract, scopes=["a/s1", "a"])
        await g.add_texts(["Zeus lived on Mount Olympus."], extractor=_extract, scopes=["a/s2", "a"])
        await g.add_texts(["Zeus is the king of the gods."], extractor=_extract, scopes=["b/s1", "b"])
        yield g


async def test_query_walks_within_walk_scopes_only(graph):
    # isolate alone: confined to the seed session
    only_s1 = await graph.query("Zeus", scopes=["a/s1"], isolate=True, top_k=10)
    assert only_s1 == ["Zeus threw lightning bolts."]
    # no isolation: crosses everything (the library default)
    everything = await graph.query("Zeus", scopes=["a/s1"], top_k=10)
    assert "Zeus is the king of the gods." in everything
    # walk_scopes: seed from s1, walk tenant a, never tenant b
    tenant_a = await graph.query("Zeus", scopes=["a/s1"], walk_scopes={"a"}, top_k=10)
    assert set(tenant_a) == {"Zeus threw lightning bolts.", "Zeus lived on Mount Olympus."}


async def test_discover_flags_cross_session_inside_boundary(graph):
    conns = await graph.discover("Zeus", scopes=["a/s1"], walk_scopes={"a"}, top_k=10)
    contents = {c["content"] for c in conns}
    assert "Zeus lived on Mount Olympus." in contents
    assert "Zeus is the king of the gods." not in contents
    assert any(c["cross_session"] for c in conns if c["content"] == "Zeus lived on Mount Olympus.")


async def test_detailed_and_sync_wrappers_accept_walk_scopes(graph):
    detailed = await graph.query_detailed("Zeus", scopes=["a/s1"], walk_scopes={"a"}, top_k=10)
    assert {d["content"] for d in detailed} == {"Zeus threw lightning bolts.", "Zeus lived on Mount Olympus."}
    # trace/what_if accept it without error on a graph without causal edges
    assert (await graph.trace_effects("Zeus threw lightning bolts.", scopes=["a/s1"], walk_scopes={"a"}))["chain"] == []
    assert "collapsed" in await graph.what_if("Zeus threw lightning bolts.", scopes=["a/s1"], walk_scopes={"a"})
