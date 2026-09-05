"""Service-level features that previously lived only in the graph: dedup on write,
time-travel over HTTP, opt-in conflict resolution, causal chains, and the deferred
worker running models off the event loop."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from reasongraph.backends._memory import MemoryBackend
from reasongraph.service import MemoryService
from reasongraph.service.http import create_app


def _enc(text):
    """Deterministic pseudo-random unit-ish vector per key (distinct keys are
    nearly orthogonal, so only texts sharing a key look like duplicates)."""
    import random
    rng = random.Random(text)
    return [rng.uniform(-1, 1) for _ in range(384)]


class _Embed:
    """Fake embedder with controllable near-duplicates: texts that share a
    ``~key`` prefix embed identically."""
    def __call__(self, x):
        if isinstance(x, list):
            return [self._one(t) for t in x]
        return self._one(x)

    @staticmethod
    def _one(t):
        key = t.split("~", 1)[0] if "~" in t else t
        return _enc(key)


def _zeus(text):
    return ["Zeus"] if "Zeus" in text else []


def _causal(texts):
    out = []
    for t in texts:
        if "->" in t:
            cause, effect = [p.strip() for p in t.split("->", 1)]
            cause = cause.split("~", 1)[1] if "~" in cause else cause   # drop the embedding key
            out.append({"causal": True, "relations": [{"cause": cause, "effect": effect}]})
        else:
            out.append({"causal": False, "relations": []})
    return out


class _Resolver:
    """Declares a contradiction when the new fact says 'now' and the old one does not."""
    calls = 0

    def contradictions(self, new_text, candidates):
        _Resolver.calls += 1
        return [c for c in candidates if "now" in new_text and "now" not in c]


def _service(**kw):
    return MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=_zeus,
                         causal_extractor=False, **kw)


async def test_dedup_merges_paraphrases_sync_and_deferred():
    for deferred in (False, True):
        svc = _service(defer_extraction=deferred, dedup_threshold=0.99)
        async with svc:
            await svc.push("a", "k1~Zeus threw lightning.")
            res = await svc.push("b", "k1~Zeus hurled a lightning bolt.")  # same embedding key
            if deferred:
                await svc._enrich_queue.join()
                assert res.get("duplicate") is True
            facts = await svc.query("k1", top_k=10)
            assert facts == ["k1~Zeus threw lightning."]
            scopes = await svc.graph.backend.get_scopes(["k1~Zeus threw lightning."])
            assert scopes["k1~Zeus threw lightning."] == {"a", "b"}     # scope unioned
            many = await svc.push_many("c", ["k1~Zeus zapped.", "k2~Zeus slept."])
            if deferred:
                await svc._enrich_queue.join()
                assert many["duplicates"] == 1


async def test_conflict_resolution_is_opt_in_when_configured_so():
    svc = MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=_zeus,
                        causal_extractor=False)
    svc.graph.conflict_resolver = _Resolver()
    svc.graph.resolve_conflicts_by_default = False
    async with svc:
        await svc.push("s", "k1~Zeus lives on Olympus.")
        _Resolver.calls = 0
        await svc.push("s", "k1~Zeus now lives in Athens.")            # default: no check
        assert _Resolver.calls == 0
        assert len(await svc.query("k1", top_k=10, session="s")) == 2
        await svc.push("s", "k1~Zeus now lives in Sparta.", resolve_conflicts=True)
        assert _Resolver.calls == 1
        current = await svc.query("k1", top_k=10, session="s")
        assert "k1~Zeus lives on Olympus." not in current and "k1~Zeus now lives in Sparta." in current
        assert "k1~Zeus lives on Olympus." in await svc.query("k1", top_k=10, session="s", include_superseded=True)


async def test_deferred_worker_resolves_conflicts_when_asked():
    svc = MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=_zeus,
                        causal_extractor=False, defer_extraction=True)
    svc.graph.conflict_resolver = _Resolver()
    svc.graph.resolve_conflicts_by_default = False
    async with svc:
        await svc.push("s", "k1~Zeus lives on Olympus.")
        await svc._enrich_queue.join()
        await svc.push("s", "k1~Zeus now lives in Athens.", resolve_conflicts=True)
        await svc._enrich_queue.join()
        assert await svc.query("k1", top_k=10, session="s") == ["k1~Zeus now lives in Athens."]


def test_http_time_travel_and_causal_chain():
    svc = MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=_zeus,
                        causal_extractor=_causal)
    with TestClient(create_app(svc)) as c:
        t0 = datetime.now(timezone.utc)
        assert c.post("/sessions/s/memory", json={"text": "k1~Zeus lives on Olympus."}).status_code == 200
        r = c.post("/supersede", json={"session": "s", "old_text": "k1~Zeus lives on Olympus.",
                                        "new_text": "k1~Zeus lives in Athens."})
        assert r.status_code == 200
        now = c.post("/query", json={"query": "k1", "top_k": 10}).json()["facts"]
        assert now == ["k1~Zeus lives in Athens."]
        # supersede() deletes the old fact outright, so as_of cannot revive it; but a
        # bad timestamp is rejected and a valid one is accepted.
        assert c.post("/query", json={"query": "k1", "as_of": "not-a-date"}).status_code == 422
        past = (t0 - timedelta(days=1)).isoformat()
        assert c.post("/query", json={"query": "k1", "as_of": past}).status_code == 200
        # include_superseded is accepted on query and discover
        assert c.post("/discover", json={"query": "k1", "include_superseded": True}).status_code == 200

        # causal chain over HTTP: A -> B, B -> C ; chain from A to C exists, C to A does not
        for t in ["rain~heavy rain -> flooding", "flood~flooding -> road closures"]:
            assert c.post("/sessions/s/memory", json={"text": t}).status_code == 200
        r = c.post("/causal_chain", json={"from_content": "rain", "to_content": "flood"})
        assert r.status_code == 200 and r.json()["chain"], r.text
        r = c.post("/causal_chain", json={"from_content": "flood", "to_content": "rain"})
        assert r.status_code == 200 and not r.json()["chain"]


async def test_time_travel_after_soft_supersede():
    svc = MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=_zeus,
                        causal_extractor=False)
    svc.graph.conflict_resolver = _Resolver()
    async with svc:
        await svc.push("s", "k1~Zeus lives on Olympus.")
        await asyncio.sleep(0.01)
        mid = datetime.now()
        await asyncio.sleep(0.01)
        await svc.push("s", "k1~Zeus now lives in Athens.")   # resolver on by default here
        assert await svc.query("k1", top_k=10) == ["k1~Zeus now lives in Athens."]
        assert await svc.query("k1", top_k=10, as_of=mid) == ["k1~Zeus lives on Olympus."]
        hist = await svc.history("k1~Zeus lives on Olympus.")
        assert hist["superseded_by"] == ["k1~Zeus now lives in Athens."]


def test_mcp_lists_new_tools():
    pytest.importorskip("mcp")
    from reasongraph.service.mcp_server import create_mcp
    mcp = create_mcp(_service())
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"causal_chain_memory", "push_memory", "query_memory"} <= names


async def test_span_linking_lets_causal_chains_cross_wording():
    facts = ["rain~heavy rain -> f1~the river flooded the old town",
             "flood~f1~the old town flooded -> road closures"]      # f1~ = same embedding key
    def causal(texts):
        out = []
        for t in texts:
            if "->" in t:
                cause, effect = [p.strip() for p in t.split("->", 1)]
                cause = cause.split("~", 1)[1] if cause.count("~") == 2 else cause  # drop fact key only
                out.append({"causal": True, "relations": [{"cause": cause, "effect": effect}]})
            else:
                out.append({"causal": False, "relations": []})
        return out
    for threshold, expect in ((None, False), (0.85, True)):
        svc = MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=_zeus,
                            causal_extractor=causal)
        svc.graph.span_link_threshold = threshold
        async with svc:
            for f in facts:
                await svc.push("s", f)
            res = await svc.causal_chain("rain", "flood")
            assert bool(res["chain"]) is expect, (threshold, res)
