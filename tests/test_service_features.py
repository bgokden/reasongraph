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
    # f1~ = same embedding key, but no shared content word: only a same_as span link
    # (not the lexical bridge used by causal_chain) can connect the two wordings.
    facts = ["rain~heavy rain -> f1~the river overflowed its banks",
             "flood~f1~the old town was inundated -> road closures"]
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


async def test_conflict_resolution_never_reaches_other_scopes():
    """A push tagged with tenant A's scopes must not retire tenant B's fact even when
    the resolver would call it a contradiction and it is the nearest neighbour."""
    class AlwaysYes:
        def contradictions(self, new_text, candidates):
            return list(candidates)
    svc = MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=_zeus,
                        causal_extractor=False)
    svc.graph.conflict_resolver = AlwaysYes()
    async with svc:
        await svc.graph.add_texts(["k1~Zeus lives on Olympus."], extractor=_zeus, scopes=["b/s", "b"])
        await svc.graph.add_texts(["k1~Zeus lives in Athens."], extractor=_zeus, scopes=["a/s", "a"])
        await svc.graph.add_texts(["k1~Zeus now lives in Sparta."], extractor=_zeus, scopes=["a/s", "a"],
                                  resolve_conflicts=True)
        hist_b = await svc.graph.supersession_history("k1~Zeus lives on Olympus.")
        hist_a = await svc.graph.supersession_history("k1~Zeus lives in Athens.")
        assert hist_b["superseded_by"] == []                          # other tenant untouched
        assert hist_a["superseded_by"] == ["k1~Zeus now lives in Sparta."]


async def test_pending_counts_in_flight_work():
    """stats.pending must stay > 0 while the worker is still processing a fact,
    not only while it sits in the queue (clients poll it to know when bridges and
    contradiction checks are done)."""
    import asyncio as _asyncio
    started = _asyncio.Event(); release = _asyncio.Event()

    def slow_extractor(text):
        started.set()
        # block the worker thread until the test says go
        import time
        while not release.is_set():
            time.sleep(0.01)
        return _zeus(text)

    svc = MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=slow_extractor,
                        causal_extractor=False, defer_extraction=True)
    async with svc:
        await svc.push("s", "k1~Zeus threw lightning.")
        await _asyncio.wait_for(started.wait(), 5)
        await _asyncio.sleep(0.05)                    # item is dequeued and in flight
        assert svc.pending_extractions == 1
        release.set()
        await svc._enrich_queue.join()
        assert svc.pending_extractions == 0


async def test_knn_search_returns_cosine_scores_memory_backend():
    svc = _service()
    async with svc:
        await svc.push("s", "k1~Zeus threw lightning.")
        await svc.push("s", "k2~Zeus slept.")
        hits = await svc.graph.backend.knn_search(_Embed()("k1"), top_k=2)
        assert hits[0]["content"] == "k1~Zeus threw lightning."
        assert hits[0]["score"] > 0.99 and hits[1]["score"] < 0.5


def test_finetuned_conflict_resolver_prompt_grammar_and_fail_open():
    from reasongraph._conflict import FineTunedConflictResolver
    calls = []
    def fake_post(url, body):
        calls.append((url, body))
        # the served model answers with the JSON the grammar admits
        existing = body["prompt"].split("existing: ", 1)[1].split("\nnew:", 1)[0]
        return {"content": '{"conflict": true}' if "Olympus" in existing else '{"conflict": false}'}
    r = FineTunedConflictResolver("http://llm:8080/", post=fake_post)
    hits = r.contradictions("Zeus now lives in Athens.", ["Zeus lives on Olympus.", "Hera is Zeus's wife.", "Zeus now lives in Athens."])
    assert hits == ["Zeus lives on Olympus."]
    assert calls[0][0] == "http://llm:8080/completion"
    assert calls[0][1]["prompt"] == "[conflict] existing: Zeus lives on Olympus.\nnew: Zeus now lives in Athens."
    assert calls[0][1]["temperature"] == 0 and "conflict" in calls[0][1]["grammar"]
    assert len(calls) == 2            # identical text skipped, no call for it
    # a dead model never blocks a push
    def dead(url, body): raise OSError("connection refused")
    r2 = FineTunedConflictResolver("http://llm:8080", post=dead)
    assert r2.contradictions("x", ["y"]) == []
    assert FineTunedConflictResolver("http://llm:8080", post=dead, fail_open=False).is_conflict.__name__ == "is_conflict"


async def test_push_split_stores_one_fact_per_sentence():
    from reasongraph._split import RegexSplitter
    svc = MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=_zeus, causal_extractor=False)
    svc.graph.sentence_splitter = RegexSplitter()
    async with svc:
        out = await svc.push("s", "Zeus lives on Olympus. Hera is his wife! Athena was born from his head.", split=True)
        assert out["count"] == 3 and len(out["sentences"]) == 3
        assert out["sentences"][1] == "Hera is his wife!"
        # default is off: the same text as one fact
        out2 = await svc.push("t", "One fact. Two facts.")
        assert "sentences" not in out2
    # service-wide default on
    svc2 = MemoryService(backend=MemoryBackend(), embed_model=_Embed(), extractor=_zeus, causal_extractor=False,
                         split_sentences=True)
    async with svc2:
        out3 = await svc2.push("s", "One fact. Two facts.")
        assert out3["count"] == 2
        out4 = await svc2.push("s", "Three facts. Four facts.", split=False)
        assert "sentences" not in out4
