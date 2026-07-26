import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from reasongraph.service import MemoryService
from reasongraph.service.http import create_app
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


def _client():
    svc = MemoryService(
        backend=MemoryBackend(), embed_model=_fake_embed,
        extractor=_zeus_extractor, synthesizer=_fake_synth,
        causal_extractor=False,  # no real causal model in unit tests
    )
    return TestClient(create_app(svc))


def test_http_multi_agent_discovery_and_synthesis():
    with _client() as client:
        # two agents push to separate sessions
        assert client.post("/sessions/agent-1/memory", json={"text": "Zeus threw lightning bolts."}).status_code == 200
        assert client.post("/sessions/agent-2/memory", json={"text": "Zeus lived on Mount Olympus."}).status_code == 200

        # agent-1 discovers agent-2's fact via the shared entity, with a synthesized answer
        r = client.post("/discover", json={"query": "Zeus", "session": "agent-1", "synthesize": True})
        assert r.status_code == 200
        data = r.json()
        contents = {c["content"] for c in data["connections"]}
        assert "Zeus lived on Mount Olympus." in contents
        cross = [c for c in data["connections"] if c["cross_session"]]
        assert any(c["content"] == "Zeus lived on Mount Olympus." for c in cross)
        assert "Zeus" in data["answer"]

        # sessions + stats
        assert set(client.get("/sessions").json()["sessions"]) == {"agent-1", "agent-2"}
        assert client.get("/stats").json()["facts"] == 2


def test_http_synthesize_without_synthesizer_returns_400():
    svc = MemoryService(backend=MemoryBackend(), embed_model=_fake_embed, extractor=_zeus_extractor, causal_extractor=False)
    with TestClient(create_app(svc)) as client:
        client.post("/sessions/a/memory", json={"text": "Zeus is a god."})
        r = client.post("/query", json={"query": "Zeus", "session": "a", "synthesize": True})
        assert r.status_code == 400


def test_http_push_many_and_stats():
    with _client() as client:
        r = client.post("/sessions/a/memory/batch", json={"texts": [
            "Zeus threw lightning bolts.", "Zeus lived on Mount Olympus.",
        ]})
        assert r.status_code == 200 and r.json()["count"] == 2
        assert client.get("/stats").json()["facts"] == 2


def test_http_query_with_synthesize():
    with _client() as client:
        client.post("/sessions/a/memory", json={"text": "Zeus rules the sky."})
        r = client.post("/query", json={"query": "Zeus", "session": "a", "synthesize": True})
        assert r.status_code == 200
        data = r.json()
        assert "Zeus rules the sky." in data["facts"]
        assert "Zeus" in data["answer"]


def test_http_supersede_replaces_fact():
    with _client() as client:
        client.post("/sessions/a/memory", json={"text": "Zeus is mortal."})
        r = client.post("/supersede", json={
            "session": "a", "old_text": "Zeus is mortal.", "new_text": "Zeus is immortal.",
        })
        assert r.status_code == 200 and r.json()["superseded"] is True
        facts = client.post("/query", json={"query": "Zeus", "session": "a"}).json()["facts"]
        assert "Zeus is immortal." in facts and "Zeus is mortal." not in facts


def test_http_forget_endpoint():
    with _client() as client:
        client.post("/sessions/a/memory", json={"text": "Zeus is a god."})
        r = client.post("/forget")
        assert r.status_code == 200 and "deleted" in r.json()


def test_http_health_and_ready():
    with _client() as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"ready": True}


def test_http_api_key_auth():
    svc = MemoryService(backend=MemoryBackend(), embed_model=_fake_embed,
                        extractor=_zeus_extractor, causal_extractor=False)
    with TestClient(create_app(svc, api_key="secret")) as client:
        # probes stay open for load balancers
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        # data endpoints reject a missing/wrong key
        assert client.post("/sessions/a/memory", json={"text": "Zeus is a god."}).status_code == 401
        assert client.get("/stats", headers={"X-API-Key": "nope"}).status_code == 401
        # both Bearer and X-API-Key are accepted
        assert client.post("/sessions/a/memory", json={"text": "Zeus is a god."},
                           headers={"Authorization": "Bearer secret"}).status_code == 200
        assert client.get("/stats", headers={"X-API-Key": "secret"}).status_code == 200


def test_http_delete_endpoint():
    with _client() as client:
        client.post("/sessions/a/memory", json={"text": "Zeus is a god."})
        r = client.post("/delete", json={"text": "Zeus is a god.", "purge_orphans": True})
        assert r.status_code == 200 and r.json()["deleted"] is True
        assert "Zeus is a god." not in client.post("/query", json={"query": "Zeus"}).json()["facts"]


def test_http_query_detailed():
    with _client() as client:
        client.post("/sessions/a/memory", json={"text": "Zeus rules the sky."})
        facts = client.post(
            "/query", json={"query": "Zeus", "session": "a", "detailed": True}
        ).json()["facts"]
        assert facts and isinstance(facts[0], dict)
        assert set(facts[0]) == {"content", "score", "created_at", "scopes"}
