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
    svc = MemoryService(backend=MemoryBackend(), embed_model=_fake_embed, extractor=_zeus_extractor)
    with TestClient(create_app(svc)) as client:
        client.post("/sessions/a/memory", json={"text": "Zeus is a god."})
        r = client.post("/query", json={"query": "Zeus", "session": "a", "synthesize": True})
        assert r.status_code == 400
