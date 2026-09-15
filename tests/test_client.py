"""The hosted-service client sends what the endpoints accept, and nothing it was not given."""
import json

import pytest

httpx = pytest.importorskip("httpx")

from reasongraph.client import MemoryClient


def _client(seen):
    def handler(request):
        seen.append((request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(200, json={"reply": "ok", "context": [], "stored": []})
    c = MemoryClient(url="https://example.test", api_key="rgm_x")
    c._http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://example.test")
    return c


def test_chat_passes_the_folding_budget_only_when_given():
    seen = []
    c = _client(seen)
    c.chat([{"role": "user", "content": "hi"}], session="s")
    assert seen[-1][0] == "/chat"
    assert "max_history_tokens" not in seen[-1][1] and "keep_tail_tokens" not in seen[-1][1]

    c.chat([{"role": "user", "content": "hi"}], session="s", max_history_tokens=2000, keep_tail_tokens=800)
    assert seen[-1][1]["max_history_tokens"] == 2000 and seen[-1][1]["keep_tail_tokens"] == 800
