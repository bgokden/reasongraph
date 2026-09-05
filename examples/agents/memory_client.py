"""Tiny HTTP client for a reasongraph memory service (self-hosted or ReasonGraph Cloud).

Both expose the same routes, so the same agent code runs against
``reasongraph-serve`` on localhost and against ``https://memory.primaxiom.ai``.

    from memory_client import Memory
    mem = Memory()                          # reads MEMORY_URL / MEMORY_API_KEY
    mem.remember("research", "TSMC is building a fab in Arizona.")
    mem.recall("water and chips", session="research")
    mem.discover("Arizona", session="research")

Only dependency: httpx.
"""

from __future__ import annotations

import os

import httpx

DEFAULT_URL = "http://localhost:8000"


class Memory:
    def __init__(self, url: str | None = None, api_key: str | None = None, timeout: float = 60.0):
        self.url = (url or os.environ.get("MEMORY_URL") or DEFAULT_URL).rstrip("/")
        key = api_key or os.environ.get("MEMORY_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = httpx.Client(base_url=self.url, headers=headers, timeout=timeout)

    def _post(self, path: str, **body):
        r = self._http.post(path, json=body)
        r.raise_for_status()
        return r.json()

    # -- write --
    def remember(self, session: str, text: str) -> dict:
        return self._post(f"/sessions/{session}/memory", text=text)

    def remember_many(self, session: str, texts: list[str]) -> dict:
        return self._post(f"/sessions/{session}/memory/batch", texts=texts)

    def correct(self, session: str, old_text: str, new_text: str) -> dict:
        return self._post("/supersede", session=session, old_text=old_text, new_text=new_text)

    # -- read --
    def recall(self, query: str, session: str | None = None, top_k: int = 5, hops: int = 4,
               detailed: bool = False) -> list:
        return self._post("/query", query=query, session=session, top_k=top_k, hops=hops,
                          detailed=detailed)["facts"]

    def discover(self, query: str, session: str | None = None, top_k: int = 5, hops: int = 4,
                 max_results: int = 10) -> list[dict]:
        return self._post("/discover", query=query, session=session, top_k=top_k, hops=hops,
                          max_results=max_results)["connections"]

    def trace(self, content: str, direction: str = "effects", session: str | None = None) -> dict:
        return self._post("/trace", content=content, direction=direction, session=session)

    def sessions(self) -> list[str]:
        r = self._http.get("/sessions")
        r.raise_for_status()
        return r.json()["sessions"]

    def stats(self) -> dict:
        r = self._http.get("/stats")
        r.raise_for_status()
        return r.json()

    def wait_until_enriched(self, timeout: float = 120.0, interval: float = 1.0) -> None:
        """Block until the service has finished extracting entities/causal links for
        everything pushed so far (hosted services defer that work). Returns
        immediately when the service extracts synchronously."""
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.stats().get("pending", 0) == 0:
                return
            time.sleep(interval)
        raise TimeoutError("memory service still extracting after %.0fs" % timeout)


def format_connections(connections: list[dict]) -> str:
    """Render discover() output as compact evidence for an LLM prompt."""
    lines = []
    for c in connections:
        tag = " (from another session)" if c.get("cross_session") else ""
        path = " -> ".join(
            step["entity"] if "entity" in step else f"[{step['content'][:40]}...]"
            for step in c.get("path", [])
        )
        lines.append(f"- {c['content']}{tag}\n    via: {path}")
        for rel in c.get("causes", []):
            lines.append(f"    cause: {rel['cause']} -> {rel['effect']}")
    return "\n".join(lines) if lines else "(nothing connected yet)"
