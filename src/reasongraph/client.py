"""A small HTTP client for a hosted ReasonGraph service (ReasonGraph Cloud or your own
``reasongraph serve``). Mirrors the routes the agents use; every method returns the JSON body.

    from reasongraph.client import MemoryClient
    mem = MemoryClient("https://memory.primaxiom.ai", api_key="rgm_...")
    mem.remember("scout", "TSMC is building a chip fab in Phoenix, Arizona.")
    mem.discover("Why might Apple face shortages?")
"""
from __future__ import annotations

import os
from typing import Any


class MemoryClient:
    def __init__(self, url: str | None = None, api_key: str | None = None, timeout: float = 60.0) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError("MemoryClient needs httpx: pip install httpx") from exc
        self.url = (url or os.environ.get("MEMORY_URL") or "https://memory.primaxiom.ai").rstrip("/")
        key = api_key or os.environ.get("MEMORY_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = httpx.Client(base_url=self.url, headers=headers, timeout=timeout)

    def _post(self, path: str, **body: Any) -> dict:
        r = self._http.post(path, json={k: v for k, v in body.items() if v is not None})
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = self._http.get(path)
        r.raise_for_status()
        return r.json()

    # writes
    def remember(self, session: str, text: str, *, split: bool | None = None) -> dict:
        return self._post(f"/sessions/{session}/memory", text=text, split=split)

    def remember_many(self, session: str, texts: list[str], *, split: bool | None = None) -> dict:
        return self._post(f"/sessions/{session}/memory/batch", texts=texts, split=split)

    def correct(self, session: str, old_text: str, new_text: str) -> dict:
        return self._post("/supersede", session=session, old_text=old_text, new_text=new_text)

    def forget_session(self, session: str) -> dict:
        r = self._http.delete(f"/sessions/{session}")
        r.raise_for_status()
        return r.json()

    # reads
    def recall(self, query: str, *, session: str | None = None, top_k: int = 5, hops: int = 4) -> list:
        return self._post("/query", query=query, session=session, top_k=top_k, hops=hops).get("facts", [])

    def discover(self, query: str, *, session: str | None = None, top_k: int = 5, hops: int = 4,
                 max_results: int = 10) -> list[dict]:
        res = self._post("/discover", query=query, session=session, top_k=top_k, hops=hops, max_results=max_results)
        return res.get("connections", res) if isinstance(res, dict) else res

    def chat(self, messages: list[dict], *, session: str = "chat", system: str | None = None,
             observe: bool = True, max_facts: int = 8) -> dict:
        return self._post("/chat", messages=messages, session=session, system=system, observe=observe, max_facts=max_facts)

    def facts(self, texts: list[str]) -> dict:
        return self._post("/facts", texts=texts)

    def stats(self) -> dict:
        return self._get("/stats")
