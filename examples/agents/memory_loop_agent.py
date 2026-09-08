"""Deep memory integration without tools: a chat agent whose every turn is remembered
and whose relevant memories are injected automatically before each model call.

Works with any OpenAI-compatible endpoint (Groq, Ollama, llama.cpp, vLLM). Run:

    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:4b python memory_loop_agent.py

Local memory by default; point MEMORY_URL + MEMORY_API_KEY at ReasonGraph Cloud to share
the memory with your other agents (the loop only needs a graph-like object).
"""

from __future__ import annotations

import os
import sys

import httpx

from reasongraph import MemoryLoop, ReasonGraph

BASE = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
KEY = os.environ.get("LLM_API_KEY", "ollama")
MODEL = os.environ.get("LLM_MODEL", "qwen3:4b")
SYSTEM = ("You are a careful assistant. Use the remembered facts when they apply, say which "
          "ones you used, and say plainly when you do not know.")


def call_model(messages: list[dict]) -> str:
    r = httpx.post(f"{BASE.rstrip('/')}/chat/completions", timeout=120,
                   headers={"Authorization": f"Bearer {KEY}"},
                   json={"model": MODEL, "messages": messages, "temperature": 0.3})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main() -> None:
    graph = ReasonGraph()                       # local, in-memory; swap for a Postgres backend to persist
    graph.initialize_sync()
    loop = MemoryLoop(graph, session="chat", max_facts=8)
    history: list[dict] = []
    print("Memory-backed chat. Type facts or questions; /memory shows what was recalled last; /quit exits.")
    last_block = None
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/memory":
            print(last_block.text if last_block and last_block.text else "(nothing recalled yet)")
            continue
        history.append({"role": "user", "content": text})
        reply, last_block = loop.chat_sync(call_model, history, system=SYSTEM)
        history.append({"role": "assistant", "content": reply})
        print(f"\n{reply}\n")
        if last_block.facts:
            print(f"  (used {len(last_block.facts)} remembered fact(s); /memory to see them)\n")
    graph.close_sync()


if __name__ == "__main__":
    main()
