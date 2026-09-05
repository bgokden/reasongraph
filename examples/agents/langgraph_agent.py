"""The research agent as a LangGraph ReAct graph with reasongraph memory tools.

Run:
    pip install langgraph langchain-openai httpx
    MEMORY_URL=http://localhost:8000 LLM_API_KEY=gsk_... python langgraph_agent.py
"""

from __future__ import annotations

import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from memory_client import Memory, format_connections
from research_agent import SOURCES

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
SESSION = os.environ.get("MEMORY_SESSION", "research-langgraph")

mem = Memory()


@tool
def remember(text: str) -> str:
    """Store a fact you learned so you can use it later."""
    mem.remember(SESSION, text)
    return "stored"


@tool
def recall(query: str) -> str:
    """Retrieve facts relevant to a query from memory."""
    return "\n".join(f"- {f}" for f in mem.recall(query, session=SESSION, top_k=8)) or "(nothing)"


@tool
def discover(query: str) -> str:
    """Find how facts connect to a topic across everything remembered (paths through
    shared entities and cause->effect links). Use for 'how does X affect Y'."""
    return format_connections(mem.discover(query, session=SESSION, top_k=5, max_results=8))


def main() -> None:
    for facts in SOURCES.values():
        mem.remember_many(SESSION, facts)
    mem.wait_until_enriched()
    llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY, temperature=0)
    agent = create_react_agent(
        llm, [remember, recall, discover],
        prompt="You are a research analyst with a persistent graph memory. Call discover and/or "
               "recall before answering, and cite the facts you used.",
    )
    q = "How does the Arizona water crisis affect Apple's chip supply?"
    out = agent.invoke({"messages": [("user", q)]})
    print("Q:", q)
    print("A:", out["messages"][-1].content)


if __name__ == "__main__":
    main()
