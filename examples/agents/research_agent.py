"""A research agent with graph memory: read sources, remember, answer with evidence.

The agent gets three tools -- remember, recall, discover -- backed by a reasongraph
memory service. It first "reads" a handful of sources (each pushed into memory as
facts), then answers questions. The interesting part is `discover`: the answer to
"how does the water crisis affect chip supply" is assembled from facts that were
written by different sources and never mention each other.

Run:
    MEMORY_URL=http://localhost:8000 LLM_API_KEY=gsk_... python research_agent.py
Deps:
    pip install httpx openai
"""

from __future__ import annotations

import json
import os
import sys

from openai import OpenAI

from memory_client import Memory, format_connections

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
SESSION = os.environ.get("MEMORY_SESSION", "research")

SOURCES = {
    "tech-report": [
        "TSMC announced plans to build a $40 billion semiconductor fabrication plant in Phoenix, Arizona.",
        "The Phoenix fab requires 10 million gallons of purified water daily to cool wafers during chip etching.",
        "TSMC signed a long-term supply agreement with Apple to manufacture M-series processors at the Arizona facility.",
    ],
    "environment-report": [
        "Arizona declared a water emergency after Lake Mead dropped to its lowest level since the 1930s.",
        "The Arizona Department of Water Resources ordered mandatory water cuts for all industrial users in Maricopa County, where Phoenix is located.",
        "Intel paused expansion of its Chandler, Arizona chip plant citing water availability concerns.",
    ],
    "markets-brief": [
        "Apple warned investors that component shortages from North American suppliers could affect iPhone production through 2026.",
    ],
}

TOOLS = [
    {"type": "function", "function": {
        "name": "remember",
        "description": "Store a fact you learned so you can use it later.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Retrieve facts relevant to a query from memory.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "discover",
        "description": "Find HOW facts connect to a topic across everything remembered: returns connection paths through shared entities and cause->effect links. Use this for 'how does X affect Y' questions.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

SYSTEM = (
    "You are a research analyst with a persistent graph memory. Before answering, call "
    "`discover` (for relationships) and/or `recall` (for facts). Cite the facts you used. "
    "If memory has nothing relevant, say so instead of guessing."
)


def run_tool(mem: Memory, name: str, args: dict) -> str:
    if name == "remember":
        mem.remember(SESSION, args["text"])
        return "stored"
    if name == "recall":
        return "\n".join(f"- {f}" for f in mem.recall(args["query"], session=SESSION, top_k=8)) or "(nothing)"
    if name == "discover":
        return format_connections(mem.discover(args["query"], session=SESSION, top_k=5, max_results=8))
    return f"unknown tool {name}"


def agent(llm: OpenAI, mem: Memory, question: str, max_steps: int = 6) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(max_steps):
        resp = llm.chat.completions.create(model=LLM_MODEL, messages=messages, tools=TOOLS, tool_choice="auto")
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content or ""
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = run_tool(mem, call.function.name, args)
            print(f"  [tool] {call.function.name}({args.get('query') or args.get('text', '')[:50]!r}) -> {len(result)} chars")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "(stopped: too many steps)"


def main() -> None:
    if not LLM_API_KEY:
        sys.exit("set LLM_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY)")
    mem = Memory()
    llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    print(f"memory: {mem.url}  session: {SESSION}")
    print("reading sources into memory...")
    for name, facts in SOURCES.items():
        mem.remember_many(SESSION, facts)
        print(f"  {name}: {len(facts)} facts")
    mem.wait_until_enriched()   # let the service finish entity/causal extraction

    for q in [
        "How does the Arizona water crisis affect Apple's chip supply?",
        "What is the root cause chain behind Apple's warning to investors?",
    ]:
        print(f"\nQ: {q}")
        print("A:", agent(llm, mem, q))


if __name__ == "__main__":
    main()
