"""Multi-agent memory + cross-session discovery with MemoryService.

Two agents push memories into their own knowledge sessions. Neither sees the
other's data. A third query -- scoped to one agent's session -- discovers a
connection into the *other* agent's session through a shared entity, returns the
connection path, and synthesizes a logical free-text answer.

This is the in-process core; `reasongraph.service.http` / `.mcp_server` expose
the same MemoryService over HTTP and MCP.

Run:  uv run python examples/agent_memory_service.py
Deps: pip install reasongraph[service,gliner,fastembed]
"""

import asyncio
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # models on CPU for the demo

from reasongraph.service import MemoryService
from reasongraph import FastEmbedEmbedder


def logical_synthesizer(query, context):
    """A tiny template synthesizer (swap in a real small LLM here).

    Turns the retrieved facts + cross-session flags into logical free text.
    """
    if not context:
        return f"I have nothing on '{query}' yet."
    lines = [f"On '{query}', the memory graph connects these facts:"]
    for item in context:
        origin = " [discovered in another session]" if item.get("cross_session") else ""
        lines.append(f"  - {item['content']}{origin}")
    bridged = [i for i in context if i.get("cross_session")]
    if bridged:
        vias = {step["entity"] for i in bridged for step in i["path"] if "entity" in step}
        lines.append(f"The link across sessions runs through: {', '.join(sorted(vias))}.")
    return "\n".join(lines)


async def main():
    service = MemoryService(
        embed_model=FastEmbedEmbedder(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        synthesizer=logical_synthesizer,
        # extractor defaults to gliner_small-v2.5 (fast, multilingual)
    )

    async with service:
        # Agent A (a research bot) records what it learned.
        await service.push("research-bot", "TSMC is building a chip fabrication plant in Arizona.")
        await service.push("research-bot", "The Arizona plant will supply chips to Apple.")

        # Agent B (a news bot) records unrelated news, in its own session.
        await service.push("news-bot", "Arizona declared a water emergency amid record drought.")

        print("Sessions:", await service.list_sessions())
        print("Stats:", await service.stats())

        # The research bot asks about Arizona -- seeds from its own session, but
        # discovery crosses into the news bot's session via the shared entity.
        print("\n--- discover (session=research-bot) ---")
        connections = await service.discover("Arizona", session="research-bot", hops=4)
        for c in connections:
            tag = "  <cross-session>" if c["cross_session"] else ""
            print(f"  {c['content']}{tag}")
            path = " -> ".join(
                step["entity"] if "entity" in step else f"[{step['content'][:28]}...]"
                for step in c["path"]
            )
            print(f"      path: {path}")

        print("\n--- answer (synthesized) ---")
        print(await service.answer("Arizona", session="research-bot"))


if __name__ == "__main__":
    asyncio.run(main())
