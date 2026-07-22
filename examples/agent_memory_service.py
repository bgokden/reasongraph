"""Multi-agent memory + cross-session discovery with MemoryService.

Five agents each push what they learned into their own knowledge session --
markets, supply chain, energy, health, policy. No agent sees another's data.
Discovery, scoped to one agent's session, walks the SHARED entity graph and
surfaces connections that live in *other* sessions: a market query about Nvidia
reaches a Taiwan drought a supply-chain bot recorded; an energy query about
Arizona reaches a chip fab a different bot is tracking.

The entities that bridge sessions (TSMC, Nvidia, Apple, Arizona, Taiwan, ASML)
are extracted automatically from the raw text -- nothing is wired by hand.

This is the in-process core; `reasongraph.service.http` / `.mcp_server` expose
the same MemoryService over HTTP and MCP.

Run:  uv run python examples/agent_memory_service.py
Deps: pip install reasongraph[service,gliner,fastembed]
"""

import asyncio
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # models on CPU for the demo

from reasongraph.service import MemoryService
from reasongraph import FastEmbedEmbedder, TemplateSynthesizer


# Each agent's private knowledge session. The facts interlock only through the
# named entities they share, which is what makes cross-session discovery work.
WORLD = {
    "markets-bot": [
        "Nvidia's market capitalization crossed three trillion dollars on surging AI chip demand.",
        "Apple shares climbed after strong iPhone sales in China.",
        "TSMC reported record quarterly revenue driven by orders for AI accelerators.",
        "Investors rotated into semiconductor stocks as the AI boom accelerated.",
    ],
    "supply-bot": [
        "TSMC manufactures the advanced chips that Nvidia designs.",
        "Apple sources its custom A-series processors from TSMC in Taiwan.",
        "ASML is the only company that supplies EUV lithography machines to TSMC.",
        "TSMC is building a new chip fabrication plant in Arizona.",
        "A severe drought in Taiwan disrupted the water supply TSMC needs for fabrication.",
    ],
    "energy-bot": [
        "Arizona declared a water emergency during a record-breaking drought.",
        "Taiwan expanded desalination capacity to protect its semiconductor industry.",
        "AI data centers pushed electricity demand in Arizona to new highs.",
        "Nevada lithium mining scaled up to feed battery production.",
    ],
    "health-bot": [
        "Prolonged drought in Arizona worsened dust storms and respiratory illness.",
        "Groundwater near semiconductor plants raised local contamination concerns.",
    ],
    "policy-bot": [
        "The US CHIPS Act subsidized companies like TSMC to build domestic fabs.",
        "New export controls restricted Nvidia's most advanced AI chips from China.",
    ],
}


def _print_connections(connections):
    for c in connections:
        tag = "  <cross-session>" if c["cross_session"] else ""
        scopes = ",".join(c["scopes"])
        print(f"  [{scopes}] {c['content']}{tag}")
        path = " -> ".join(
            step["entity"] if "entity" in step else f"[{step['content'][:32]}...]"
            for step in c["path"]
        )
        print(f"      path: {path}")
        for rel in c.get("causes", []):
            print(f"      causes: {rel['cause']} -> {rel['effect']}")


async def main():
    service = MemoryService(
        embed_model=FastEmbedEmbedder(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        # Deterministic, model-free rephrasing. For an LLM-written answer, swap in
        # PromptSynthesizer(your_generate_fn) or TransformersSynthesizer().
        synthesizer=TemplateSynthesizer(),
        # extractor defaults to gliner_small-v2.5 (fast, multilingual)
    )

    async with service:
        # Every agent records its own facts into its own session.
        for session, facts in WORLD.items():
            for fact in facts:
                await service.push(session, fact)

        print("Sessions:", await service.list_sessions())
        print("Stats:", await service.stats())

        # A market bot asks about Nvidia. Its seeds come only from its own
        # session, but discovery crosses -- via the shared Nvidia and TSMC
        # entities -- into the supply, energy, and policy sessions.
        print("\n--- markets-bot discovers around 'Nvidia' ---")
        _print_connections(
            await service.discover("Nvidia AI chips", session="markets-bot", hops=5)
        )

        # An energy bot asks about Arizona water. Discovery reaches the chip fab
        # a supply bot is tracking and the health impact a health bot recorded.
        print("\n--- energy-bot discovers around 'Arizona water' ---")
        _print_connections(
            await service.discover("Arizona water shortage", session="energy-bot", hops=5)
        )

        # Supply bot asks about Taiwan; discovery links manufacturing to the
        # drought risk that markets and energy bots see from other angles.
        print("\n--- supply-bot discovers around 'Taiwan' ---")
        _print_connections(
            await service.discover("Taiwan chip manufacturing", session="supply-bot", hops=5)
        )

        print("\n--- answer (synthesized, markets-bot on 'Nvidia') ---")
        print(await service.answer("Nvidia AI chips", session="markets-bot", hops=5))


if __name__ == "__main__":
    asyncio.run(main())
