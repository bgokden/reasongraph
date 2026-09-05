"""Two agents, one memory, separate sessions: discovery without message passing.

A *scout* agent records what it reads under its own session. An *analyst* agent,
with its own session, is asked a question it has no facts for. Its `discover`
call seeds from its session and reaches the scout's facts through shared
entities (Arizona, TSMC, Apple) -- the service flags them `cross_session`.

Against ReasonGraph Cloud both sessions live inside your tenant; another tenant's
memory is never reachable.

Run:  python two_agents_shared_memory.py        (no LLM needed)
"""

from __future__ import annotations

from memory_client import Memory, format_connections

SCOUT = "scout"
ANALYST = "analyst"


def main() -> None:
    mem = Memory()
    print(f"memory: {mem.url}")

    # The scout records field notes -- never mentions chips or Apple in the same sentence as water.
    mem.remember_many(SCOUT, [
        "Arizona declared a water emergency during a record-breaking drought.",
        "Maricopa County ordered mandatory water cuts for industrial users.",
        "TSMC is building a chip fabrication plant in Phoenix, Arizona.",
    ])
    # The analyst only knows about Apple.
    mem.remember_many(ANALYST, [
        "Apple depends on TSMC for its M-series processors.",
        "Apple warned investors about component shortages from North American suppliers.",
    ])

    q = "Why might Apple face component shortages?"
    print(f"\nanalyst asks: {q}")
    conns = mem.discover(q, session=ANALYST, top_k=5, max_results=8)
    print(format_connections(conns))

    reached = [c for c in conns if c.get("cross_session")]
    print(f"\n{len(reached)} of {len(conns)} connections came from the scout's session, "
          f"via entities: {sorted({s['entity'] for c in reached for s in c['path'] if 'entity' in s})}")


if __name__ == "__main__":
    main()
