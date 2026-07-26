"""MCP transport for the memory service -- exposes memory as agent tools.

    from reasongraph.service import MemoryService
    from reasongraph.service.mcp_server import create_mcp

    service = MemoryService(backend=..., synthesizer=my_llm)
    await service.initialize()
    create_mcp(service).run()          # stdio MCP server

Agents get push_memory / query_memory / query_memory_detailed /
discover_connections / answer / update_memory / delete_memory / memory_history /
forget_stale / list_sessions tools -- including self-correction (update/delete) and
supersession audit (memory_history) so an agent can fix and explain its own memory.
Requires ``pip install reasongraph[service]``.
"""

from __future__ import annotations

from reasongraph.service.core import MemoryService


def create_mcp(service: MemoryService):
    """Build a FastMCP server exposing the memory service as agent tools.

    The service must be initialized before the server is run.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("reasongraph-memory")

    @mcp.tool()
    async def push_memory(session: str, text: str) -> dict:
        """Store a memory in a knowledge session; returns the extracted entities."""
        return await service.push(session, text)

    @mcp.tool()
    async def query_memory(
        query: str, session: str | None = None, hops: int = 4, top_k: int = 5,
        recency_weight: float = 0.0, isolate: bool = False,
    ) -> list[str]:
        """Retrieve ranked facts. Seeds from `session`; reasoning crosses sessions
        unless `isolate` is true (then it stays within `session`). `recency_weight`
        (0-1) favours newer facts over older contradicting ones."""
        return await service.query(
            query, session=session, hops=hops, top_k=top_k,
            recency_weight=recency_weight, isolate=isolate,
        )

    @mcp.tool()
    async def query_memory_detailed(
        query: str, session: str | None = None, hops: int = 4, top_k: int = 5,
        recency_weight: float = 0.0, isolate: bool = False,
    ) -> list[dict]:
        """Like query_memory but each fact comes with a relevance score, its
        created_at timestamp, and its scopes -- so you can threshold on confidence,
        dedupe, or say when something was remembered."""
        return await service.query(
            query, session=session, hops=hops, top_k=top_k,
            recency_weight=recency_weight, isolate=isolate, detailed=True,
        )

    @mcp.tool()
    async def discover_connections(
        query: str, session: str | None = None, hops: int = 4, top_k: int = 5,
        isolate: bool = False,
    ) -> list[dict]:
        """Discover how facts connect across sessions: returns connection paths,
        each flagging whether it was found in another agent's session. Set
        `isolate` true to confine the walk to `session`."""
        return await service.discover(
            query, session=session, hops=hops, top_k=top_k, isolate=isolate,
        )

    @mcp.tool()
    async def answer(query: str, session: str | None = None, hops: int = 4) -> str:
        """Answer in logical free text synthesized from discovered facts
        (requires a synthesizer configured on the service)."""
        return await service.answer(query, session=session, hops=hops)

    @mcp.tool()
    async def update_memory(
        session: str, old_text: str, new_text: str, purge_orphans: bool = False,
    ) -> dict:
        """Correct memory: add `new_text` and remove the stale `old_text`. Set
        `purge_orphans` true to also erase entities left dangling by the removal
        (right-to-be-forgotten); shared entities are kept."""
        return await service.supersede(
            session, old_text, new_text, purge_orphans=purge_orphans,
        )

    @mcp.tool()
    async def delete_memory(text: str, purge_orphans: bool = False) -> dict:
        """Delete a fact by its exact text. Set `purge_orphans` true to also erase
        entities left dangling by the removal; entities still used elsewhere stay."""
        return await service.delete(text, purge_orphans=purge_orphans)

    @mcp.tool()
    async def memory_history(text: str) -> dict:
        """Audit a fact's supersession: what it replaced ('supersedes') and what has
        replaced it ('superseded_by'). A non-empty 'superseded_by' is why a fact is
        no longer surfaced by default."""
        return await service.history(text)

    @mcp.tool()
    async def forget_stale() -> dict:
        """Drop facts not accessed within the service's forget window."""
        return await service.forget()

    @mcp.tool()
    async def list_sessions() -> list[str]:
        """List all knowledge sessions currently in memory."""
        return await service.list_sessions()

    return mcp
