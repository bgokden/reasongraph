"""MCP transport for the memory service -- exposes memory as agent tools.

    from reasongraph.service import MemoryService
    from reasongraph.service.mcp_server import create_mcp

    service = MemoryService(backend=..., synthesizer=my_llm)
    await service.initialize()
    create_mcp(service).run()          # stdio MCP server

Agents get push_memory / query_memory / discover_connections / answer /
list_sessions tools. Requires ``pip install reasongraph[service]``.
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
        query: str, session: str | None = None, hops: int = 4, top_k: int = 5
    ) -> list[str]:
        """Retrieve ranked facts. Seeds from `session`; reasoning crosses sessions."""
        return await service.query(query, session=session, hops=hops, top_k=top_k)

    @mcp.tool()
    async def discover_connections(
        query: str, session: str | None = None, hops: int = 4, top_k: int = 5
    ) -> list[dict]:
        """Discover how facts connect across sessions: returns connection paths,
        each flagging whether it was found in another agent's session."""
        return await service.discover(query, session=session, hops=hops, top_k=top_k)

    @mcp.tool()
    async def answer(query: str, session: str | None = None, hops: int = 4) -> str:
        """Answer in logical free text synthesized from discovered facts
        (requires a synthesizer configured on the service)."""
        return await service.answer(query, session=session, hops=hops)

    @mcp.tool()
    async def list_sessions() -> list[str]:
        """List all knowledge sessions currently in memory."""
        return await service.list_sessions()

    return mcp
