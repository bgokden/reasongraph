"""Agent-memory + discovery service over a shared ReasonGraph.

Knowledge sessions are scopes: agents push memory to a session and query it,
while traversal crosses sessions so agents discover connections beyond their own.
The core (``MemoryService``) is transport-agnostic; ``http`` and ``mcp`` expose
it over FastAPI and MCP respectively.
"""

from reasongraph.service.core import MemoryService

__all__ = ["MemoryService"]
