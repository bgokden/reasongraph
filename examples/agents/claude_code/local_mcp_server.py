"""Local stdio MCP memory server for Claude Code / Claude Desktop / Cursor.

    claude mcp add memory -- python local_mcp_server.py

Same as the installed ``reasongraph-mcp`` command (``claude mcp add memory -- reasongraph-mcp``).
Persists to ~/.reasongraph/memory.sqlite (override with REASONGRAPH_DATABASE_URL).
Deps: pip install "reasongraph[service,gliner,fastembed,sqlite]"
"""

from reasongraph.service.mcp_stdio import main

if __name__ == "__main__":
    main()
