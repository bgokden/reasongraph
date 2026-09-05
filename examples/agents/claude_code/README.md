# Claude Code with persistent memory

Give Claude Code (or Claude Desktop, Cursor, any MCP client) a memory that survives
across sessions and projects: it can `push_memory` what it learns about your codebase,
decisions, and preferences, and `query_memory` / `discover_connections` them back later.

## Option A: ReasonGraph Cloud (remote MCP, nothing to run)

```bash
claude mcp add --transport http memory https://memory.primaxiom.ai/mcp \
  --header "Authorization: Bearer rgm_YOUR_KEY"
```

Or in `.mcp.json` at your project root (checked in, key from the environment):

```json
{
  "mcpServers": {
    "memory": {
      "type": "http",
      "url": "https://memory.primaxiom.ai/mcp",
      "headers": { "Authorization": "Bearer ${MEMORY_API_KEY}" }
    }
  }
}
```

## Option B: local (stdio, runs on your machine)

```bash
pip install "reasongraph[service,gliner,fastembed,sqlite]"
claude mcp add memory -- python /path/to/examples/agents/claude_code/local_mcp_server.py
```

`local_mcp_server.py` keeps memory in `~/.reasongraph/memory.sqlite` so it persists
between runs. Set `REASONGRAPH_DATABASE_URL` to move it.

## Suggested `CLAUDE.md` snippet

```
## Memory
You have a persistent graph memory (MCP server "memory"). Use session "<repo-name>".
- At the start of a task, `query_memory` for prior decisions about the files you touch.
- When you learn something durable (an architectural decision, a gotcha, a preference),
  `push_memory` it as one plain sentence.
- For "why" questions, prefer `discover_connections`: it returns how facts link.
```

Tools exposed: `push_memory`, `push_memories`, `query_memory`, `query_memory_detailed`,
`discover_connections`, `answer`, `update_memory`, `delete_memory`, `trace_memory`,
`what_if_memory`, `memory_history`, `list_sessions`.
