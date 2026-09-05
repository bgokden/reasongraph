# Example agents

Copy-paste starting points that use reasongraph as **agent memory**. Every example
talks to a memory service over HTTP, so the same code runs against:

- a local service: `pip install "reasongraph[service,gliner,fastembed]"` then `reasongraph-serve`
  (`MEMORY_URL=http://localhost:8000`, no key), or
- [ReasonGraph Cloud](https://memory.primaxiom.ai) (`MEMORY_URL=https://memory.primaxiom.ai`,
  `MEMORY_API_KEY=rgm_...`, EU-hosted, alpha keys on request).

The LLM side is OpenAI-compatible so it works with Groq (default, fast and free tier),
OpenAI, Ollama, or anything else that speaks that API:

```bash
export LLM_BASE_URL=https://api.groq.com/openai/v1   # default
export LLM_API_KEY=gsk_...                            # or OPENAI_API_KEY / GROQ_API_KEY
export LLM_MODEL=openai/gpt-oss-120b                  # default
pip install httpx openai
```

| Example | What it shows | Run |
|---|---|---|
| [`research_agent.py`](research_agent.py) | A tool-using agent that **reads sources, remembers facts, and answers questions** by recalling + discovering connections across everything it has read. | `python research_agent.py` |
| [`two_agents_shared_memory.py`](two_agents_shared_memory.py) | Two agents with **separate sessions on one memory**: the analyst reaches facts only the scout recorded, through shared entities. No message passing. | `python two_agents_shared_memory.py` |
| [`claude_code/`](claude_code/) | **Claude Code with persistent memory** across sessions via MCP (remote HTTP or local stdio). | see its README |
| [`langgraph_agent.py`](langgraph_agent.py) | The research agent as a **LangGraph** graph with memory tools. | `pip install langgraph langchain-openai && python langgraph_agent.py` |
| [`memory_client.py`](memory_client.py) | The 60-line HTTP client the others share. | import it |

What makes this different from a vector store: facts written independently get
connected through extracted entities and cause->effect relations, so `discover`
returns *paths* ("Arizona water cuts -> Phoenix fab -> Apple supply") rather than
nearest neighbours, and one agent's session can reach another's.
