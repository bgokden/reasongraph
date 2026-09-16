# langchain-reasongraph

LangChain integration for [ReasonGraph](https://github.com/bgokden/reasongraph), a graph
memory for AI agents that links facts by entities and cause -> effect and answers *why*
questions with the chain of facts behind the answer.

```bash
pip install langchain-reasongraph
```

## Retriever

```python
from reasongraph import ReasonGraph
from langchain_reasongraph import ReasonGraphRetriever

graph = ReasonGraph()
graph.initialize_sync()
graph.add_texts_sync([
    "The checkout service stores shopping carts in Redis since the March release.",
    "Redis runs on the same node as Elasticsearch.",
    "Because Elasticsearch rebuilds its index at 09:00 every day, the node's CPU is saturated each morning.",
])

retriever = ReasonGraphRetriever(target=graph, k=4)
for doc in retriever.invoke("Why is the checkout service slow every morning?"):
    print(doc.page_content, doc.metadata["via"], doc.metadata["causes"])
```

`metadata` carries `sources` (the sessions a fact came from), `via` (the names the fact
was reached through), `causes` (its cause -> effect links) and `cross_session`.

Hosted: `target=MemoryClient("https://memory.primaxiom.ai", api_key="rgm_...")` from
`reasongraph.client` uses the same retriever against ReasonGraph Cloud (free plan, no card).

## Memory and tools

`ReasonGraphMemory` (the classic `BaseMemory` shape), `ReasonGraphChatMessageHistory`
(for `RunnableWithMessageHistory` and LangGraph), `with_memory(model, target)` (recall
before the call, remember after) and `memory_tools(target)` (remember / recall / discover
as `StructuredTool`s for agents) are re-exported from `reasongraph.integrations.langchain`;
examples in the main repo under `examples/agents/langchain_*.py`.

## Tests

```bash
pip install "langchain-reasongraph[test]"
pytest packages/langchain-reasongraph/tests
```

The integration tests are LangChain's standard `RetrieversIntegrationTests` on an
in-memory graph; no service or API key needed.
