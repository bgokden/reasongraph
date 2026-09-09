"""ReasonGraph + LangChain in three ways: a retriever, a remembering model, agent tools.

    pip install "reasongraph[langchain]" langchain-openai
    MEMORY_API_KEY=rgm_... OPENAI_API_KEY=... python langchain_memory.py

Swap MemoryClient for a local ReasonGraph() and everything below works the same.
"""
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from reasongraph.client import MemoryClient
from reasongraph.integrations.langchain import ReasonGraphRetriever, memory_tools, with_memory

mem = MemoryClient(api_key=os.environ["MEMORY_API_KEY"])
mem.remember_many("scout", ["TSMC is building a chip fab in Phoenix, Arizona.",
                            "Arizona declared a water emergency after a drought."])
mem.remember("analyst", "Apple depends on TSMC for its M-series processors.")

# 1. A retriever for any RAG chain: the facts the graph connects to the question.
retriever = ReasonGraphRetriever(target=mem)
for doc in retriever.invoke("Why might Apple face shortages?"):
    print("-", doc.page_content, doc.metadata["sources"], doc.metadata["via"])

# 2. A model that remembers: recall is injected before every call, the exchange is stored after.
model = with_memory(ChatOpenAI(model="gpt-4o-mini"), mem, session="support-chat")
print(model.invoke([HumanMessage(content="Why might Apple face shortages?")]).content)

# 3. Tools for an agent (LangGraph's create_react_agent takes them as they are).
tools = memory_tools(mem, session="agent")
print([t.name for t in tools])
