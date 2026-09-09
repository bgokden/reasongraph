"""LangChain and LangGraph adapters.

Three things, each a few lines to use:

* :class:`ReasonGraphRetriever` -- a ``BaseRetriever`` whose documents are the facts the
  graph connects to the question (paths through shared names, cause->effect links),
  for any RAG chain that takes a retriever.
* :func:`with_memory` -- wraps a chat model so every call gets what the graph remembers
  injected before the messages, and every exchange is remembered afterwards
  (the deep memory integration, as a Runnable).
* :func:`memory_tools` -- ``remember`` / ``recall`` / ``discover`` tools for agents
  (``langgraph.prebuilt.create_react_agent(model, tools=memory_tools(...))``).

All three work with a local :class:`reasongraph.ReasonGraph` or a hosted service through
:class:`reasongraph.client.MemoryClient`. Install with ``pip install reasongraph[langchain]``.
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import StructuredTool

__all__ = ["ReasonGraphRetriever", "with_memory", "memory_tools"]


def _is_client(target: Any) -> bool:
    return hasattr(target, "discover") and hasattr(target, "remember_many") and not hasattr(target, "add_texts")


def _connections(target: Any, query: str, session: str | None, top_k: int, hops: int, max_results: int) -> list[dict]:
    if _is_client(target):
        return list(target.discover(query, session=session, top_k=top_k, hops=hops, max_results=max_results))
    scopes = {session} if session else None
    found = target.discover_sync(query, top_k=top_k, hops=hops, max_results=max_results, scopes=scopes)
    return list(found if isinstance(found, list) else found.get("connections", []))


def _to_document(c: dict) -> Document:
    return Document(
        page_content=c["content"],
        metadata={
            "sources": list(c.get("scopes") or []),
            "via": [st["entity"] for st in (c.get("path") or []) if "entity" in st],
            "causes": list(c.get("causes") or []),
            "cross_session": bool(c.get("cross_session")),
        },
    )


class ReasonGraphRetriever(BaseRetriever):
    """Facts the graph connects to the question, as LangChain documents.

    ``target`` is a ``ReasonGraph`` or a ``MemoryClient``; ``session`` limits the seeds
    to one session (the walk still crosses sessions). Metadata carries the sources, the
    names the fact was reached through and its cause->effect links.
    """

    target: Any
    session: str | None = None
    top_k: int = 5
    hops: int = 3
    max_results: int = 8

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [_to_document(c) for c in _connections(self.target, query, self.session, self.top_k, self.hops, self.max_results)]


def _render(facts: list[dict], header: str) -> str:
    lines = [header]
    for f in facts:
        src = ", ".join(f.get("scopes") or []) or "memory"
        line = f"- {f['content']} [{src}]"
        for r in f.get("causes") or []:
            line += f"\n  because: {r['cause']} -> {r['effect']}"
        lines.append(line)
    lines.append("These are your own memories: answer from them when they relate to the question.")
    return "\n".join(lines)


def with_memory(model: Runnable, target: Any, session: str = "chat", *, recall_session: str | None = None,
                max_facts: int = 8, hops: int = 3, observe: bool = True, observe_assistant: bool = False,
                header: str = "What you remember that is relevant (with sources):") -> Runnable:
    """Wrap a chat model so it remembers.

    The returned Runnable takes a list of messages (or ``{"messages": [...]}``), recalls
    what the graph connects to the latest human message, injects it as a system message
    after any system prompt, calls ``model``, and remembers the exchange in ``session``.
    With a local ``ReasonGraph`` the library's ``MemoryLoop`` does the recall (traced
    chains, root causes, follow-up queries); with a ``MemoryClient`` the hosted
    ``discover`` does.
    """
    loop = None
    if not _is_client(target):
        from reasongraph.loop import MemoryLoop
        loop = MemoryLoop(target, session=session, max_facts=max_facts, hops=hops,
                          recall_scopes={recall_session} if recall_session else None,
                          observe_assistant=observe_assistant)

    def _messages(x: Any) -> list[BaseMessage]:
        msgs = x["messages"] if isinstance(x, dict) else x
        return list(msgs)

    def _inject(msgs: list[BaseMessage]) -> list[BaseMessage]:
        humans = [m for m in msgs if isinstance(m, HumanMessage)]
        if not humans:
            return msgs
        question = str(humans[-1].content)
        if loop is not None:
            block = loop.recall_sync(question)
            text = block.text
        else:
            facts = _connections(target, question, recall_session, 5, hops, max_facts)
            text = _render(facts, header) if facts else ""
        if not text:
            return msgs
        sys_idx = 0
        while sys_idx < len(msgs) and isinstance(msgs[sys_idx], SystemMessage):
            sys_idx += 1
        return msgs[:sys_idx] + [SystemMessage(content=text)] + msgs[sys_idx:]

    def _remember(pair: dict) -> AIMessage:
        reply, msgs = pair["reply"], pair["messages"]
        if observe:
            humans = [m for m in msgs if isinstance(m, HumanMessage)]
            texts = [str(humans[-1].content)] if humans else []
            reply_text = reply.content if isinstance(reply, BaseMessage) else str(reply)
            if observe_assistant and reply_text:
                texts.append(str(reply_text))
            if texts:
                if loop is not None:
                    loop.observe_sync(texts[0], texts[1] if len(texts) > 1 else None)
                else:
                    target.remember_many(session, texts)
        return reply if isinstance(reply, AIMessage) else AIMessage(content=str(getattr(reply, "content", reply)))

    def _run(x: Any) -> AIMessage:
        msgs = _inject(_messages(x))
        reply = model.invoke(msgs)
        return _remember({"reply": reply, "messages": msgs})

    return RunnableLambda(_run)


def memory_tools(target: Any, session: str = "agent") -> list[StructuredTool]:
    """``remember``, ``recall`` and ``discover`` as tools for a LangChain or LangGraph agent."""

    def remember(text: str) -> str:
        """Store one fact (a plain sentence) in memory."""
        if _is_client(target):
            target.remember(session, text)
        else:
            target.add_texts_sync([text], scopes={session})
        return f"Remembered: {text}"

    def recall(query: str) -> str:
        """Facts in memory most similar to the query, one per line."""
        if _is_client(target):
            facts = target.recall(query, top_k=5)
            return "\n".join(f["content"] if isinstance(f, dict) else str(f) for f in facts) or "Nothing remembered about that."
        return "\n".join(target.query_sync(query, top_k=5)) or "Nothing remembered about that."

    def discover(question: str) -> str:
        """Facts connected to the question through shared names and cause->effect links, with sources."""
        conns = _connections(target, question, None, 5, 3, 8)
        if not conns:
            return "Nothing in memory connects to that."
        return _render(conns, "What connects:")

    return [
        StructuredTool.from_function(remember, name="remember", description=remember.__doc__),
        StructuredTool.from_function(recall, name="recall", description=recall.__doc__),
        StructuredTool.from_function(discover, name="discover", description=discover.__doc__),
    ]
