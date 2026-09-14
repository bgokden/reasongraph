"""LangChain and LangGraph adapters.

Five things, each a few lines to use:

* :class:`ReasonGraphRetriever` -- a ``BaseRetriever`` whose documents are the facts the
  graph connects to the question (paths through shared names, cause->effect links),
  for any RAG chain that takes a retriever.
* :func:`with_memory` -- wraps a chat model so every call gets what the graph remembers
  injected before the messages, and every exchange is remembered afterwards
  (the deep memory integration, as a Runnable).
* :func:`memory_tools` -- ``remember`` / ``recall`` / ``discover`` tools for agents
  (``langgraph.prebuilt.create_react_agent(model, tools=memory_tools(...))``).
* :class:`ReasonGraphMemory` -- the classic ``BaseMemory`` shape (``load_memory_variables``
  / ``save_context``): what the agent sees each turn is the facts the graph connects to
  the input, with sources, plus the transcript folded to a token budget with a rolling
  summary. Nothing is deleted by folding; a folded-out turn is still a fact and comes
  back by recall.
* :class:`ReasonGraphChatMessageHistory` -- the same transcript as a
  ``BaseChatMessageHistory`` for ``RunnableWithMessageHistory`` and LangGraph.

All of them work with a local :class:`reasongraph.ReasonGraph` or a hosted service through
:class:`reasongraph.client.MemoryClient`. Install with ``pip install reasongraph[langchain]``.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, PrivateAttr

# ``BaseMemory`` lived in langchain-core until 1.0 and in langchain-classic after; without
# either the class keeps the same shape (a pydantic model with the four memory methods) so
# it still drops into anything that duck-types a memory.
try:  # pragma: no cover - which import succeeds depends on the installed LangChain
    from langchain_core.memory import BaseMemory as _BaseMemory
except ImportError:
    try:
        from langchain_classic.memory import BaseMemory as _BaseMemory
    except ImportError:
        _BaseMemory = BaseModel

__all__ = ["ReasonGraphRetriever", "with_memory", "memory_tools",
           "ReasonGraphMemory", "ReasonGraphChatMessageHistory"]


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


# -- memory classes -------------------------------------------------------------


def _run(target: Any, coro):
    """Drive a coroutine from sync code, on the graph's loop when there is one."""
    if hasattr(target, "_run"):
        return target._run(coro)
    return asyncio.run(coro)


def _text_of(m: Any) -> str:
    return str(m.content if isinstance(m, BaseMessage) else m)


def _to_dicts(msgs: list[BaseMessage]) -> list[dict]:
    role = {HumanMessage: "user", AIMessage: "assistant", SystemMessage: "system"}
    return [{"role": next((r for t, r in role.items() if isinstance(m, t)), "user"), "content": _text_of(m)}
            for m in msgs]


def _from_dicts(msgs: list[dict]) -> list[BaseMessage]:
    kinds = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
    return [kinds.get(m["role"], HumanMessage)(content=m["content"]) for m in msgs]


def _as_string(msgs: list[dict], human_prefix: str, ai_prefix: str) -> str:
    names = {"user": human_prefix, "assistant": ai_prefix, "system": "Summary"}
    return "\n".join(f"{names.get(m['role'], m['role'])}: {m['content']}" for m in msgs)


class _Transcript:
    """The conversation so far plus the folding that decides what the model sees.

    Every turn is stored as a fact in ``target`` when observed; the transcript held here
    is the folded view (rolling summary, then the newest turns), which is what gets
    injected. With a local ``ReasonGraph`` the library's ``MemoryLoop`` does both
    recall and folding; with a ``MemoryClient`` recall goes through the hosted
    ``discover`` and folding runs locally on the same code.
    """

    def __init__(self, target: Any, session: str, *, max_facts: int, hops: int,
                 max_history_tokens: int | None, keep_tail_tokens: int | None,
                 summarizer: Callable[[list[dict]], str] | None, observe_assistant: bool,
                 header: str) -> None:
        from reasongraph.loop import MemoryLoop
        self.target, self.session, self.hops, self.max_facts, self.header = target, session, hops, max_facts, header
        self.client = _is_client(target)
        self.loop = MemoryLoop(None if self.client else target, session=session, max_facts=max_facts, hops=hops,
                               max_history_tokens=max_history_tokens, keep_tail_tokens=keep_tail_tokens,
                               summarizer=summarizer, summarize_in_background=True,
                               observe_assistant=observe_assistant)
        self.observe_assistant = observe_assistant
        self.history: list[dict] = []

    def recall(self, question: str) -> str:
        if not question:
            return ""
        if self.client:
            facts = _connections(self.target, question, None, 5, self.hops, self.max_facts)
            return _render(facts, self.header) if facts else ""
        previous = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), None)
        return self.loop.recall_sync(question, previous=previous).text

    def folded(self) -> list[dict]:
        return self.loop.fold_history(self.history)

    def add(self, user: str | None, assistant: str | None) -> None:
        if user:
            self.history.append({"role": "user", "content": user})
        if assistant:
            self.history.append({"role": "assistant", "content": assistant})
        if self.client:
            texts = [t for t, keep in ((user, True), (assistant, self.observe_assistant)) if t and keep]
            if texts:
                self.target.remember_many(self.session, texts)
        else:
            self.loop.observe_sync(user, assistant)
        # Fold now that the reply is out, never in the middle of a turn: the turns past the
        # budget become the rolling summary and the transcript is replaced by the folded view,
        # so the next fold merges the previous summary instead of stacking another on top.
        folded = self.loop.fold_history(self.history)
        if self.loop._pending is not None:
            if _run(None if self.client else self.target, self.loop.flush_summary()):
                folded = self.loop.fold_history(self.history)
        self.history = folded

    @property
    def summary(self) -> str | None:
        return self.loop._summary

    def clear(self) -> None:
        self.history = []
        self.loop._summary = None
        self.loop._pending = None
        if self.client:
            self.target.forget_session(self.session)
        else:
            self.target.forget_sync({self.session})


class ReasonGraphMemory(_BaseMemory):
    """Memory for a chain or agent, in the classic ``BaseMemory`` shape.

    ``load_memory_variables`` returns two variables:

    * ``memory_key`` (default ``"memory"``): the facts the graph connects to the input,
      rendered with their sources and cause->effect links, empty when nothing relates.
    * ``history_key`` (default ``"history"``): the transcript, folded to
      ``max_history_tokens`` with the oldest turns as one rolling summary and the newest
      ``keep_tail_tokens`` verbatim. Unset means the whole transcript, as a buffer memory.

    ``save_context`` stores the exchange as facts in ``session`` and appends it to the
    transcript. Folding decides what the model sees; it deletes nothing, and a turn that
    left the window comes back through ``memory_key`` when it is relevant again.
    ``clear`` forgets the session in the graph as well as the transcript.

    ``target`` is a local ``ReasonGraph`` or a ``MemoryClient``. ``summarizer`` takes the
    folded-out messages (``[{"role", "content"}]``) and returns a summary; it runs after
    ``save_context``, never in the middle of a turn. Without one the budget still holds
    and the old turns simply leave the window.
    """

    target: Any
    session: str = "chat"
    memory_key: str = "memory"
    history_key: str = "history"
    input_key: str | None = None
    output_key: str | None = None
    return_messages: bool = False
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    max_facts: int = 8
    hops: int = 3
    max_history_tokens: int | None = None
    keep_tail_tokens: int | None = None
    summarizer: Any = None
    observe_assistant: bool = True
    header: str = "What you remember that is relevant (with sources):"

    model_config = {"arbitrary_types_allowed": True}
    _transcript: _Transcript = PrivateAttr(default=None)

    def _t(self) -> _Transcript:
        if self._transcript is None:
            self._transcript = _Transcript(self.target, self.session, max_facts=self.max_facts, hops=self.hops,
                                           max_history_tokens=self.max_history_tokens,
                                           keep_tail_tokens=self.keep_tail_tokens, summarizer=self.summarizer,
                                           observe_assistant=self.observe_assistant, header=self.header)
        return self._transcript

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key, self.history_key]

    def _pick(self, values: dict[str, Any], key: str | None, exclude: list[str]) -> str | None:
        if key is not None:
            v = values.get(key)
            return _text_of(v) if v is not None else None
        candidates = [k for k in values if k not in exclude]
        if len(candidates) != 1:
            raise ValueError(f"one input expected, got {sorted(candidates)}; set input_key/output_key")
        return _text_of(values[candidates[0]])

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        t = self._t()
        question = self._pick(inputs, self.input_key, self.memory_variables) if inputs else None
        folded = t.folded()
        history: Any = _from_dicts(folded) if self.return_messages else _as_string(folded, self.human_prefix, self.ai_prefix)
        return {self.memory_key: t.recall(question or ""), self.history_key: history}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        user = self._pick(inputs, self.input_key, self.memory_variables)
        reply = self._pick(outputs, self.output_key, [])
        self._t().add(user, reply)

    def clear(self) -> None:
        self._t().clear()

    @property
    def summary(self) -> str | None:
        """The rolling summary of the turns that left the window, once one has been written."""
        return self._t().summary

    @property
    def messages(self) -> list[BaseMessage]:
        """The transcript as the model would see it now: summary first, newest turns verbatim."""
        return _from_dicts(self._t().folded())


class ReasonGraphChatMessageHistory(BaseChatMessageHistory):
    """The transcript as a ``BaseChatMessageHistory``, for ``RunnableWithMessageHistory``.

    ``messages`` is the folded view (rolling summary, then the newest turns verbatim) and
    ``add_messages`` stores each exchange as facts in ``session``. Pair it with
    :class:`ReasonGraphRetriever` or :func:`with_memory` for the recall half.
    """

    def __init__(self, target: Any, session: str = "chat", *, max_history_tokens: int | None = None,
                 keep_tail_tokens: int | None = None, summarizer: Callable[[list[dict]], str] | None = None,
                 observe_assistant: bool = True) -> None:
        self._t = _Transcript(target, session, max_facts=8, hops=3, max_history_tokens=max_history_tokens,
                              keep_tail_tokens=keep_tail_tokens, summarizer=summarizer,
                              observe_assistant=observe_assistant, header="")

    @property
    def messages(self) -> list[BaseMessage]:
        return _from_dicts(self._t.folded())

    def add_messages(self, messages: list[BaseMessage]) -> None:
        user = assistant = None
        for m in messages:
            if isinstance(m, AIMessage):
                assistant = _text_of(m)
            elif not isinstance(m, SystemMessage):
                if user is not None:
                    self._t.add(user, assistant); assistant = None
                user = _text_of(m)
        if user is not None or assistant is not None:
            self._t.add(user, assistant)

    def clear(self) -> None:
        self._t.clear()
