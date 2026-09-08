"""Deep memory integration: a loop around any chat model.

Everything the agent hears or says becomes memory, and everything relevant comes back
by itself before each model call. No tool calls needed from the model.

    loop = MemoryLoop(graph, session="support-chat")
    reply = loop.chat(call_model, history)      # recall -> inject -> call -> observe

``call_model`` is any ``fn(messages) -> str`` (OpenAI-style message dicts), so the loop
works with Groq, Ollama, llama.cpp, the Claude API, or a LangGraph node.

The hot path stays local: recall is embeddings plus graph walks; the chat model is the
agent's own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

_QUESTION = re.compile(r"\?\s*$|^(why|how|what|when|who|which|where|is|are|does|do|did|can|could|should|will)\b", re.I)


@dataclass
class ContextBlock:
    """What the loop recalled for one message."""
    facts: list[dict] = field(default_factory=list)      # {content, scopes, path, causes, cross_session}
    chain: list[dict] = field(default_factory=list)      # cause->effect hops for why/what-if questions
    text: str = ""                                       # the rendered block injected into the prompt

    @property
    def empty(self) -> bool:
        return not self.facts and not self.chain


class MemoryLoop:
    """Recall before, observe after, for every exchange.

    Args:
        graph: a ReasonGraph (or a MemoryService's ``.graph``).
        session: where this conversation's facts are stored; recall still walks all
            sessions unless ``recall_scopes`` narrows it.
        recall_scopes: seed scopes for recall (None = everything).
        max_facts / max_chars: context budget, most relevant first.
        observe_user / observe_assistant: what to remember after each exchange.
        resolve_conflicts: retire facts the new ones replace (uses the graph's resolver).
        redact: optional ``fn(text) -> text | None`` applied before storing; None drops it.
        header: first line of the injected system message.
    """

    def __init__(self, graph, session: str = "chat", *, recall_scopes=None, max_facts: int = 8,
                 max_chars: int = 1600, top_k: int = 5, hops: int = 3,
                 observe_user: bool = True, observe_assistant: bool = True,
                 resolve_conflicts: bool = False, redact: Callable[[str], str | None] | None = None,
                 header: str = "What you remember that is relevant (with sources):") -> None:
        self.graph = graph
        self.session = session
        self.recall_scopes = set(recall_scopes) if recall_scopes else None
        self.max_facts = max_facts
        self.max_chars = max_chars
        self.top_k = top_k
        self.hops = hops
        self.observe_user = observe_user
        self.observe_assistant = observe_assistant
        self.resolve_conflicts = resolve_conflicts
        self.redact = redact
        self.header = header

    # -- recall ---------------------------------------------------------------

    async def recall(self, message: str, *, previous: str | None = None) -> ContextBlock:
        """Facts, paths and causal hops relevant to ``message`` (and the previous turn)."""
        query = message if not previous else f"{previous}\n{message}"
        found = await self.graph.discover(query, top_k=self.top_k, hops=self.hops,
                                          max_results=self.max_facts, scopes=self.recall_scopes)
        seen = {f["content"] for f in found}
        if len(found) < self.max_facts:          # discover walks; query fills with direct hits
            direct = await self.graph.query_detailed(query, top_k=self.max_facts, scopes=self.recall_scopes)
            for r in direct:
                content = r["content"] if isinstance(r, dict) else str(r)
                scopes = sorted(r.get("scopes", [])) if isinstance(r, dict) else []
                if content not in seen and len(found) < self.max_facts:
                    found.append({"content": content, "scopes": scopes, "path": [],
                                  "causes": [], "cross_session": False})
                    seen.add(content)
        chain: list[dict] = []
        if _QUESTION.search(message) and found:
            try:
                traced = await self.graph.trace_causes(found[0]["content"], max_depth=self.hops)
                chain = traced.get("chain", [])[:6]
            except Exception:
                chain = []
        block = ContextBlock(facts=found, chain=chain)
        block.text = self._render(block)
        return block

    def _render(self, block: ContextBlock) -> str:
        if block.empty:
            return ""
        lines = [self.header]
        used = len(lines[0])
        for f in block.facts:
            src = ", ".join(f.get("scopes") or []) or "memory"
            via = " via " + ", ".join(st["entity"] for st in f.get("path", []) if "entity" in st) if any("entity" in st for st in f.get("path", [])) else ""
            line = f"- {f['content']} [{src}{via}]"
            for c in f.get("causes") or []:
                line += f"\n  because: {c['cause']} -> {c['effect']}"
            if used + len(line) > self.max_chars:
                break
            lines.append(line); used += len(line)
        if block.chain:
            hops = "; ".join(f"{h['cause']} -> {h['effect']}" for h in block.chain)
            if used + len(hops) + 20 <= self.max_chars:
                lines.append(f"Causal chain behind it: {hops}")
        return "\n".join(lines)

    # -- observe --------------------------------------------------------------

    async def observe(self, user: str | None = None, assistant: str | None = None) -> list[str]:
        """Remember the exchange (sentence-split by the graph); returns what was stored."""
        texts: list[str] = []
        for flag, text in ((self.observe_user, user), (self.observe_assistant, assistant)):
            if not flag or not text:
                continue
            if self.redact is not None:
                text = self.redact(text)
                if not text:
                    continue
            texts.append(text)
        if not texts:
            return []
        await self.graph.add_texts(texts, scopes={self.session}, resolve_conflicts=self.resolve_conflicts)
        return texts

    # -- glue -----------------------------------------------------------------

    async def messages(self, history: list[dict], *, system: str | None = None) -> tuple[list[dict], ContextBlock]:
        """OpenAI-style messages with the recalled context injected as a system message.
        ``history`` is the conversation so far, last item the user's new message."""
        user_msgs = [m["content"] for m in history if m.get("role") == "user"]
        last_user = user_msgs[-1] if user_msgs else ""
        prev_assistant = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), None)
        block = await self.recall(last_user, previous=prev_assistant) if last_user else ContextBlock()
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        if block.text:
            out.append({"role": "system", "content": block.text})
        out.extend(history)
        return out, block

    async def chat(self, call_model: Callable[[list[dict]], Any], history: list[dict], *,
                   system: str | None = None) -> tuple[str, ContextBlock]:
        """One full turn: recall -> inject -> call the model -> observe. ``call_model`` may be
        sync or async and must return the reply text."""
        msgs, block = await self.messages(history, system=system)
        reply = call_model(msgs)
        if hasattr(reply, "__await__"):
            reply = await reply
        reply = str(reply)
        last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), None)
        await self.observe(last_user, reply)
        return reply, block

    # -- sync wrappers ---------------------------------------------------------

    def recall_sync(self, message: str, **kw) -> ContextBlock:
        return self.graph._run(self.recall(message, **kw))

    def observe_sync(self, user: str | None = None, assistant: str | None = None) -> list[str]:
        return self.graph._run(self.observe(user, assistant))

    def messages_sync(self, history: list[dict], **kw):
        return self.graph._run(self.messages(history, **kw))

    def chat_sync(self, call_model, history: list[dict], **kw):
        return self.graph._run(self.chat(call_model, history, **kw))
