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


def _seed_of(f: dict) -> str | None:
    """The fact a discover result was walked from (the first content step of its path)."""
    for step in f.get("path") or []:
        if isinstance(step, dict) and "content" in step:
            return step["content"]
    return None


def _walked(f: dict) -> bool:
    """True when the fact was reached by walking from another fact, not found by wording."""
    seed = _seed_of(f)
    return seed is not None and seed != f.get("content")


@dataclass
class ContextBlock:
    """What the loop recalled for one message."""
    facts: list[dict] = field(default_factory=list)      # {content, scopes, path, causes, cross_session}
    chain: list[dict] = field(default_factory=list)      # cause->effect hops for why/what-if questions
    roots: list[str] = field(default_factory=list)       # root causes the chain terminates in
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
        min_score: direct hits below this cosine score are not used as filler (default 0.25).
        observe_user / observe_assistant: what to remember after each exchange.
        resolve_conflicts: retire facts the new ones replace (uses the graph's resolver).
        redact: optional ``fn(text) -> text | None`` applied before storing; None drops it.
        min_score / min_ratio: a seed or direct hit is kept when its embedding cosine to
            the question clears ``min_score`` AND ``min_ratio`` times the best hit's cosine.
            The ratio does the work: cosines to a vague question ("anything about Friday?")
            sit at 0.13-0.25 for the right facts, while a sharp question puts them at 0.5+
            and filler at 0.15. Facts reached by the walk are kept with their seed.
        rerank_min: optional cross-encoder cutoff applied to the recalled facts on top of
            ``min_score``. Embedding cosine cannot tell "same topic" from "answers this"
            (facts about another Dutch city score like a real hit); the reranker can.
            With the default reranker unrelated facts score below -7 and direct hits
            above -2, so -4 is a safe value. Facts pulled in by the causal walk are
            exempt: they are relevant by structure, not by wording.
        extend_query: when a why-question's chain ends in a root cause that no recalled
            fact states, run one more query with that root cause so the plain fact
            behind it (often the deepest, hardest one to retrieve) is pulled in too.
        header: first line of the injected system message.
    """

    def __init__(self, graph, session: str = "chat", *, recall_scopes=None, max_facts: int = 8,
                 max_chars: int = 1600, top_k: int = 5, hops: int = 3, min_score: float = 0.1,
                 min_ratio: float = 0.45,
                 extend_query: bool = True, rerank_min: float | None = None,
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
        self.extend_query = extend_query
        self.rerank_min = rerank_min
        # direct-hit filler below this cosine score is left out: an empty context beats
        # padding the prompt with unrelated facts
        self.min_score = min_score
        self.min_ratio = min_ratio
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
        # Cutoffs judge only what was found by wording: discover's seeds and the direct
        # hits. A fact reached by walking from a seed through a shared entity is kept
        # because of that structure (it rarely reads like an answer to the question:
        # "Redis runs on the same node as Elasticsearch" scores like noise against
        # "why is checkout slow?"), and is dropped only when its seed is dropped.
        if found and (self.min_score > -1.0 or self.rerank_min is not None):
            seeds = [f for f in found if not _walked(f)]
            walked = [f for f in found if _walked(f)]
            keep = seeds
            if keep and self.min_score > -1.0:
                try:
                    scores = self.graph.embeddings.score(query, [f["content"] for f in keep])
                    floor = max(self.min_score, self.min_ratio * max(scores)) if scores else self.min_score
                    keep = [f for f, sc in zip(keep, scores) if sc >= floor]
                except Exception:
                    pass
            if keep and self.rerank_min is not None:
                try:
                    rel = self.graph.embeddings.relevance(query, [f["content"] for f in keep])
                    keep = [f for f, x in zip(keep, rel) if x >= self.rerank_min]
                except Exception:
                    pass
            kept_seeds = {f["content"] for f in keep}
            # A fact discover picked as a seed by wording may also sit one or two entity
            # hops from a kept seed ("Maria reported that Bulk Export fails" next to
            # "Maria downgraded"): being a seed hid that structure. Walk from the kept
            # seeds and reinstate any dropped seed the walk reaches.
            dropped = {f["content"] for f in seeds} - kept_seeds
            if dropped:
                for k in keep[:5]:
                    try:
                        # seed on the fact itself (top_k=1: its own text) and walk two
                        # entity hops; what comes back with a path is structure, not wording
                        near = await self.graph.discover(k["content"], top_k=1, hops=2, max_results=20,
                                                         scopes=self.recall_scopes)
                    except Exception:
                        continue
                    for r in near:
                        if isinstance(r, dict) and _walked(r) and r["content"] in dropped:
                            kept_seeds.add(r["content"]); dropped.discard(r["content"])
                    if not dropped:
                        break
            keep_set = kept_seeds | {f["content"] for f in walked if _seed_of(f) in kept_seeds}
            found = [f for f in found if f["content"] in keep_set]
            seen = {f["content"] for f in found}     # a dropped fact may come back by structure below
        chain: list[dict] = []
        roots: list[str] = []
        if _QUESTION.search(message) and found:
            # walk backwards from the top few facts: the deepest (root) fact is the one
            # retrieval misses most, and a small model answers with the nearest cause
            # unless the root is spelled out
            hops_seen: set[tuple] = set()
            for f in found[:3]:
                try:
                    traced = await self.graph.trace_causes(f["content"], max_depth=self.hops)
                except Exception:
                    continue
                for h in traced.get("chain", []):
                    key = (h.get("cause"), h.get("effect"))
                    if key not in hops_seen and len(chain) < 8:
                        hops_seen.add(key); chain.append(h)
                for r in traced.get("terminals", []):
                    if r not in roots:
                        roots.append(r)
            for h in chain:                          # pull in the facts that assert those hops
                fact = h.get("fact")
                if fact and fact not in seen and len(found) < self.max_facts:
                    found.append({"content": fact, "scopes": [], "path": [],
                                  "causes": [], "cross_session": False})
                    seen.add(fact)
            if self.extend_query and roots:
                # the chain's loose ends: a root cause span whose own fact is not in
                # context yet (a plain statement with no causal relation of its own).
                # One targeted query per root replaces the vague first question.
                for root in roots[:3]:
                    if len(found) >= self.max_facts:
                        break
                    # no "already stated" short-circuit: the root span is by construction
                    # part of the fact that asserts the last hop, and the plain fact behind
                    # it rarely repeats the span's words. The cosine gate below decides.
                    try:
                        more = await self.graph.query_detailed(root, top_k=2, scopes=self.recall_scopes)
                        more = [r for r in more if isinstance(r, dict)]
                        # query_detailed's score is the reranker's, not bounded: gate on the
                        # embedding cosine to the root span, as the first pass did to the question
                        if more and self.min_score > -1.0:
                            # a root span is short and specific: the fact behind it scores
                            # 0.6+ against it, unrelated facts 0.3 and below
                            sc = self.graph.embeddings.score(root, [r["content"] for r in more])
                            more = [r for r, x in zip(more, sc) if x >= max(self.min_score, 0.35)]
                    except Exception:
                        continue
                    for r in more:
                        content = r["content"]
                        if content not in seen and len(found) < self.max_facts:
                            found.append({"content": content, "scopes": sorted(r.get("scopes", [])),
                                          "path": [], "causes": [], "cross_session": False})
                            seen.add(content)
        block = ContextBlock(facts=found, chain=chain, roots=roots)
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
                lines.append(f"Causal chain behind it: {hops}"); used += len(hops) + 20
        if block.roots:
            roots = "; ".join(block.roots[:4])
            if used + len(roots) + 90 <= self.max_chars:
                lines.append(f"Root cause(s) at the start of that chain: {roots}. "
                             "When asked why, name the root cause as well as the nearest one.")
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
