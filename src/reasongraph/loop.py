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

import os

import re
from dataclasses import dataclass, field
from typing import Any, Callable

_QUESTION = re.compile(r"\?\s*$|^(why|how|what|when|who|which|where|is|are|does|do|did|can|could|should|will)\b", re.I)

#: First line of a folded-history summary, and how the next fold recognises its own work.
SUMMARY_HEADER = "Summary of earlier messages in this conversation:"


def _approx_tokens(text: str) -> int:
    """Tokens, approximately, without loading a tokenizer.

    A real count needs the chat model's own tokenizer, which the loop does not have and
    should not guess at. Four characters per token is the usual English rule of thumb and
    runs a little low on other languages, so budgets set with it are conservative rather
    than optimistic. Pass ``count_tokens`` to use the real thing.
    """
    return max(1, (len(text) + 3) // 4)


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
    narrative: str = ""                                  # the chain's facts, root first, as one passage
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
                 search_mode: str | None = None,
                 causal_hops: tuple[int, int] | None = None,
                 max_history_tokens: int | None = None, keep_tail_tokens: int | None = None,
                 summarizer: Callable[[list[dict]], str] | None = None,
                 summarize_in_background: bool = False,
                 count_tokens: Callable[[str], int] | None = None,
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
        # History folding: when the transcript passes max_history_tokens, the oldest messages
        # become one summary and the newest keep_tail_tokens stay verbatim. Folding a block at a
        # time rather than a message at a time means the summarizer runs rarely, and a
        # conversation that never reaches the budget never summarises at all.
        self.max_history_tokens = max_history_tokens
        self.keep_tail_tokens = keep_tail_tokens
        self.summarizer = summarizer
        # Off the hot path: the fold itself never calls the summarizer. It folds against the
        # summary it already has and records what still needs summarising; that work runs
        # after the turn when background is on, or inline before the call when it is off.
        self.summarize_in_background = summarize_in_background
        self.count_tokens = count_tokens or _approx_tokens
        self._summary: str | None = None
        self._pending: list[dict] | None = None
        # how the seeds are found: "embedding" (default), "hybrid" (cosine fused with word-level
        # trigram matches: names, codes, numbers) or "keyword"; REASONGRAPH_LOOP_SEARCH sets the default
        self.search_mode = search_mode or os.environ.get("REASONGRAPH_LOOP_SEARCH", "embedding")
        # separate depth for consequences and for causes, e.g. (3, 2); None keeps the walk
        # symmetric. REASONGRAPH_CAUSAL_HOPS="3,2" sets it without code.
        if causal_hops is None:
            env = os.environ.get("REASONGRAPH_CAUSAL_HOPS", "").strip()
            if env:
                try:
                    f, _, b = env.partition(",")
                    causal_hops = (int(f), int(b or f))
                except ValueError:
                    causal_hops = None
        self.causal_hops = causal_hops
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
        cache = getattr(self.graph, "request_cache", None)
        if cache is None:      # a graph-like object without the memo (a tenant view, a stub)
            return await self._recall(message, previous=previous)
        async with cache():
            return await self._recall(message, previous=previous)

    async def _recall(self, message: str, previous: str | None = None):
        # A short follow-up ("and why?") needs the previous turn to mean anything; a full
        # question does not, and dragging the previous answer into the query pulls the
        # seeds back to the old topic when the conversation moves on.
        short = len(message.split()) < 4
        query = f"{previous[:300]}\n{message}" if (previous and short) else message
        found = await self.graph.discover(query, top_k=self.top_k, hops=self.hops,
                                          max_results=self.max_facts, scopes=self.recall_scopes,
                                          search_mode=self.search_mode, causal_hops=self.causal_hops)
        seen = {f["content"] for f in found}
        if len(found) < self.max_facts:          # discover walks; query fills with direct hits
            direct = await self.graph.query_detailed(query, top_k=self.max_facts, scopes=self.recall_scopes,
                                                     search_mode=self.search_mode)
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
        # The conversation's own turns (stored by observe) come back here too. They stay
        # memory, but they must not crowd out real facts: the question itself and the
        # model's earlier "I don't know" answers score highest against the question and
        # would take every slot. They are judged after everything else, never as the best
        # hit, and never when they repeat the current message.
        own = [f for f in found if self._is_own_turn(f, message)]
        found = [f for f in found if not self._is_own_turn(f, message)]
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
        if own:
            own = [f for f in own if f["content"].strip() != message.strip()]
            if own and self.min_score > -1.0:
                try:
                    sc = self.graph.embeddings.score(query, [f["content"] for f in own])
                    own = [f for f, x in zip(own, sc) if x >= max(self.min_score, 0.3)]
                except Exception:
                    pass
            for f in own[: max(0, self.max_facts - len(found))]:
                found.append(f); seen.add(f["content"])
        chain: list[dict] = []
        roots: list[str] = []
        chain_ids: set[str] = set()          # facts that belong to the traced chain: they keep their slot
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
                if traced.get("chain"):
                    chain_ids.add(f["content"])
                for h in traced.get("chain", []):
                    key = (h.get("cause"), h.get("effect"))
                    if key not in hops_seen and len(chain) < 8:
                        hops_seen.add(key); chain.append(h)
                for r in traced.get("terminals", []):
                    if r not in roots:
                        roots.append(r)
            for h in chain:                          # pull in the facts that assert those hops
                fact = h.get("fact")
                if fact and fact not in seen:
                    found.append({"content": fact, "scopes": [], "path": [],
                                  "causes": [], "cross_session": False})
                    seen.add(fact); chain_ids.add(fact)
            if self.extend_query and roots:
                # the chain's loose ends: a root cause span whose own fact is not in
                # context yet (a plain statement with no causal relation of its own).
                # One targeted query per root replaces the vague first question.
                for root in roots[:3]:
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
                        if content not in seen:
                            found.append({"content": content, "scopes": sorted(r.get("scopes", [])),
                                          "path": [], "causes": [], "cross_session": False})
                            seen.add(content); chain_ids.add(content)
        # The deepest fact of the chain rarely states its own cause; the plain fact behind it
        # ("the boiler was switched off in April") shares a name with it and nothing else. One
        # entity hop from that fact, plain facts only (no causal relation of their own), cap 2.
        if chain and _QUESTION.search(message):
            deepest = None
            produced = {h.get("effect") for h in chain}
            for h in sorted(chain, key=lambda h: -int(h.get("depth", 0))):
                if h.get("cause") not in produced and h.get("fact"):
                    deepest = h["fact"]; break
            if deepest is None:
                deepest = max(chain, key=lambda h: int(h.get("depth", 0))).get("fact")
            if deepest:
                try:
                    near = await self.graph.discover(deepest, top_k=1, hops=1, max_results=12,
                                                     scopes=self.recall_scopes)
                except Exception:
                    near = []
                added = 0
                for r in near:
                    if not isinstance(r, dict) or not _walked(r) or r["content"] in seen:
                        continue
                    if r.get("causes"):
                        continue                      # a causal fact would have been a hop already
                    found.append({"content": r["content"], "scopes": sorted(r.get("scopes", [])),
                                  "path": r.get("path", []), "causes": [], "cross_session": r.get("cross_session", False)})
                    seen.add(r["content"]); chain_ids.add(r["content"]); added += 1
                    if added >= 2:
                        break
        # The context budget: a fact of the traced chain keeps its slot ahead of anything
        # found by wording alone (in a busy memory most misses were roots that were reached
        # and then lost to filler), and the conversation's own turns come last.
        if len(found) > self.max_facts or chain_ids:
            own_ids = {f["content"] for f in found if self._is_own_turn(f, message)}
            first = [f for f in found if f["content"] in chain_ids]
            middle = [f for f in found if f["content"] not in chain_ids and f["content"] not in own_ids]
            last = [f for f in found if f["content"] in own_ids and f["content"] not in chain_ids]
            found = (first + middle + last)[: self.max_facts]
        missing_scopes = [f["content"] for f in found if not f.get("scopes")]
        if missing_scopes:
            try:
                sc = await self.graph.backend.get_scopes(missing_scopes)
                for f in found:
                    if not f.get("scopes") and f["content"] in sc:
                        f["scopes"] = sorted(sc[f["content"]])
            except Exception:
                pass
        narrative = ""
        if chain:
            # cause -> effect order: start from hops whose cause nothing else produces (the
            # roots) and follow effect -> next cause; hops merged from several traces carry
            # depths on different bases, so depth only breaks ties
            produced = {h.get("effect") for h in chain}
            starts = [h for h in chain if h.get("cause") not in produced] or chain
            order: list[dict] = []
            def walk(h):
                if any(h is x for x in order):
                    return
                order.append(h)
                for n in chain:
                    if n.get("cause") == h.get("effect"):
                        walk(n)
            for h in sorted(starts, key=lambda h: -int(h.get("depth", 0))):
                walk(h)
            for h in sorted(chain, key=lambda h: -int(h.get("depth", 0))):
                walk(h)
            ordered: list[str] = []
            for h in order:
                fact = h.get("fact")
                if fact and fact not in ordered:
                    ordered.append(fact)
            known = {f["content"] for f in found}
            ordered = [f for f in ordered if f in known] or ordered
            narrative = " ".join(ordered)
        block = ContextBlock(facts=found, chain=chain, roots=roots, narrative=narrative)
        block.text = self._render(block)
        return block

    def _is_own_turn(self, f: dict, message: str) -> bool:
        """A fact that lives only in this loop's session (a stored conversation turn).

        Hosted services tag every fact with the session ("tenant/chat") and with the
        tenant ("tenant") as well; the tenant tag is a prefix of the session tag and
        does not make the fact belong to anything else.
        """
        scopes = [str(s) for s in (f.get("scopes") or [])]
        if not scopes:
            return False
        sess = [s for s in scopes if s == self.session or s.endswith("/" + self.session)]
        if not sess:
            return False
        others = [s for s in scopes if s not in sess and not any(x.startswith(s + "/") for x in sess)]
        return not others

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
                used += len(roots) + 90
        if block.narrative and used + len(block.narrative) + 60 <= self.max_chars:
            lines.append(f"In order, cause to effect: {block.narrative}")
        if block.facts:
            lines.append("These are your own memories: answer from them when they relate to the question, "
                         "and do not say you have no record of something listed above.")
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


    # -- history folding ------------------------------------------------------

    def _tokens_of(self, msg: dict) -> int:
        return self.count_tokens(str(msg.get("content") or ""))

    def _is_summary(self, msg: dict) -> bool:
        return (msg.get("role") == "system"
                and str(msg.get("content") or "").startswith(SUMMARY_HEADER))

    def fold_history(self, history: list[dict]) -> list[dict]:
        """Fold the oldest messages into one summary when the transcript outgrows its budget.

        Returns ``history`` unchanged while it fits ``max_history_tokens``, so short
        conversations never summarise. Past that, the newest messages worth
        ``keep_tail_tokens`` stay verbatim and everything older is replaced by the rolling
        summary; an earlier summary is folded in with them, so summaries merge instead of
        stacking.

        This never calls the summarizer. It uses the summary the session already has and
        records the folded-away messages on ``_pending`` for :meth:`flush_summary`, so a slow
        model can never sit in the middle of a turn. Until a summary exists the old messages
        are simply dropped, which keeps the budget; their content stays recallable because
        ``observe`` stored each turn as a fact.
        """
        if not history or not self.max_history_tokens:
            return history
        if sum(self._tokens_of(m) for m in history) <= self.max_history_tokens:
            return history

        tail_budget = self.keep_tail_tokens
        if tail_budget is None:
            tail_budget = max(1, self.max_history_tokens // 2)

        tail: list[dict] = []
        used = 0
        for msg in reversed(history):
            if self._is_summary(msg):
                break                                  # a prior summary belongs to the fold
            cost = self._tokens_of(msg)
            if tail and used + cost > tail_budget:
                break
            tail.append(msg)
            used += cost
        tail.reverse()

        older = history[: len(history) - len(tail)]
        if not older:
            return history

        self._pending = older
        if not self._summary:
            return tail
        return [{"role": "system", "content": self._summary}] + tail

    async def flush_summary(self) -> str | None:
        """Summarise whatever the last fold set aside; returns the new summary, or None.

        Safe to call when nothing is pending. A summarizer that raises or returns nothing
        leaves the previous summary in place: losing the wording of old turns is acceptable,
        breaking the conversation is not.
        """
        older, self._pending = self._pending, None
        if not older or self.summarizer is None:
            return None
        try:
            result = self.summarizer(older)
            if hasattr(result, "__await__"):
                result = await result
            text = str(result).strip()
        except Exception:
            return None
        if not text:
            return None
        if not text.startswith(SUMMARY_HEADER):
            text = f"{SUMMARY_HEADER}\n{text}"
        self._summary = text
        return text

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
        folded = self.fold_history(history)
        if self._pending is not None and not self.summarize_in_background:
            if await self.flush_summary():
                folded = self.fold_history(history)   # this turn gets the fresh summary
        out.extend(folded)
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
        if self._pending is not None and self.summarize_in_background:
            await self.flush_summary()                # after the reply, never before it
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
