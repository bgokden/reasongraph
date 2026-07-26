"""Pluggable synthesizers: rephrase retrieved facts into logical free text.

A synthesizer turns the structured output of ``ReasonGraph.discover`` /
``query`` (facts, their scopes, and the connection paths that bridge sessions)
into a natural-language answer. The graph core stays model-free -- bring one of
these, or any ``callable(query, context) -> str``.

Three options, cheapest first:

- ``TemplateSynthesizer``  -- deterministic, no model. A zero-dependency default.
- ``PromptSynthesizer``    -- bring your own ``generate(prompt) -> str`` (any LLM
  API or local model; sync or async). Builds the prompt; you supply the model.
- ``TransformersSynthesizer`` -- a concrete local small-LLM adapter over a
  transformers text-generation pipeline (reuses the installed torch/transformers
  stack; no extra heavy dependency).

``context`` is the list returned by ``discover`` (dicts with ``content``,
``scopes``, ``cross_session``, ``path``) or, with ``use_discover=False``, the
lighter ``query`` shape (``content`` + a single-step ``path``). All three read
only ``content``, ``cross_session``, and ``path`` so they work with either.
"""

from __future__ import annotations

from typing import Any, Callable


def _unique_facts(context: list[dict], limit: int) -> list[str]:
    """First ``limit`` distinct fact contents, in the order given."""
    facts: list[str] = []
    seen: set[str] = set()
    for item in context:
        content = item.get("content")
        if not content or content in seen:
            continue
        seen.add(content)
        facts.append(content)
        if len(facts) >= limit:
            break
    return facts


def _bridge_entities(context: list[dict]) -> list[str]:
    """Entities that bridge cross-session discoveries, first-seen order."""
    bridges: list[str] = []
    seen: set[str] = set()
    for item in context:
        if not item.get("cross_session"):
            continue
        for step in item.get("path", []):
            entity = step.get("entity") if isinstance(step, dict) else None
            if entity and entity not in seen:
                seen.add(entity)
                bridges.append(entity)
    return bridges


class TemplateSynthesizer:
    """Deterministic, model-free synthesizer.

    Rephrases retrieved facts into logical free text and names the entities that
    bridge sessions. No LLM, no network -- a safe default or fallback.
    """

    def __init__(self, max_facts: int = 12) -> None:
        self.max_facts = max_facts

    def synthesize(self, query: str, context: list[dict]) -> str:
        if not context:
            return f"No stored facts connect to '{query}'."
        lines = [f"On '{query}', the memory graph connects these facts:"]
        cross = {
            item["content"]
            for item in context
            if item.get("cross_session") and item.get("content")
        }
        for content in _unique_facts(context, self.max_facts):
            origin = " (from another session)" if content in cross else ""
            lines.append(f"- {content}{origin}")
        bridges = _bridge_entities(context)
        if bridges:
            lines.append(f"The link across sessions runs through: {', '.join(bridges)}.")
        return "\n".join(lines)

    def __call__(self, query: str, context: list[dict]) -> str:
        return self.synthesize(query, context)


class PromptSynthesizer:
    """Rephrase retrieved facts with any text-generation model you bring.

    Supply ``generate``: a ``callable(prompt: str) -> str`` (sync or async). It
    can wrap an OpenAI/Anthropic client, a transformers pipeline, a local GGUF
    model -- anything that turns a prompt into text. This class builds the prompt
    from the query, the retrieved facts, and the connection paths (so the model
    can explain *how* facts connect), then returns the completion. When
    ``generate`` is async, ``synthesize`` returns the coroutine and
    ``ReasonGraph.answer`` awaits it.
    """

    DEFAULT_INSTRUCTION = (
        "You are a reasoning assistant. Using only the facts and connections "
        "below, answer the question in clear, logical prose. Explain how the "
        "facts connect to each other, and do not add anything the facts do not "
        "support. If the facts are insufficient, say so."
    )

    def __init__(
        self,
        generate: Callable[[str], Any],
        *,
        instruction: str | None = None,
        max_facts: int = 12,
    ) -> None:
        if not callable(generate):
            raise TypeError("generate must be a callable(prompt: str) -> str")
        self._generate = generate
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.max_facts = max_facts

    def build_prompt(self, query: str, context: list[dict]) -> str:
        facts = _unique_facts(context, self.max_facts)
        fact_block = "\n".join(f"- {f}" for f in facts) if facts else "- (no facts found)"

        link_lines: list[str] = []
        for item in context:
            if not item.get("cross_session"):
                continue
            entities = [
                step["entity"]
                for step in item.get("path", [])
                if isinstance(step, dict) and "entity" in step
            ]
            if entities:
                bridge = ", ".join(dict.fromkeys(entities))
                link_lines.append(f'- "{item["content"]}" connects via {bridge}')

        parts = [self.instruction, "", f"Question: {query}", "", "Facts:", fact_block]
        if link_lines:
            parts += ["", "Cross-session connections:", "\n".join(link_lines)]
        parts += ["", "Answer:"]
        return "\n".join(parts)

    def synthesize(self, query: str, context: list[dict]) -> Any:
        return self._generate(self.build_prompt(query, context))

    def __call__(self, query: str, context: list[dict]) -> Any:
        return self.synthesize(query, context)


class TransformersSynthesizer(PromptSynthesizer):
    """Local small-LLM synthesizer over a transformers text-generation pipeline.

    Loads a small instruct model once (default ``Qwen/Qwen2.5-0.5B-Instruct``)
    and reuses it -- a fully local, offline "rephrase into logical free text"
    option that leans on the torch/transformers stack already pulled in by
    ``sentence-transformers`` (no extra heavy dependency). Pass any instruct
    model name or a preconstructed pipeline.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(
        self,
        model: Any = None,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        instruction: str | None = None,
        max_facts: int = 12,
        **pipeline_kwargs: Any,
    ) -> None:
        try:
            from transformers import pipeline, Pipeline
        except ImportError as e:  # pragma: no cover - transformers is a base dep
            raise ImportError(
                "transformers not installed. It ships with sentence-transformers; "
                "reinstall reasongraph to get it."
            ) from e

        if isinstance(model, Pipeline):
            self._pipe = model
        else:
            self._pipe = pipeline(
                "text-generation", model=model or self.DEFAULT_MODEL, **pipeline_kwargs
            )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        super().__init__(self._complete, instruction=instruction, max_facts=max_facts)

    def _complete(self, prompt: str) -> str:
        output = self._pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        return self._extract_reply(output)

    @staticmethod
    def _extract_reply(output: Any) -> str:
        """Pull the assistant text out of a transformers pipeline result.

        The chat text-generation pipeline returns
        ``[{"generated_text": [ ...messages..., {"role": "assistant", "content": ...}]}]``;
        older/plain pipelines return a bare string in ``generated_text``.
        """
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            return str(generated[-1]["content"]).strip()
        return str(generated).strip()
