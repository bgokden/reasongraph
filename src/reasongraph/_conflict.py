"""Contradiction detection for write-time conflict resolution.

A ``ConflictResolver`` decides which existing facts a new fact contradicts. The
graph uses it to soft-supersede outdated facts (mark them so they drop out of
default recall while staying auditable) instead of accumulating contradictions.

The core stays model-free: pass any object with a ``contradictions`` method, or the
shipped ``NLIConflictResolver`` (a natural-language-inference cross-encoder, no text
generation). Cosine similarity cannot tell contradiction from complement -- "lives
in Berlin" vs "lives in Munich" (contradiction) and vs "works in Munich"
(complement) are equally similar -- so a semantic judge is required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ConflictResolver(Protocol):
    """Anything that decides which existing facts a new fact contradicts."""

    def contradictions(self, new_text: str, candidates: list[str]) -> list[str]:
        """Return the subset of ``candidates`` that ``new_text`` contradicts."""
        ...


class NLIConflictResolver:
    """Detect contradictions with an NLI cross-encoder (no text generation).

    For each candidate the model scores the pair (premise=candidate,
    hypothesis=new_text); a candidate is contradicted when the contradiction
    probability is at least ``threshold``. The model is lazy-loaded on first use.
    """

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
    # id order emitted by the nli-deberta cross-encoders.
    _CONTRADICTION = 0

    def __init__(self, model: str | object | None = None, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._model_name = model or self.DEFAULT_MODEL
        self._model = None
        self.threshold = threshold

    def _load(self) -> None:
        if self._model is not None:
            return
        if isinstance(self._model_name, str):
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        else:
            # Any object exposing predict(pairs) -> (N, 3) logits.
            self._model = self._model_name

    def contradictions(self, new_text: str, candidates: list[str]) -> list[str]:
        if not candidates:
            return []
        self._load()
        pairs = [(candidate, new_text) for candidate in candidates]
        logits = np.asarray(self._model.predict(pairs), dtype=float)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        # softmax over the 3 classes so ``threshold`` is a probability.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        contradiction = probs[:, self._CONTRADICTION]
        return [c for c, p in zip(candidates, contradiction) if p >= self.threshold]


class LLMConflictResolver:
    """Detect contradictions with a bring-your-own LLM.

    More accurate than the NLI cross-encoder on complements (an NLI model tends to
    call "works in Munich" vs "lives in Berlin" a contradiction; an instructed LLM
    can be told that facts which can both hold are not contradictions). ``generate``
    is any callable(prompt: str) -> str -- the same pattern as the synthesizers, so
    the core stays model-free.
    """

    PROMPT = (
        "You maintain a memory of facts. A new fact has arrived.\n"
        "New fact: {new}\n"
        "Existing fact: {old}\n"
        "Does the new fact CONTRADICT the existing one -- can they NOT both be true, "
        "so the existing fact is now outdated? Facts that can both hold "
        "(complementary) are NOT contradictions. Answer only 'yes' or 'no'."
    )

    def __init__(self, generate) -> None:
        if not callable(generate):
            raise TypeError("generate must be a callable(prompt) -> str")
        self._generate = generate

    def contradictions(self, new_text: str, candidates: list[str]) -> list[str]:
        out = []
        for candidate in candidates:
            answer = str(self._generate(
                self.PROMPT.format(new=new_text, old=candidate)
            )).strip().lower()
            if answer.startswith("yes"):
                out.append(candidate)
        return out
