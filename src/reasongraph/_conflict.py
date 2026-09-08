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

    # Asks whether the OLD fact is no longer true given the NEW one. On the 40-pair
    # benchmark (tests/bench_contradictions.py) this phrasing scores F1 0.97 with
    # qwen3.8-27b (precision 1.0) vs 0.86 for the "do they contradict" phrasing,
    # which missed state changes ("cancelled", "paused", "left").
    PROMPT = (
        "You maintain a memory of facts about the current state of the world.\n"
        "Existing fact: {old}\n"
        "New fact (more recent): {new}\n"
        "Given the new fact, is the existing fact NO LONGER TRUE as a statement about the "
        "present (it has been replaced, reversed, ended, or its value changed)? Facts about "
        "different things, or past events that still happened, do not count. "
        "Answer only 'yes' or 'no'."
    )
    PROMPT_LEGACY = (
        "You maintain a memory of facts. A new fact has arrived.\n"
        "New fact: {new}\n"
        "Existing fact: {old}\n"
        "Does the new fact CONTRADICT the existing one -- can they NOT both be true, "
        "so the existing fact is now outdated? Facts that can both hold "
        "(complementary) are NOT contradictions. Answer only 'yes' or 'no'."
    )

    def __init__(self, generate, prompt: str | None = None) -> None:
        if not callable(generate):
            raise TypeError("generate must be a callable(prompt) -> str")
        self._generate = generate
        if prompt is not None:
            self.PROMPT = prompt

    def contradictions(self, new_text: str, candidates: list[str]) -> list[str]:
        out = []
        for candidate in candidates:
            answer = str(self._generate(
                self.PROMPT.format(new=new_text, old=candidate)
            )).strip().lower()
            if answer.startswith("yes"):
                out.append(candidate)
        return out


class FineTunedConflictResolver:
    """Conflict detection with our own fine-tuned small model served by ``llama.cpp``.

    The model (the ``reasongraph-extractor`` adapters trained in the lab) was taught one
    instruction per task; the conflict one is exactly::

        [conflict] existing: <existing fact>\nnew: <new fact>  ->  {"conflict": true|false}

    This resolver sends that prompt, one candidate at a time, to a ``llama-server``
    (``<endpoint>/completion``) with a grammar that only admits the JSON answer, so the
    reply is a strict yes/no and never free text. It keeps the write path on our own
    hardware: no external LLM provider sees the facts.

    ``fail_open`` (default True) means a down or slow model yields "no conflicts" instead
    of blocking a push; the miss is logged by the caller's normal path. Pass ``post`` to
    substitute the HTTP call (tests).
    """

    GRAMMAR = (
        'root ::= "{" ws "\\"conflict\\"" ws ":" ws ("true" | "false") ws "}"\n'
        "ws ::= [ \\n\\t]*\n"
    )

    def __init__(self, endpoint: str = "http://127.0.0.1:8080", *, timeout: float = 20.0,
                 max_candidates: int = 10, fail_open: bool = True, post=None,
                 prefilter: "str | dict | None" = None, prefilter_threshold: float | None = None,
                 prefilter_encoder=None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_candidates = max_candidates
        self.fail_open = fail_open
        self._post = post or self._http_post
        # Optional first stage: a logistic regression on sentence-embedding pair features
        # ``[a, b, |a-b|, a*b]`` (a = existing fact, b = new fact) trained for recall
        # (lab task H4: recall 0.99 at threshold 0.2, cutting ~65% of candidate pairs). Only
        # pairs it passes reach the model. ``prefilter`` is a .joblib path, an ``hf://`` ref,
        # or the dict the trainer saves ({embed_model, clf, threshold, features}).
        self.prefilter = prefilter
        self.prefilter_threshold = prefilter_threshold
        self._prefilter_encoder = prefilter_encoder
        self._prefilter_clf = None

    def _load_prefilter(self) -> None:
        if self._prefilter_clf is not None or self.prefilter is None:
            return
        payload = self.prefilter
        if isinstance(payload, str):
            import joblib
            path = payload
            if path.startswith("hf://"):
                from huggingface_hub import hf_hub_download
                parts = path[5:].split("/")
                path = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))
            payload = joblib.load(path)
        self._prefilter_clf = payload["clf"]
        if self.prefilter_threshold is None:
            self.prefilter_threshold = float(payload.get("threshold", 0.2))
        if self._prefilter_encoder is None:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(payload["embed_model"])
            self._prefilter_encoder = lambda texts: st.encode(
                texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def prefilter_probs(self, new_text: str, candidates: list[str]) -> "list[float] | None":
        """P(conflict) per candidate from the pre-filter, or None when none is set."""
        if self.prefilter is None:
            return None
        self._load_prefilter()
        if not candidates:
            return []
        import numpy as np
        embs = np.asarray(self._prefilter_encoder(candidates + [new_text]))
        a, b = embs[:-1], np.repeat(embs[-1:], len(candidates), axis=0)
        feats = np.concatenate([a, b, np.abs(a - b), a * b], axis=1)
        probs = self._prefilter_clf.predict_proba(feats)
        classes = list(getattr(self._prefilter_clf, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else len(classes) - 1
        return [float(p[idx]) for p in probs]

    @staticmethod
    def prompt(existing: str, new: str) -> str:
        return f"[conflict] existing: {existing}\nnew: {new}"

    def _http_post(self, url: str, body: dict) -> dict:
        import json
        import urllib.request
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_conflict(self, existing: str, new: str) -> bool | None:
        """True/False from the model, or None when the model could not answer
        (endpoint down, timeout, or a reply the grammar should have prevented).
        Only transport and reply-format errors fail open; anything else propagates."""
        import json
        import logging
        import urllib.error
        body = {"prompt": self.prompt(existing, new), "n_predict": 16, "temperature": 0,
                "grammar": self.GRAMMAR, "cache_prompt": True}
        try:
            out = self._post(self.endpoint + "/completion", body)
            text = out.get("content") if isinstance(out, dict) else None
            if text is None and isinstance(out, dict) and out.get("choices"):
                text = out["choices"][0].get("text")
            return bool(json.loads(text.strip())["conflict"])
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                KeyError, TypeError, AttributeError, ValueError) as exc:
            if not self.fail_open:
                raise
            logging.getLogger(__name__).warning(
                "FineTunedConflictResolver: no answer from %s (%s: %s); treating as no conflict",
                self.endpoint, type(exc).__name__, exc)
            return None

    def contradictions(self, new_text: str, candidates: list[str]) -> list[str]:
        cands = [c for c in candidates[: self.max_candidates] if c != new_text]
        probs = self.prefilter_probs(new_text, cands)
        if probs is not None:
            cands = [c for c, p in zip(cands, probs) if p >= self.prefilter_threshold]
        out: list[str] = []
        for cand in cands:
            if self.is_conflict(cand, new_text):
                out.append(cand)
        return out
