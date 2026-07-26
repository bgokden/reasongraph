"""Unit tests for NLIConflictResolver logic, with an injected fake NLI model.

The real cross-encoder is exercised manually; here we inject an object exposing
``predict(pairs) -> (N, 3) logits`` so the softmax + threshold + filtering logic is
tested deterministically without downloading a model.
"""

import numpy as np

from reasongraph import NLIConflictResolver, LLMConflictResolver


class _FakePredict:
    """Emit contradiction logits when premise and hypothesis share a first word."""

    def predict(self, pairs):
        rows = []
        for premise, hypothesis in pairs:
            same_subject = premise.split()[0].lower() == hypothesis.split()[0].lower()
            rows.append([5.0, 0.0, 0.0] if same_subject else [0.0, 0.0, 5.0])
        return np.array(rows)


def test_nli_resolver_flags_contradictions():
    resolver = NLIConflictResolver(model=_FakePredict())
    out = resolver.contradictions(
        "Alice lives in Berlin.",
        ["Alice lives in Munich.", "Bob plays tennis.", "Alice was born in 1990."],
    )
    assert out == ["Alice lives in Munich.", "Alice was born in 1990."]


def test_nli_resolver_empty_candidates():
    assert NLIConflictResolver(model=_FakePredict()).contradictions("x", []) == []


def test_nli_resolver_threshold_gates():
    # A model that always emits a mild contradiction lead: below 0.9 it should not fire.
    class _Mild:
        def predict(self, pairs):
            return np.array([[1.0, 0.0, 0.5] for _ in pairs])

    lenient = NLIConflictResolver(model=_Mild(), threshold=0.3)
    strict = NLIConflictResolver(model=_Mild(), threshold=0.9)
    assert lenient.contradictions("a", ["b"]) == ["b"]
    assert strict.contradictions("a", ["b"]) == []


def test_llm_resolver_parses_yes_no():
    # A fake LLM that says a fact is a contradiction iff both mention "Munich".
    def fake_generate(prompt: str) -> str:
        return "Yes, they contradict." if "Munich" in prompt else "No."

    resolver = LLMConflictResolver(fake_generate)
    out = resolver.contradictions(
        "Alice relocated abroad.",
        ["Alice lives in Munich.", "Bob lives in Berlin."],
    )
    assert out == ["Alice lives in Munich."]
