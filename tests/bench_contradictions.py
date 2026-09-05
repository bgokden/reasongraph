"""Contradiction-resolver benchmark on tests/data/contradictions.jsonl (40 labelled pairs:
20 contradictions of several kinds, 20 complements that must NOT be flagged).

    uv run python tests/bench_contradictions.py --which nli
    LLM_API_KEY=... uv run python tests/bench_contradictions.py --which llm --model qwen/qwen3.8-27b

Reports precision / recall / F1 for the "contradiction" class and ms per pair.
Not a pytest test.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

DATA = Path(__file__).parent / "data" / "contradictions.jsonl"



def load():
    return [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]


def build(which: str, model: str | None):
    if which == "nli":
        from reasongraph import NLIConflictResolver
        return NLIConflictResolver()
    if which == "llm":
        import httpx
        from reasongraph import LLMConflictResolver
        base = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        key = os.environ["LLM_API_KEY"]
        client = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=60)

        def generate(prompt: str) -> str:
            for attempt in range(6):   # free-tier rate limits: back off and retry
                r = client.post("/chat/completions", json={"model": model or "qwen/qwen3.8-27b",
                                                           "temperature": 0, "max_tokens": 200,
                                                           "messages": [{"role": "user", "content": prompt}]})
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("retry-after", 2 * (attempt + 1))))
                    continue
                r.raise_for_status()
                text = r.json()["choices"][0]["message"].get("content") or ""
                return text.split("</think>", 1)[1] if "</think>" in text else text
            raise RuntimeError("rate limited repeatedly")
        return LLMConflictResolver(generate)
    raise SystemExit(which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["nli", "llm"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default="default", choices=["default", "legacy"],
                    help="'legacy' = the older 'do they contradict' phrasing")
    args = ap.parse_args()
    if args.prompt == "legacy":
        from reasongraph import LLMConflictResolver
        LLMConflictResolver.PROMPT = LLMConflictResolver.PROMPT_LEGACY
    rows = load()
    resolver = build(args.which, args.model)
    tp = fp = fn = tn = 0
    misses = []
    t0 = time.perf_counter()
    for r in rows:
        got = bool(resolver.contradictions(r["new"], [r["old"]]))
        if got and r["label"]:
            tp += 1
        elif got and not r["label"]:
            fp += 1; misses.append(("FP", r["kind"], r["new"][:40], r["old"][:40]))
        elif not got and r["label"]:
            fn += 1; misses.append(("FN", r["kind"], r["new"][:40], r["old"][:40]))
        else:
            tn += 1
    ms = 1000 * (time.perf_counter() - t0) / len(rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    print(f"== {args.which} {args.model or ''}: precision {p:.0%}  recall {rc:.0%}  F1 {f1:.0%}  "
          f"(tp {tp} fp {fp} fn {fn} tn {tn})  {ms:.0f} ms/pair")
    for m in misses:
        print("   ", *m)


if __name__ == "__main__":
    main()
