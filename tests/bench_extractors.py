"""Extractor benchmark: speed (load / per-call) + reasoning-bridge recall.

The mixed-domain eval never calls the extractor (it loads pre-built datasets),
so extractor quality is measured here instead: does a faster/smaller extractor
still catch the entities that actually *bridge* documents in the cross-source
demos? Missing a bridge (e.g. "Arizona") breaks the multi-hop chain.

Runs ONE extractor per process (clean peak RAM). Not a pytest test.

Usage:
    uv run python tests/bench_extractors.py --which gliner2
    uv run python tests/bench_extractors.py --which place-onnx
    uv run python tests/bench_extractors.py --which gliner-multi-onnx
"""

from __future__ import annotations

import argparse
import os
import resource
import statistics
import sys
import time

# (sentence, reasoning-critical entities that must be caught to bridge documents)
CORPUS = [
    ("TSMC announced plans to build a semiconductor plant in Phoenix, Arizona.",
     {"TSMC", "Phoenix", "Arizona"}),
    ("TSMC signed a supply agreement with Apple to make M-series chips in Arizona.",
     {"TSMC", "Apple", "Arizona"}),
    ("Intel paused expansion of its Chandler, Arizona chip plant.",
     {"Intel", "Chandler", "Arizona"}),
    ("Arizona declared a water emergency after Lake Mead dropped to record lows.",
     {"Arizona", "Lake Mead"}),
    ("Apple warned that component shortages could impact iPhone production.",
     {"Apple"}),
    ("The Federal Reserve raised interest rates to cool inflation.",
     {"Federal Reserve"}),
]

PLACE_MODEL_DIR = os.environ.get(
    "REASONGRAPH_PLACE_MODEL_DIR", "/home/berk/repos/trainner/archive/web_release"
)


def _peak_ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build(which: str):
    if which == "gliner2":
        from reasongraph._extraction import GLiNER2Extractor
        return GLiNER2Extractor()
    if which == "chat":
        from reasongraph._extraction import ChatExtractor
        return ChatExtractor()
    if which == "ner":
        from reasongraph._extraction import NERExtractor
        return NERExtractor()
    if which == "place-onnx":
        from reasongraph._extraction import OnnxTokenClassifierExtractor
        return OnnxTokenClassifierExtractor(
            PLACE_MODEL_DIR, onnx_path=os.path.join(PLACE_MODEL_DIR, "onnx", "model_fp16.onnx")
        )
    if which in ("gliner-multi-onnx", "gliner-multi-torch"):
        from reasongraph._extraction import GlinerExtractor
        return GlinerExtractor(
            "urchade/gliner_multi-v2.1",
            labels=["person", "organization", "location", "event"],
            onnx=(which == "gliner-multi-onnx"),
        )
    raise SystemExit(f"unknown extractor: {which}")


def _found(gold: str, extracted: list[str]) -> bool:
    g = gold.lower()
    return any(g in e.lower() or e.lower() in g for e in extracted)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    ext = build(args.which)
    ext("warmup text to force model load")
    load_s = time.perf_counter() - t0

    per_call_ms: list[float] = []
    gold_total = 0
    gold_found = 0
    distinct: set[str] = set()
    for sentence, gold in CORPUS:
        t1 = time.perf_counter()
        ents = ext(sentence)
        per_call_ms.append((time.perf_counter() - t1) * 1000.0)
        distinct.update(ents)
        for g in gold:
            gold_total += 1
            if _found(g, ents):
                gold_found += 1

    recall = gold_found / gold_total if gold_total else 0.0
    print(f"\n{'=' * 60}")
    print(f"  Extractor: {args.which}")
    print(f"{'=' * 60}")
    print(f"  load + warmup   : {load_s:>7.2f} s")
    print(f"  per-call median : {statistics.median(per_call_ms):>7.1f} ms")
    print(f"  peak RAM        : {_peak_ram_mb():>7.0f} MB")
    print(f"  bridge recall   : {recall:>7.1%}  ({gold_found}/{gold_total})")
    print(f"  distinct ents   : {len(distinct):>7d}")
    print()


if __name__ == "__main__":
    main()
