"""Benchmark harness: speed (cold-start / query / RAM) + quality (mixed-domain eval).

Runs ONE pipeline configuration per process so peak-RAM (ru_maxrss high-water
mark, monotonic within a process) is measured cleanly per config. Reuses the
eval's datasets and metric functions so a model swap is judged on the same 32
cases as the baseline.

Not a pytest test (the ``bench_`` name avoids collection).

Usage:
    uv run python tests/bench_pipeline.py                      # baseline (current defaults)
    uv run python tests/bench_pipeline.py --embed all-MiniLM-L6-v2 --label L6
    uv run python tests/bench_pipeline.py --json               # machine-readable row
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasongraph import ReasonGraph
from eval_financial_reasoning import (
    CASES,
    AVAILABLE_DATASETS,
    chain_completeness,
    precision_at_k,
    domain_accuracy,
)


def _peak_ram_mb() -> float:
    """Peak resident set size for this process, in MB (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench(embed_model: str | None, rerank_model: str | None, label: str) -> dict:
    # Cold start: constructing the graph loads the embedder eagerly.
    t0 = time.perf_counter()
    graph = ReasonGraph(embed_model=embed_model, rerank_model=rerank_model)
    graph.initialize_sync()
    for ds in AVAILABLE_DATASETS:
        graph.load_dataset_sync(ds)
    embed_load_s = time.perf_counter() - t0

    # First query triggers the lazy reranker load (the rest of the cold path).
    t1 = time.perf_counter()
    _ = graph.query_sync(
        CASES[0].agent_thought,
        search_mode=CASES[0].search_mode,
        top_k=CASES[0].top_k,
        hops=CASES[0].hops,
    )
    first_query_s = time.perf_counter() - t1

    totals = {"completeness": 0.0, "p5": 0.0, "domain": 0.0}
    passed = 0
    warm_ms: list[float] = []
    for case in CASES:
        q0 = time.perf_counter()
        results = graph.query_sync(
            case.agent_thought,
            search_mode=case.search_mode,
            top_k=case.top_k,
            hops=case.hops,
        )
        warm_ms.append((time.perf_counter() - q0) * 1000.0)

        comp = chain_completeness(results, case.expected_chain)
        totals["completeness"] += comp
        totals["p5"] += precision_at_k(results, case.expected_chain, 5)
        totals["domain"] += domain_accuracy(results, case.domain)
        if comp >= 0.5:
            passed += 1

    graph.close_sync()
    n = len(CASES)
    return {
        "label": label,
        "embed_model": embed_model or "all-MiniLM-L12-v2 (default)",
        "rerank_model": rerank_model or "ms-marco-MiniLM-L-6-v2 (default)",
        "embed_load_s": round(embed_load_s, 2),
        "first_query_s": round(first_query_s, 2),
        "warm_query_ms": round(statistics.median(warm_ms), 1),
        "peak_ram_mb": round(_peak_ram_mb(), 0),
        "chain_completeness": round(totals["completeness"] / n, 3),
        "precision_at_5": round(totals["p5"] / n, 3),
        "domain_accuracy": round(totals["domain"] / n, 3),
        "pass_rate": f"{passed}/{n}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default=None, help="embedder model name (default: current)")
    ap.add_argument("--rerank", default=None, help="reranker model name (default: current)")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--json", action="store_true", help="print one JSON row only")
    args = ap.parse_args()

    row = bench(args.embed, args.rerank, args.label)

    if args.json:
        print(json.dumps(row))
        return

    print(f"\n{'=' * 68}")
    print(f"  Pipeline benchmark: {row['label']}")
    print(f"{'=' * 68}")
    print(f"  embedder            : {row['embed_model']}")
    print(f"  reranker            : {row['rerank_model']}")
    print(f"  --- speed ---")
    print(f"  embedder cold load  : {row['embed_load_s']:>7.2f} s")
    print(f"  first query (+rerank load): {row['first_query_s']:>7.2f} s")
    print(f"  warm query (median) : {row['warm_query_ms']:>7.1f} ms")
    print(f"  peak RAM            : {row['peak_ram_mb']:>7.0f} MB")
    print(f"  --- quality (32 cases) ---")
    print(f"  chain completeness  : {row['chain_completeness']:>7.1%}")
    print(f"  precision@5         : {row['precision_at_5']:>7.1%}")
    print(f"  domain accuracy     : {row['domain_accuracy']:>7.1%}")
    print(f"  pass rate           : {row['pass_rate']:>7s}")
    print()


if __name__ == "__main__":
    main()
