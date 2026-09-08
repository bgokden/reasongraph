"""End-to-end evaluation of automatic cause/effect extraction.

eval_financial_reasoning.py loads HAND-CURATED datasets, so it bypasses
extraction. This one ingests the same raw domain sentences through the full
causal-on-by-default pipeline -- entity extraction + the hybrid causal extractor
-- and runs the same 32 reasoning cases on the AUTO-BUILT graph. It measures
whether auto-extracted entities and cause->effect edges support the same
multi-hop reasoning as the hand-built graph.

Run:  uv run python tests/eval_causal_extraction.py
"""

from __future__ import annotations

import json
import os
import time

from reasongraph import ReasonGraph
from reasongraph.datasets import AVAILABLE_DATASETS
from eval_financial_reasoning import (
    CASES,
    DOMAIN_TEXTS,
    chain_completeness,
    recall_at_k,
    precision_at_k,
    domain_accuracy,
)


def _corpus() -> list[str]:
    """All raw domain sentences (deduped, stable order)."""
    return sorted({t for texts in DOMAIN_TEXTS.values() for t in texts})


def build_handcrafted() -> ReasonGraph:
    g = ReasonGraph()
    g.initialize_sync()
    for ds in AVAILABLE_DATASETS:
        g.load_dataset_sync(ds)
    return g


def _embed_model():
    """Mirror the deployment: REASONGRAPH_EMBED_MODEL (a ``fastembed:`` prefix is dropped,
    the sentence-transformers name is used); unset = library default (English MiniLM)."""
    name = os.environ.get("REASONGRAPH_EMBED_MODEL") or None
    if name and name.startswith("fastembed:"):
        name = name.split(":", 1)[1]
    return name


def build_autoextracted() -> tuple[ReasonGraph, float]:
    g = ReasonGraph(embed_model=_embed_model())
    g.initialize_sync()
    t0 = time.perf_counter()
    # Default pipeline: gliner_small entities + hybrid causal (cue + relex), causal on.
    g.add_texts_sync(_corpus())
    return g, time.perf_counter() - t0


def _graph_stats(g: ReasonGraph) -> dict:
    nodes = g._run(g.get_all_nodes())
    edges = g._run(g.get_all_edges())
    return {
        "text": sum(1 for n in nodes if n.type == "text"),
        "entity": sum(1 for n in nodes if n.type == "entity"),
        "edges": len(edges),
        "causal_edges": sum(1 for e in edges if e.label == "causes"),
    }


def evaluate(graph: ReasonGraph) -> list[dict]:
    rows = []
    for case in CASES:
        results = graph.query_sync(
            case.agent_thought, search_mode=case.search_mode,
            top_k=case.top_k, hops=case.hops,
        )
        rows.append({
            "name": case.name,
            "domain": case.domain,
            "comp": chain_completeness(results, case.expected_chain),
            "r5": recall_at_k(results, case.expected_chain, 5),
            "p5": precision_at_k(results, case.expected_chain, 5),
            "da": domain_accuracy(results, case.domain),
        })
    return rows


EXTRA_CASES_PATH = os.path.join(os.path.dirname(__file__), "data", "reasoning_cases_extra.jsonl")


def load_extra_cases() -> list[dict]:
    """32 reviewed extra cases (logistics, legal, education, sport; 8 non-English), each
    with its own raw corpus. Auto-extraction only: there is no hand-built graph for them."""
    with open(EXTRA_CASES_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_autoextracted_extra(cases: list[dict]) -> tuple[ReasonGraph, float]:
    g = ReasonGraph(embed_model=_embed_model())
    g.initialize_sync()
    corpus = list(dict.fromkeys(s for c in cases for s in c["corpus"]))
    t0 = time.perf_counter()
    g.add_texts_sync(corpus)
    return g, time.perf_counter() - t0


def evaluate_extra(graph: ReasonGraph, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        top_k = int(os.environ.get("EVAL_TOP_K") or case.get("top_k", 5))   # window sweep (X1 follow-up)
        results = graph.query_sync(case["agent_thought"], search_mode=case.get("search_mode", "hybrid"),
                                   top_k=top_k, hops=case.get("hops", 4))
        rows.append({"name": case["name"], "domain": case["domain"], "lang": case.get("lang", "en"),
                     "comp": chain_completeness(results, case["expected_chain"]),
                     "r5": recall_at_k(results, case["expected_chain"], 5),
                     "p5": precision_at_k(results, case["expected_chain"], 5)})
    return rows


def _avg(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def _domain_table(hand, auto, domains):
    print(f"  {'Domain':<16s} {'n':>3s} {'Chain(hand)':>11s} {'Chain(auto)':>11s} {'Δ':>6s}")
    print(f"  {'-'*16} {'-'*3} {'-'*11} {'-'*11} {'-'*6}")
    for d in domains:
        h = [r for r in hand if r["domain"] == d]
        a = [r for r in auto if r["domain"] == d]
        hc, ac = _avg(h, "comp"), _avg(a, "comp")
        print(f"  {d:<16s} {len(h):>3d} {hc:>10.0%} {ac:>10.0%} {ac-hc:>+6.0%}")


def main():
    print("Building hand-crafted graph (load_dataset)...")
    hand = build_handcrafted()
    hs = _graph_stats(hand)

    print("Building auto-extracted graph (add_texts, causal on)...")
    auto, ingest_s = build_autoextracted()
    as_ = _graph_stats(auto)

    print()
    print(f"{'=' * 78}")
    print("  Graph construction")
    print(f"{'=' * 78}")
    print(f"  {'':<14s} {'text':>6s} {'entity':>7s} {'edges':>7s} {'causal-edges':>13s}")
    print(f"  {'hand-built':<14s} {hs['text']:>6d} {hs['entity']:>7d} {hs['edges']:>7d} {hs['causal_edges']:>13d}")
    print(f"  {'auto-extracted':<14s} {as_['text']:>6d} {as_['entity']:>7d} {as_['edges']:>7d} {as_['causal_edges']:>13d}")
    print(f"  auto-extraction ingested {as_['text']} sentences in {ingest_s:.1f}s "
          f"({1000*ingest_s/max(as_['text'],1):.0f} ms/sentence)")
    print()

    hand_rows = evaluate(hand)
    auto_rows = evaluate(auto)
    hand.close_sync()
    auto.close_sync()

    domains = sorted({r["domain"] for r in hand_rows})

    print(f"{'=' * 78}")
    print("  Chain completeness by domain: hand-built vs auto-extracted")
    print(f"{'=' * 78}")
    _domain_table(hand_rows, auto_rows, domains)
    print()

    print(f"{'=' * 78}")
    print("  Overall (32 cases)")
    print(f"{'=' * 78}")
    for label, rows in (("hand-built", hand_rows), ("auto-extracted", auto_rows)):
        passes = sum(1 for r in rows if r["comp"] >= 0.5)
        print(f"  {label:<16s}  Chain {_avg(rows,'comp'):.0%}   R@5 {_avg(rows,'r5'):.0%}"
              f"   P@5 {_avg(rows,'p5'):.0%}   Domain {_avg(rows,'da'):.0%}"
              f"   Pass(>=50%) {passes}/{len(rows)}")
    print()

    print(f"{'=' * 78}")
    print("  Causal-domain focus (per case)")
    print(f"{'=' * 78}")
    print(f"  {'case':<40s} {'hand':>6s} {'auto':>6s}")
    print(f"  {'-'*40} {'-'*6} {'-'*6}")
    hand_by = {r["name"]: r for r in hand_rows}
    for r in auto_rows:
        if r["domain"] != "causal":
            continue
        print(f"  {r['name'][:40]:<40s} {hand_by[r['name']]['comp']:>6.0%} {r['comp']:>6.0%}")
    print(f"{'=' * 78}")


def run_extra() -> None:
    cases = load_extra_cases()
    print(f"\nExtra cases ({len(cases)}, auto-extracted only; reviewed 2026-09-08)")
    g, secs = build_autoextracted_extra(cases)
    rows = evaluate_extra(g, cases)
    print(f"  ingest {secs:.1f}s for {_graph_stats(g)['text']} sentences")
    print(f"  {'domain':<12s} {'n':>3s}  Chain   R@5   P@5")
    for key in ("domain", "lang"):
        for val in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == val]
            print(f"  {val:<12s} {len(sub):>3d}  {_avg(sub,'comp'):>5.0%}  {_avg(sub,'r5'):>4.0%}  {_avg(sub,'p5'):>4.0%}")
    print(f"  {'OVERALL':<12s} {len(rows):>3d}  {_avg(rows,'comp'):>5.0%}  {_avg(rows,'r5'):>4.0%}  {_avg(rows,'p5'):>4.0%}")


if __name__ == "__main__":
    main()
    run_extra()
