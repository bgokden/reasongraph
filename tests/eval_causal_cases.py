"""Run the E2 causal-specific eval on the draft case set.

Unlike the 32-case rg eval (whose reference graph has zero causal edges, so it
measures entity bridging), every case here has a >=2-hop cause->effect chain and a
distractor that shares an entity but is off-chain. Passing needs the extractor to
produce cause/effect spans that CHAIN across facts, so the graph can walk them.

For each case we build a fresh ReasonGraph with the chosen causal extractor, push
each session's facts (resolve_conflicts=False), then score:
  - Chain: fraction of gold_chain facts that appear in the discovered path.
  - ChainOrd: fraction recovered as an in-order subsequence (direction-aware).
  - R@5: fraction of gold_chain facts among the top-5 discover results.
  - Answer: gold_answer found in a discovered fact or a causal_chain hop span.
  - CausalChain: causal_chain(root -> final effect) returns a directed chain.

The extractor is parameterized so later tasks compare candidates:
  --causal-model / --gate-threshold        the span-pointer model + its baked gate
  --embed-gate <joblib> / --embed-gate-threshold   decoupled embedding gate (G1):
      run the pointer gate OFF (--gate-threshold 1.0) and let this classifier decide.

Examples:
  # production v1 (baked-in gate)
  uv run python eval/run_causal_eval.py \
      --causal-model berk/causal-span-pointer-mdeberta --gate-threshold 0.5 \
      --label v1 --out results/e2_v1.json
  # candidate: R2 spans gate-OFF + multilingual MLP embedding gate @0.9
  uv run python eval/run_causal_eval.py \
      --causal-model Berk/causal-span-pointer-v2 --gate-threshold 1.0 \
      --embed-gate causal/models/gate_paraphrase-multilingual-MiniLM-L12-v2_mlp.joblib \
      --embed-gate-threshold 0.9 --label R2+embed@0.9 --out results/e2_cand.json
"""

import argparse
from pathlib import Path
import collections
import json
import os
import time

DEFAULT_CASES = str(Path(__file__).parent / "data" / "causal_cases.jsonl")


class EmbedGatedExtractor:
    """Wrap a causal extractor with a decoupled embedding gate (G1).

    Embeds the sentence with the joblib's sentence-transformer and returns []
    when P(causal) is below the threshold; otherwise delegates to the inner
    span-pointer extractor. Decoupling keeps the span benchmark untouched.
    """

    def __init__(self, inner, gate_path, threshold):
        import joblib

        bundle = joblib.load(gate_path)
        self.inner = inner
        self.embed_model_name = bundle["embed_model"]
        self.clf = bundle["clf"]
        self.threshold = threshold
        self._causal_col = list(self.clf.classes_).index(1)
        self._st = None

    def _embed(self, text):
        if self._st is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._st = SentenceTransformer(self.embed_model_name, device=device)
        return self._st.encode([text], normalize_embeddings=True)

    def relations_for(self, text):
        proba = self.clf.predict_proba(self._embed(text))[0][self._causal_col]
        if float(proba) < self.threshold:
            return []
        return self.inner.relations_for(text)

    def extract_causal(self, texts):
        results = []
        for text in texts:
            relations = self.relations_for(text)
            results.append({"text": text, "causal": len(relations) > 0, "relations": relations})
        return results

    __call__ = extract_causal


class LLMCausalExtractor:
    """Wrap the multi-task LLM (L1) as a causal_extractor.

    Prompts `[causal] <sentence>`, greedy-decodes the JSON completion the model was
    trained to produce, and returns its cause/effect pairs. Same contract as
    CausalPointerExtractor so the runner can compare the LLM head to head.
    """

    def __init__(self, model_dir, max_new_tokens=256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def relations_for(self, text):
        import torch

        prompt = f"[causal] {text}"
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            start = gen.index("{")
            obj = json.loads(gen[start:gen.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return []
        if not obj.get("causal"):
            return []
        rels = []
        for r in obj.get("relations", []):
            cause, effect = str(r.get("cause", "")).strip(), str(r.get("effect", "")).strip()
            if cause and effect and cause != effect:
                rels.append({"cause": cause, "effect": effect})
        return rels

    def extract_causal(self, texts):
        results = []
        for text in texts:
            rels = self.relations_for(text)
            results.append({"text": text, "causal": len(rels) > 0, "relations": rels})
        return results

    __call__ = extract_causal


def build_extractor(args):
    if args.llm_model:
        extractor = LLMCausalExtractor(args.llm_model)
    else:
        from reasongraph._extraction import CausalPointerExtractor

        extractor = CausalPointerExtractor(model=args.causal_model, gate_threshold=args.gate_threshold,
                                           embed_gate=args.embed_gate or None,
                                           embed_gate_threshold=args.embed_gate_threshold)
    return extractor


def load_cases(path, limit=None):
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def _discovered_contents(results):
    """Ordered, de-duplicated fact contents across all discover results and paths."""
    seen = set()
    ordered = []
    for r in results:
        for node in [r] + list(r.get("path", [])):
            content = node.get("content")
            if content and content not in seen:
                seen.add(content)
                ordered.append(content)
    return ordered


def _ordered_fraction(discovered, gold_chain):
    """Fraction of gold_chain recovered as an in-order subsequence of discovered."""
    if not gold_chain:
        return 1.0
    i = 0
    for content in discovered:
        if i < len(gold_chain) and content == gold_chain[i]:
            i += 1
    return i / len(gold_chain)


def score_case(graph, case):
    from reasongraph import ReasonGraph  # noqa: F401  (graph already built by caller)

    ingest = getattr(graph, "_eval_ingest", "sentences")
    for session, facts in case["sessions"].items():
        if ingest == "paragraphs":   # one paragraph per session, the splitter (if any) cuts it
            graph.add_texts_sync([" ".join(facts)], scopes={session}, resolve_conflicts=False)
        else:
            graph.add_texts_sync(facts, scopes={session}, resolve_conflicts=False)

    results = graph.discover_sync(case["question"], top_k=5, hops=4, max_results=10)
    discovered = _discovered_contents(results)
    top5 = [r.get("content") for r in results[:5]]
    gold = case["gold_chain"]

    chain = sum(1 for g in gold if g in discovered) / len(gold)
    chain_ord = _ordered_fraction(discovered, gold)
    r5 = sum(1 for g in gold if g in top5) / len(gold)

    hops = graph.causal_chain_sync(gold[0], gold[-1]) or []
    hop_spans = {h.get("cause", "") for h in hops} | {h.get("effect", "") for h in hops}

    answer = case["gold_answer"].casefold()
    answer_hit = any(answer in c.casefold() for c in discovered) or \
        any(answer in s.casefold() for s in hop_spans)

    return {
        "id": case["id"], "lang": case["lang"], "domain": case["domain"],
        "chain": chain, "chain_ord": chain_ord, "r5": r5,
        "answer": 1.0 if answer_hit else 0.0,
        "causal_chain": 1.0 if hops else 0.0, "hops": len(hops),
    }


def _avg(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def _group_table(rows, key, title):
    print(f"  {title:<10s} {'n':>3s} {'Chain':>6s} {'ChainOrd':>8s} {'R@5':>6s} {'Answer':>7s} {'CausChn':>8s}")
    print(f"  {'-'*10} {'-'*3} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*8}")
    for g in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == g]
        print(f"  {g:<10s} {len(sub):>3d} {_avg(sub,'chain'):>6.0%} {_avg(sub,'chain_ord'):>8.0%} "
              f"{_avg(sub,'r5'):>6.0%} {_avg(sub,'answer'):>7.0%} {_avg(sub,'causal_chain'):>8.0%}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--causal-model", default="berk/causal-span-pointer-mdeberta")
    parser.add_argument("--llm-model", default=None,
                        help="dir of the L1 multi-task LLM (merged); used instead of the pointer")
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--embed-gate", default=None)
    parser.add_argument("--embed-gate-threshold", type=float, default=0.9)
    parser.add_argument("--span-link-threshold", type=float, default=0.85,
                        help="same_as link for causal spans of near-identical meaning "
                             "(production REASONGRAPH_SPAN_LINK_THRESHOLD); lets cause/effect "
                             "spans chain across facts. Set 0 to disable.")
    parser.add_argument("--label", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ingest", choices=["sentences", "paragraphs"], default="sentences",
                        help="paragraphs = join each session's facts into one text before pushing")
    parser.add_argument("--split", default="off", help="sentence splitter for ingest: off | sat | sat:<model> | regex")
    parser.add_argument("--out", default=None, help="write per-case rows to this JSON")
    args = parser.parse_args(argv)

    label = args.label or args.llm_model or args.causal_model
    cases = load_cases(args.cases, args.limit)
    extractor = build_extractor(args)

    from reasongraph import ReasonGraph
    from eval_causal_extraction import _embed_model

    span_link = args.span_link_threshold if args.span_link_threshold > 0 else None
    embed_model = _embed_model()   # mirror production: REASONGRAPH_EMBED_MODEL (multilingual) not the English default
    t0 = time.perf_counter()
    rows = []
    for case in cases:
        graph = ReasonGraph(causal_extractor=extractor, span_link_threshold=span_link,
                            embed_model=embed_model,
                            sentence_splitter=(None if args.split == "off" else args.split))
        graph._eval_ingest = args.ingest
        graph.initialize_sync()
        rows.append(score_case(graph, case))
        graph.close_sync()
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*70}\n  E2 causal eval: {label}")
    print(f"  model={args.causal_model} gate={args.gate_threshold} "
          f"embed_gate={os.path.basename(args.embed_gate) if args.embed_gate else 'none'}"
          f"{'@'+str(args.embed_gate_threshold) if args.embed_gate else ''} "
          f"span_link={span_link} embed_model={embed_model}")
    print(f"  {len(cases)} cases in {elapsed:.0f}s\n{'='*70}")
    _group_table(rows, "lang", "Language")
    print()
    _group_table(rows, "domain", "Domain")
    print(f"\n  {'OVERALL':<10s} {len(rows):>3d} {_avg(rows,'chain'):>6.0%} {_avg(rows,'chain_ord'):>8.0%} "
          f"{_avg(rows,'r5'):>6.0%} {_avg(rows,'answer'):>7.0%} {_avg(rows,'causal_chain'):>8.0%}")
    print(f"{'='*70}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"label": label, "model": args.causal_model,
                       "gate_threshold": args.gate_threshold,
                       "embed_gate": args.embed_gate,
                       "embed_gate_threshold": args.embed_gate_threshold,
                       "rows": rows}, fh, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
