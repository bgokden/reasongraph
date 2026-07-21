"""Multilingual NER benchmark on a real gold dataset (WikiANN / PAN-X).

Answers the fair version of "which extractor should we use": entity recall /
precision / F1 per language on Wikipedia-derived gold spans, not a handful of
English sentences. Type-agnostic string matching (the models use different type
schemes; what matters for graph bridging is catching the entity string).

WikiANN is silver-standard (auto-derived from Wikipedia links) and CJK text is
reconstructed by joining tokens, so treat absolute numbers as indicative and
focus on the *relative* ranking and the English-vs-non-English gap.

Runs ONE extractor per process. Not a pytest test.

Usage:
    uv run python tests/bench_ner_multilingual.py --which gliner2
    uv run python tests/bench_ner_multilingual.py --which gliner-multi-v2.1
    uv run python tests/bench_ner_multilingual.py --which place-onnx --langs en,es,de,tr
"""

from __future__ import annotations

import argparse
import os
import re
import resource
import time

LANGS = ["en", "es", "fr", "de", "nl", "tr", "ru", "ar", "zh", "ko"]
NO_SPACE = {"zh", "ja"}
PLACE_MODEL_DIR = os.environ.get(
    "REASONGRAPH_PLACE_MODEL_DIR", "/home/berk/repos/trainner/archive/web_release"
)
LABELS = ["person", "organization", "location"]


def _peak_ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


NAMED = {
    "gliner-multi-v2.1": "urchade/gliner_multi-v2.1",
    "gliner-small-v2.5": "gliner-community/gliner_small-v2.5",
    "gliner-large-v2.5": "gliner-community/gliner_large-v2.5",
}


def build(which: str, threshold: float, onnx: bool = False):
    if which == "gliner2":
        from reasongraph._extraction import GLiNER2Extractor
        return GLiNER2Extractor(entity_types=LABELS)
    if which == "place-onnx":
        from reasongraph._extraction import OnnxTokenClassifierExtractor
        return OnnxTokenClassifierExtractor(
            PLACE_MODEL_DIR, onnx_path=os.path.join(PLACE_MODEL_DIR, "onnx", "model_fp16.onnx")
        )
    from reasongraph._extraction import GlinerExtractor
    model = NAMED.get(which) or (which.split(":", 1)[1] if which.startswith("gliner:") else None)
    if model:
        return GlinerExtractor(model, labels=LABELS, threshold=threshold, onnx=onnx)
    raise SystemExit(f"unknown extractor: {which}")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _match(gold: str, extracted: list[str]) -> bool:
    g = _norm(gold)
    if not g:
        return False
    return any(g == e or g in e or e in g for e in (_norm(x) for x in extracted))


def _gold_entities(spans: list[str]) -> list[str]:
    out = []
    for sp in spans:  # e.g. "LOC: India"
        _, _, text = sp.partition(":")
        text = text.strip()
        if text:
            out.append(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True)
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--n", type=int, default=40, help="sentences per language")
    ap.add_argument("--threshold", type=float, default=0.3, help="GLiNER score threshold")
    ap.add_argument("--onnx", action="store_true", help="run GLiNER via ONNX (convert+cache)")
    args = ap.parse_args()
    langs = args.langs.split(",")

    from datasets import load_dataset

    t0 = time.perf_counter()
    ext = build(args.which, args.threshold, args.onnx)
    ext("warmup text")
    load_s = time.perf_counter() - t0

    per_lang = {}
    tot_g = tot_e = tot_hit = tot_precision_hit = 0
    calls = 0
    infer_s = 0.0
    for lang in langs:
        ds = load_dataset("unimelb-nlp/wikiann", lang, split=f"test[:{args.n}]")
        g_tot = e_tot = hit = phit = 0
        for ex in ds:
            gold = _gold_entities(ex["spans"])
            if not gold:
                continue
            joiner = "" if lang in NO_SPACE else " "
            text = joiner.join(ex["tokens"])
            t1 = time.perf_counter()
            ents = ext(text)
            infer_s += time.perf_counter() - t1
            calls += 1
            g_tot += len(gold)
            e_tot += len(ents)
            hit += sum(1 for gd in gold if _match(gd, ents))
            phit += sum(1 for e in ents if any(_match(gd, [e]) for gd in gold))
        recall = hit / g_tot if g_tot else 0.0
        precision = phit / e_tot if e_tot else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_lang[lang] = (recall, precision, f1, g_tot)
        tot_g += g_tot
        tot_e += e_tot
        tot_hit += hit
        tot_precision_hit += phit

    R = tot_hit / tot_g if tot_g else 0.0
    P = tot_precision_hit / tot_e if tot_e else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0

    print(f"\n{'=' * 66}")
    print(f"  Multilingual NER (WikiANN): {args.which}")
    print(f"{'=' * 66}")
    print(f"  load+warmup {load_s:.2f}s   avg infer {1000*infer_s/max(calls,1):.0f} ms/call"
          f"   peak RAM {_peak_ram_mb():.0f} MB")
    print(f"  {'lang':<6s} {'recall':>7s} {'prec':>7s} {'F1':>7s} {'#gold':>7s}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for lang in langs:
        r, p, f, n = per_lang[lang]
        print(f"  {lang:<6s} {r:>6.0%} {p:>7.0%} {f:>7.0%} {n:>7d}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    print(f"  {'ALL':<6s} {R:>6.0%} {P:>7.0%} {F:>7.0%} {tot_g:>7d}")
    print()


if __name__ == "__main__":
    main()
