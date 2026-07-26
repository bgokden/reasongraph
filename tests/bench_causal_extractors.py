"""Head-to-head causal extraction benchmark.

Compares candidate cause->effect extractors on a labeled probe set spanning
four regimes -- English explicit, multilingual explicit, implicit (no
connective), and reversed/direction-sensitive phrasing -- so the default causal
extractor is chosen on measured directed-pair accuracy, not by elimination.

Runs ONE extractor per process (each loads a heavy model). Not a pytest test.

Metrics:
  recall         directed cause->effect pairs recovered (both spans, correct roles)
  direction      of pairs where both spans were found, fraction with correct
                 cause/effect assignment (catches inverted extractions)
  span_fidelity  fraction of extracted spans that are substrings of the source
                 (catches LLM hallucination)
  pairs/probe    average extracted pairs per probe (over-generation / noise)

Usage:
    uv run python tests/bench_causal_extractors.py --which gliner2
    uv run python tests/bench_causal_extractors.py --which relex-multi
    uv run python tests/bench_causal_extractors.py --which qwen
    uv run python tests/bench_causal_extractors.py --which cue
"""

from __future__ import annotations

import argparse
import re
import resource
import time
import unicodedata


# (kind, lang, text, gold_cause, gold_effect). Gold spans are canonical head
# phrases; matching is overlap-based so paraphrases still count.
PROBES = [
    # -- English explicit --
    ("explicit", "en", "Heavy rainfall caused severe flooding.", "heavy rainfall", "severe flooding"),
    ("explicit", "en", "Deforestation led to soil erosion across the region.", "deforestation", "soil erosion"),
    ("explicit", "en", "Smoking causes lung cancer.", "smoking", "lung cancer"),
    ("explicit", "en", "The rise in interest rates slowed economic growth.", "rise in interest rates", "economic growth"),
    ("explicit", "en", "Because the dam failed, the valley flooded.", "the dam failed", "the valley flooded"),
    ("explicit", "en", "Prolonged drought led to crop failure.", "prolonged drought", "crop failure"),
    ("explicit", "en", "The new tariff raised consumer prices.", "the new tariff", "consumer prices"),
    ("explicit", "en", "Antibiotic overuse results in bacterial resistance.", "antibiotic overuse", "bacterial resistance"),
    # -- Multilingual explicit --
    ("multilingual", "de", "Starke Regenfälle verursachten Überschwemmungen.", "Regenfälle", "Überschwemmungen"),
    ("multilingual", "es", "La deforestación provocó la erosión del suelo.", "deforestación", "erosión del suelo"),
    ("multilingual", "fr", "La sécheresse a provoqué de mauvaises récoltes.", "sécheresse", "mauvaises récoltes"),
    ("multilingual", "tr", "Aşırı yağış sele neden oldu.", "Aşırı yağış", "sel"),
    ("multilingual", "zh", "持续干旱导致农作物歉收。", "持续干旱", "农作物歉收"),
    ("multilingual", "ar", "أدى الجفاف الطويل إلى نقص المياه.", "الجفاف", "نقص المياه"),
    ("multilingual", "ru", "Сильные дожди вызвали наводнение.", "дожди", "наводнение"),
    ("multilingual", "pt", "O desmatamento causou a erosão do solo.", "desmatamento", "erosão do solo"),
    # -- Implicit (no connective) --
    ("implicit", "en", "The bridge collapsed. Commuters were stranded for hours.", "the bridge collapsed", "commuters were stranded"),
    ("implicit", "en", "A power outage hit the city. Hospitals switched to backup generators.", "power outage", "hospitals switched to backup generators"),
    ("implicit", "en", "The factory dumped chemicals into the river. Fish populations plummeted.", "factory dumped chemicals", "fish populations plummeted"),
    ("implicit", "en", "Interest rates jumped. Home sales fell sharply.", "interest rates jumped", "home sales fell"),
    ("implicit", "en", "The vaccine rollout accelerated. Infection rates dropped.", "vaccine rollout", "infection rates dropped"),
    ("implicit", "en", "Heavy snowfall blanketed the pass. The highway was closed.", "heavy snowfall", "the highway was closed"),
    # -- Reversed / direction-sensitive --
    ("reversed", "en", "The crash resulted from brake failure.", "brake failure", "the crash"),
    ("reversed", "en", "Thousands were evacuated because of the wildfire.", "the wildfire", "evacuated"),
    ("reversed", "en", "The outage was caused by a lightning strike.", "lightning strike", "the outage"),
    ("reversed", "en", "Crop yields fell due to the prolonged drought.", "prolonged drought", "crop yields fell"),
    ("reversed", "en", "The company went bankrupt as a result of the fraud.", "the fraud", "bankrupt"),
    ("reversed", "en", "Many flights were delayed owing to the storm.", "the storm", "flights were delayed"),
]

KINDS = ["explicit", "multilingual", "implicit", "reversed"]


def _peak_ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = re.sub(r"[\s​]+", " ", s)
    return s.strip(" .,:;!?،。")


_STOP = {"the", "a", "an", "of", "to", "in", "was", "were", "is", "are", "that", "which", "and"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"\s+", _norm(s)) if t and t not in _STOP}


def _span_matches(span: str, gold: str) -> bool:
    """A predicted span matches a gold phrase by containment or token overlap."""
    a, b = _norm(span), _norm(gold)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    ta, tb = _tokens(span), _tokens(gold)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return inter / min(len(ta), len(tb)) >= 0.5


def _is_substring(span: str, text: str) -> bool:
    return _norm(span) in _norm(text)


def score(pairs: list[dict], gold_cause: str, gold_effect: str, text: str) -> dict:
    """Score predicted pairs for one probe against its single gold pair."""
    directed_hit = False
    swapped_hit = False
    for p in pairs:
        c, e = p.get("cause", ""), p.get("effect", "")
        if _span_matches(c, gold_cause) and _span_matches(e, gold_effect):
            directed_hit = True
        elif _span_matches(c, gold_effect) and _span_matches(e, gold_cause):
            swapped_hit = True
    spans = [s for p in pairs for s in (p.get("cause", ""), p.get("effect", "")) if s]
    faithful = sum(1 for s in spans if _is_substring(s, text))
    return {
        "directed_hit": directed_hit,
        "swapped_hit": swapped_hit,
        "n_pairs": len(pairs),
        "n_spans": len(spans),
        "faithful_spans": faithful,
    }


# -- extractor builders (lazy: only the selected one loads) --

def build_gliner2():
    from reasongraph._extraction import GLiNER2Extractor
    ext = GLiNER2Extractor()
    ext("warmup")

    def extract(text):
        res = ext.extract_causal([text])[0]
        return [
            {"cause": r.get("cause", ""), "effect": r.get("effect", "")}
            for r in res.get("relations", [])
        ]
    return extract


def build_relex_multi():
    from gliner import GLiNER
    model = GLiNER.from_pretrained("knowledgator/gliner-relex-multi-v1.0")
    ent_labels = ["event", "condition", "situation", "outcome", "action", "other"]
    rel_labels = ["causes", "leads to", "results in"]

    def extract(text):
        _, relations = model.inference(
            texts=[text], labels=ent_labels, relations=rel_labels,
            threshold=0.3, relation_threshold=0.45,
            return_relations=True, flat_ner=False,
        )
        best = {}
        for r in relations[0]:
            head, tail = r["head"]["text"], r["tail"]["text"]
            if _norm(head) == _norm(tail):
                continue
            key = (_norm(head), _norm(tail))
            sc = r.get("score", 0.0)
            if key not in best or sc > best[key][2]:
                best[key] = (head, tail, sc)
        return [{"cause": h, "effect": t} for h, t, _ in best.values()]
    return extract


def _lmfe_prefix_fn(tokenizer, parser):
    """Build a transformers prefix_allowed_tokens_fn from lm-format-enforcer's
    core API. Replicates lmformatenforcer.integrations.transformers, whose own
    import breaks on transformers 5.x (it imports PreTrainedTokenizerBase from a
    moved path); the core TokenEnforcer is unaffected.
    """
    from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData

    vocab_size = len(tokenizer)
    token_0 = tokenizer.encode("0")[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for tid in range(vocab_size):
        if tid in special:
            continue
        after0 = tokenizer.decode([token_0, tid])[1:]
        plain = tokenizer.decode([tid])
        regular.append((tid, after0, len(after0) > len(plain)))

    def decode_fn(tokens):
        return tokenizer.decode(tokens).rstrip("�")

    data = TokenEnforcerTokenizerData(regular, decode_fn, tokenizer.eos_token_id, False, vocab_size)
    enforcer = TokenEnforcer(data, parser)

    def prefix_fn(batch_id, sent):
        return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
    return prefix_fn


def build_qwen():
    import json
    from transformers import pipeline
    from lmformatenforcer import JsonSchemaParser

    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"cause": {"type": "string"}, "effect": {"type": "string"}},
            "required": ["cause", "effect"],
        },
    }
    prefix_fn = _lmfe_prefix_fn(pipe.tokenizer, JsonSchemaParser(schema))
    system = (
        "You extract cause-effect relations. Return a JSON list of "
        '{"cause","effect"} objects using the text\'s own words. The cause is what '
        "produces or triggers; the effect is the outcome. Watch phrasing like "
        "'X resulted from Y' (Y is the cause) and 'X because of Y' (Y is the cause). "
        "If there is no causal relation, return []."
    )
    shots = [
        ("Heavy rainfall caused severe flooding.", '[{"cause": "Heavy rainfall", "effect": "severe flooding"}]'),
        ("The crash resulted from brake failure.", '[{"cause": "brake failure", "effect": "The crash"}]'),
        ("The sun was shining brightly.", "[]"),
    ]

    def extract(text):
        messages = [{"role": "system", "content": system}]
        for u, a in shots:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": text})
        out = pipe(
            messages, max_new_tokens=200, do_sample=False,
            prefix_allowed_tokens_fn=prefix_fn,
        )
        reply = out[0]["generated_text"][-1]["content"].strip()
        try:
            data = json.loads(reply)
        except Exception:
            return []
        pairs = []
        for d in data if isinstance(data, list) else []:
            if isinstance(d, dict) and d.get("cause") and d.get("effect"):
                pairs.append({"cause": d["cause"], "effect": d["effect"]})
        return pairs
    return extract


# Multilingual causal cue markers. "cause_first": text before marker is the
# cause (X marker Y => cause=X, effect=Y). "effect_first": text before marker is
# the effect (Y marker X => cause=X, effect=Y).
CUE_MARKERS = [
    # effect_first (the cause follows the marker)
    ("because of", "effect_first"), ("because", "effect_first"),
    ("due to", "effect_first"), ("owing to", "effect_first"),
    ("as a result of", "effect_first"), ("resulted from", "effect_first"),
    ("caused by", "effect_first"),
    ("debido a", "effect_first"), ("porque", "effect_first"),
    ("à cause de", "effect_first"), ("en raison de", "effect_first"),
    ("wegen", "effect_first"), ("nedeniyle", "effect_first"),
    ("由于", "effect_first"), ("因为", "effect_first"),
    ("بسبب", "effect_first"), ("из-за", "effect_first"),
    # cause_first (the cause precedes the marker)
    ("leads to", "cause_first"), ("led to", "cause_first"),
    ("results in", "cause_first"), ("resulted in", "cause_first"),
    ("causes", "cause_first"), ("caused", "cause_first"),
    ("so that", "cause_first"), ("therefore", "cause_first"),
    ("provocó", "cause_first"), ("causó", "cause_first"),
    ("a provoqué", "cause_first"), ("verursachten", "cause_first"),
    ("neden oldu", "cause_first"), ("导致", "cause_first"),
    ("أدى", "cause_first"), ("вызвали", "cause_first"), ("causou", "cause_first"),
]


def build_cue():
    def _clean(s):
        return s.strip(" ,.;:!?،。،").strip()

    def extract(text):
        low = text.lower()
        for marker, orient in CUE_MARKERS:
            idx = low.find(marker)
            if idx < 0:
                continue
            before = _clean(text[:idx])
            after = _clean(text[idx + len(marker):])
            if not before or not after:
                continue
            if orient == "cause_first":
                return [{"cause": before, "effect": after}]
            return [{"cause": after, "effect": before}]
        return []
    return extract


def build_hybrid():
    """Cue markers first (free, direction-correct on marked/reversed phrasing),
    falling back to gliner-relex-multi for implicit/unmarked sentences (its
    strength). The cue pass shields relex from its one weakness -- reversed
    phrasing -- because reversed sentences carry markers the cue handles first.
    """
    cue = build_cue()
    relex = build_relex_multi()

    def extract(text):
        pairs = cue(text)
        if pairs:
            return pairs
        return relex(text)
    return extract


BUILDERS = {
    "gliner2": build_gliner2,
    "relex-multi": build_relex_multi,
    "qwen": build_qwen,
    "cue": build_cue,
    "hybrid": build_hybrid,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=list(BUILDERS))
    args = ap.parse_args()

    t0 = time.perf_counter()
    extract = BUILDERS[args.which]()
    load_s = time.perf_counter() - t0

    per_kind = {k: {"n": 0, "hit": 0, "both": 0, "correct_dir": 0} for k in KINDS}
    tot = {"n": 0, "hit": 0, "both": 0, "correct_dir": 0, "spans": 0, "faithful": 0, "pairs": 0}
    infer_s = 0.0
    for kind, lang, text, gc, ge in PROBES:
        t1 = time.perf_counter()
        pairs = extract(text)
        infer_s += time.perf_counter() - t1
        s = score(pairs, gc, ge, text)
        pk = per_kind[kind]
        pk["n"] += 1
        tot["n"] += 1
        tot["spans"] += s["n_spans"]
        tot["faithful"] += s["faithful_spans"]
        tot["pairs"] += s["n_pairs"]
        if s["directed_hit"]:
            pk["hit"] += 1
            tot["hit"] += 1
        if s["directed_hit"] or s["swapped_hit"]:
            pk["both"] += 1
            tot["both"] += 1
            if s["directed_hit"]:
                pk["correct_dir"] += 1
                tot["correct_dir"] += 1

    def pct(a, b):
        return f"{100*a/b:.0f}%" if b else "  -"

    print(f"\n{'=' * 62}")
    print(f"  Causal extraction: {args.which}")
    print(f"{'=' * 62}")
    print(f"  load {load_s:.1f}s   infer {1000*infer_s/max(tot['n'],1):.0f} ms/probe"
          f"   peak RAM {_peak_ram_mb():.0f} MB")
    print(f"  {'kind':<13} {'recall':>7} {'dir-acc':>8}   (n)")
    print(f"  {'-'*13} {'-'*7} {'-'*8}")
    for k in KINDS:
        pk = per_kind[k]
        print(f"  {k:<13} {pct(pk['hit'], pk['n']):>7} {pct(pk['correct_dir'], pk['both']):>8}   ({pk['n']})")
    print(f"  {'-'*13} {'-'*7} {'-'*8}")
    print(f"  {'ALL':<13} {pct(tot['hit'], tot['n']):>7} {pct(tot['correct_dir'], tot['both']):>8}   ({tot['n']})")
    print(f"  span_fidelity {pct(tot['faithful'], tot['spans'])}   "
          f"pairs/probe {tot['pairs']/max(tot['n'],1):.1f}")
    print()


if __name__ == "__main__":
    main()
