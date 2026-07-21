from __future__ import annotations

import json
import os
from collections.abc import Callable


class NERExtractor:
    """Named entity extractor using a HuggingFace token classification model.

    Default model is dslim/bert-base-NER which recognizes PER, ORG, LOC, MISC.
    Lazy-loads the pipeline on first call.
    """

    DEFAULT_MODEL = "dslim/bert-base-NER"

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or self.DEFAULT_MODEL
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "ner",
                model=self._model_name,
                aggregation_strategy="simple",
            )

    def __call__(self, text: str) -> list[str]:
        """Extract entities from text. Returns deduplicated entity strings."""
        self._load()
        results = self._pipeline(text)
        seen = set()
        entities = []
        for ent in results:
            word = ent["word"].strip()
            # Filter out WordPiece artifacts (##subword tokens) and single chars
            if not word or word.startswith("##") or len(word) <= 1:
                continue
            if word not in seen:
                seen.add(word)
                entities.append(word)
        return entities


class GLiNER2Extractor:
    """Entity and relation extractor using GLiNER2.

    Single model that handles both NER and causal relation extraction.
    Lazy-loads the model on first call.

    Default model: fastino/gliner2-large-v1 (340M params, DeBERTa-v3-large).
    Requires: pip install reasongraph[gliner2]
    """

    DEFAULT_MODEL = "fastino/gliner2-large-v1"

    DEFAULT_ENTITY_TYPES = ["person", "organization", "location", "event"]
    DEFAULT_RELATION_TYPES = ["causes", "leads_to", "results_in"]

    def __init__(
        self,
        model: str | None = None,
        entity_types: list[str] | None = None,
        relation_types: list[str] | None = None,
    ) -> None:
        self._model_name = model or self.DEFAULT_MODEL
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self.relation_types = relation_types or self.DEFAULT_RELATION_TYPES
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from gliner2 import GLiNER2
            except ImportError:
                raise ImportError(
                    "GLiNER2 not installed. Install with: pip install reasongraph[gliner2]"
                )
            self._model = GLiNER2.from_pretrained(self._model_name)

    def __call__(self, text: str) -> list[str]:
        """Extract entities from text (compatible with ExtractorFn).

        Returns deduplicated entity strings.
        """
        self._load()
        raw = self._model.extract_entities(text, self.entity_types)
        # GLiNER2 returns {"entities": {"person": [...], "location": [...]}}
        entities_by_type = raw.get("entities", raw)

        seen = set()
        entities = []
        for type_key, ents in entities_by_type.items():
            if not isinstance(ents, list):
                continue
            for ent in ents:
                word = ent.strip() if isinstance(ent, str) else ""
                if word and word not in seen:
                    seen.add(word)
                    entities.append(word)
        return entities

    def extract_causal(self, texts: list[str]) -> list[dict]:
        """Extract cause-effect relations from texts (compatible with CausalExtractorFn).

        Args:
            texts: List of text strings to analyze.

        Returns:
            List of dicts (one per text) with keys:
                - 'text': original text
                - 'causal': bool
                - 'relations': list of {'cause': str, 'effect': str}
        """
        self._load()
        results = []
        for text in texts:
            raw = self._model.extract_relations(text, self.relation_types)
            # GLiNER2 returns {"relation_extraction": {"causes": [("a","b")], ...}}
            # or directly {"causes": [("a","b")], ...}
            rel_data = raw.get("relation_extraction", raw)

            relations = []
            seen_pairs = set()
            for rel_type, pairs in rel_data.items():
                if not isinstance(pairs, list):
                    continue
                for pair in pairs:
                    cause, effect = None, None
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        cause, effect = str(pair[0]).strip(), str(pair[1]).strip()
                    elif isinstance(pair, dict):
                        h = pair.get("head", {})
                        t = pair.get("tail", {})
                        cause = (h.get("text", "") if isinstance(h, dict) else str(h)).strip()
                        effect = (t.get("text", "") if isinstance(t, dict) else str(t)).strip()
                    if not cause or not effect or cause == effect:
                        continue
                    pair_key = (cause, effect)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    relations.append({"cause": cause, "effect": effect})

            results.append({
                "text": text,
                "causal": len(relations) > 0,
                "relations": relations,
            })
        return results


class ChatExtractor(GLiNER2Extractor):
    """GLiNER2 extractor tuned for conversational / companion memory.

    Extends the default entity set with conversational concepts -- ``preference``,
    ``plan``, ``topic`` -- alongside the usual person/organization/location/event.
    This lets things like "likes hard techno" or "planning a trip to Berlin"
    become entity nodes, so they can bridge separate conversations through the
    graph instead of being reachable only by weak embedding similarity.

    Entity and causal relation extraction are otherwise identical to
    ``GLiNER2Extractor``. Pass it explicitly:

        graph.add_texts(history, extractor=ChatExtractor())
    """

    DEFAULT_ENTITY_TYPES = [
        "person",
        "organization",
        "location",
        "event",
        "preference",
        "plan",
        "topic",
    ]


class OnnxTokenClassifierExtractor:
    """Generic BIO token-classification entity extractor backed by ONNX Runtime.

    Runs any HuggingFace token-classification model exported to ONNX (inputs
    ``input_ids`` + ``attention_mask``, output = per-token logits) and decodes
    BIO / typed-BIO tags into entity strings. The label scheme is read from the
    model's ``config.json`` ``id2label``, so the same class serves the
    place-entity model today and any future custom NER with no code change --
    only the model directory differs.

    Much lighter than GLiNER2: a single encoder forward pass (no zero-shot
    prompt) run through ONNX Runtime on CPU. Requires ``onnxruntime`` (an
    optional dependency, imported lazily on first use).

    Args:
        model_dir: Directory holding ``config.json`` and the tokenizer files.
        onnx_path: Path to the ``.onnx`` file. Defaults to
            ``<model_dir>/onnx/model.onnx``.
        keep_types: Optional iterable of entity type names to keep (e.g.
            ``{"CITY", "COUNTRY"}``). By default every non-``O`` type is kept.
        max_len: Tokenizer truncation length.
        providers: onnxruntime execution providers (default CPU).
    """

    def __init__(
        self,
        model_dir: str,
        onnx_path: str | None = None,
        keep_types: set[str] | None = None,
        max_len: int = 512,
        providers: list[str] | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.onnx_path = onnx_path or os.path.join(model_dir, "onnx", "model.onnx")
        self.keep_types = set(keep_types) if keep_types else None
        self.max_len = max_len
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = None
        self._tok = None
        self._id2label: dict[int, str] | None = None

    def _load(self):
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            )
        from transformers import AutoTokenizer

        with open(os.path.join(self.model_dir, "config.json")) as f:
            cfg = json.load(f)
        self._id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        self._session = ort.InferenceSession(self.onnx_path, providers=self.providers)

    def __call__(self, text: str) -> list[str]:
        """Extract entities from text. Returns deduplicated entity strings."""
        self._load()
        import numpy as np

        enc = self._tok(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="np",
        )
        offsets = enc["offset_mapping"][0]
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = self._session.run(None, feeds)[0][0]  # [seq_len, num_labels]
        tag_ids = logits.argmax(-1)
        return self._decode_bio(text, offsets, tag_ids, self._id2label, self.keep_types)

    @staticmethod
    def _decode_bio(text, offsets, tag_ids, id2label, keep_types) -> list[str]:
        """Decode per-token BIO(-typed) tags into deduplicated entity strings.

        Pure and model-independent: groups consecutive B-/I- tags of the same
        type into character spans (via the token offsets) and slices the source
        text. A ``B-`` tag or a type change always starts a new span, so
        adjacent entities of different types are not merged.
        """
        spans: list[list] = []
        cur: list | None = None
        for (start, end), tid in zip(offsets, tag_ids):
            if start == end:  # special token (CLS/SEP/PAD) -> zero-width offset
                continue
            label = id2label[int(tid)]
            if label == "O":
                if cur is not None:
                    spans.append(cur)
                    cur = None
                continue
            prefix, _, typ = label.partition("-")
            typ = typ or "ENT"  # untyped BIO (bare 'B'/'I') -> generic type
            if prefix == "B" or cur is None or cur[0] != typ:
                if cur is not None:
                    spans.append(cur)
                cur = [typ, int(start), int(end)]
            else:  # I- continuing the same type
                cur[2] = int(end)
        if cur is not None:
            spans.append(cur)

        keep = set(keep_types) if keep_types else None
        seen: set[str] = set()
        entities: list[str] = []
        for typ, start, end in spans:
            if keep is not None and typ not in keep:
                continue
            fragment = text[start:end].strip()
            if fragment and fragment not in seen:
                seen.add(fragment)
                entities.append(fragment)
        return entities


class GlinerExtractor:
    """GLiNER v1 (urchade) zero-shot entity extractor, optionally via ONNX.

    Zero-shot flexible NER over a chosen label set, with first-class ONNX
    Runtime inference (converted and cached on first use) for a much lighter,
    faster CPU footprint than GLiNER2 while keeping open entity types.
    Multilingual when a multilingual checkpoint (e.g. ``urchade/gliner_multi-v2.1``)
    is used. Unlike GLiNER2 it does not extract causal relations -- it is the
    fast, flexible entity path.

    The default checkpoint is ``gliner-community/gliner_small-v2.5``: on a 10-language
    WikiANN benchmark it reached 86% entity recall (the best of the tested
    extractors) at ~67 ms/call and ~2.2 GB RAM, and unlike GLiNER2 it stays strong
    on Korean/Arabic/Turkish/Russian. The older ``urchade/gliner_multi-v2.1`` scored
    only ~12% recall -- do not use it.

    Args:
        model_id: HF model id or local dir (default ``gliner-community/gliner_small-v2.5``).
        labels: Entity types to extract (GLiNER is zero-shot, so any labels work).
        onnx: Run through ONNX Runtime. On first use the model is converted and
            cached under ``cache_dir``; later runs load the cached ONNX directly.
        onnx_file: Which ONNX file to load. Use ``"model_quantized.onnx"`` for the
            int8 build (smallest / fastest on CPU).
        cache_dir: Directory for the converted ONNX model (defaults under
            ``~/.cache/reasongraph``).
        threshold: Minimum entity score to keep (0.3 suits the default v2.5 model).
    """

    DEFAULT_MODEL = "gliner-community/gliner_small-v2.5"
    DEFAULT_LABELS = ["person", "organization", "location", "event"]

    def __init__(
        self,
        model_id: str | None = None,
        labels: list[str] | None = None,
        onnx: bool = False,
        onnx_file: str = "model.onnx",
        cache_dir: str | None = None,
        threshold: float = 0.3,
    ) -> None:
        self.model_id = model_id or self.DEFAULT_MODEL
        self.labels = labels or list(self.DEFAULT_LABELS)
        self.onnx = onnx
        self.onnx_file = onnx_file
        self.cache_dir = cache_dir
        self.threshold = threshold
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from gliner import GLiNER

        if not self.onnx:
            self._model = GLiNER.from_pretrained(self.model_id)
            return

        cache = self.cache_dir or os.path.join(
            os.path.expanduser("~/.cache/reasongraph/gliner_onnx"),
            self.model_id.replace("/", "__"),
        )
        onnx_path = os.path.join(cache, self.onnx_file)
        if not os.path.exists(onnx_path):
            # One-time conversion: load torch, persist config+tokenizer, export ONNX.
            torch_model = GLiNER.from_pretrained(self.model_id)
            torch_model.save_pretrained(cache)
            torch_model.export_to_onnx(cache, quantize=("quantized" in self.onnx_file))
        self._model = GLiNER.from_pretrained(
            cache, load_onnx_model=True, onnx_model_file=self.onnx_file
        )

    def __call__(self, text: str) -> list[str]:
        """Extract entities from text. Returns deduplicated entity strings."""
        self._load()
        preds = self._model.predict_entities(text, self.labels, threshold=self.threshold)
        seen: set[str] = set()
        entities: list[str] = []
        for p in preds:
            frag = str(p.get("text", "")).strip()
            if frag and frag not in seen:
                seen.add(frag)
                entities.append(frag)
        return entities


# Type alias for any entity extractor callable: text -> list of entity strings
ExtractorFn = Callable[[str], list[str]]

# Type alias for causal extractor: list[str] -> list[dict]
CausalExtractorFn = Callable[[list[str]], list[dict]]
