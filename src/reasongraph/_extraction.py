from __future__ import annotations

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


# Type alias for any entity extractor callable: text -> list of entity strings
ExtractorFn = Callable[[str], list[str]]

# Type alias for causal extractor: list[str] -> list[dict]
CausalExtractorFn = Callable[[list[str]], list[dict]]
