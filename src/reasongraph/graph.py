from __future__ import annotations

import contextlib

import asyncio
import os
import re
import warnings
from collections import deque
from collections.abc import Mapping
from datetime import datetime

from reasongraph._canonical import AliasCanonicalizer, CanonicalizerFn
from reasongraph._embeddings import EmbeddingManager, EmbedderLike
from reasongraph._extraction import (
    NERExtractor,
    GLiNER2Extractor,
    GlinerExtractor,
    HybridCausalExtractor,
    CausalPointerExtractor,
    ExtractorFn,
    CausalExtractorFn,
)
from reasongraph._types import Node, Edge
from reasongraph.backends._base import Backend
from reasongraph.backends._memory import MemoryBackend


class ReasonGraph:
    """A graph-based reasoning engine with embedding search and multi-hop traversal.

    Uses an async-first design with sync convenience wrappers.
    Defaults to a pure Python in-memory backend with brute-force cosine similarity.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        embed_model: EmbedderLike = None,
        rerank_model: str | None = None,
        forget_after: int = 30,
        forget_every: float | None = None,
        synthesizer=None,
        causal_extractor: CausalExtractorFn | bool | None = None,
        isolate_traversal: bool = False,
        conflict_resolver=None,
        canonicalizer: CanonicalizerFn | Mapping[str, str] | None = None,
        link_contained_entities: bool | None = None,
        span_linker=None,
        span_link_logit: float | None = None,
        span_link_floor: float | None = None,
        span_link_top_k: int | None = None,
        resolve_back_references: bool | None = None,
        max_degree: int | None = None,
        span_link_threshold: float | None = None,
        sentence_splitter=None,
        embed_query_prefix: str | None = None,
        embed_document_prefix: str | None = None,
    ) -> None:
        self.backend = backend or MemoryBackend()
        # Asymmetric retrievers (e5, nomic) want "query: " / "passage: " markers; known
        # model names get them automatically, explicit prefixes override.
        self.embeddings = EmbeddingManager(
            embed_model=embed_model, rerank_model=rerank_model,
            query_prefix=embed_query_prefix, document_prefix=embed_document_prefix,
        )
        self.forget_after = forget_after
        self.forget_every = forget_every
        self._last_forget: datetime | None = None
        # Default traversal isolation. False (default) keeps the shared-graph
        # behaviour: a scoped query seeds from its scopes but the walk crosses all
        # scopes (the cross-session discovery feature). True confines the walk to
        # the query scopes -- the multi-tenant setting. Overridable per query.
        self.isolate_traversal = isolate_traversal
        # Optional conflict resolver (a ConflictResolver, e.g. NLIConflictResolver).
        # When set, a new fact that contradicts existing ones soft-supersedes them:
        # a "supersedes" edge marks the old fact so it drops out of default recall
        # while staying auditable. None (default) disables conflict resolution.
        self.conflict_resolver = conflict_resolver
        # Causal span linking: when set (e.g. 0.85), a newly extracted cause/effect
        # span is tied with a ``same_as`` edge to existing causal spans whose
        # embedding is at least this similar ("the river flooded the old town" ~
        # "the old town flooded"). The causal walk follows those ties, so chains
        # can cross facts that phrase the same event differently.
        self.span_link_threshold = span_link_threshold
        # Sentence splitting at ingest: a splitter object, or "sat" / "regex" / None. When set,
        # add_texts(split=None) splits every input into sentences (one fact each).
        from reasongraph._split import resolve_splitter
        self.sentence_splitter = resolve_splitter(sentence_splitter)
        # When a resolver is configured, resolve on every add unless a call says
        # otherwise. Set False to make resolution opt-in per call
        # (``add_texts(..., resolve_conflicts=True)``), e.g. when the resolver
        # calls an LLM and callers should decide per write.
        self.resolve_conflicts_by_default = True
        # Causal extraction (the headline feature) is ON by default. This holds
        # the caller's choice: None -> build the default hybrid causal extractor
        # lazily on first use; False -> disable causal extraction; a
        # callable/object with extract_causal -> use it. Resolved (and any model
        # built) lazily, never at construction time.
        self._causal_extractor_arg = causal_extractor
        # Optional pluggable synthesizer: a callable(query, context) -> str, or an
        # object with a synthesize(query, context) method. Keeps LLMs out of the
        # library core -- bring your own small model.
        self._synthesizer = self._normalize_synthesizer(synthesizer)
        # Optional entity canonicalizer: a callable(str) -> str, or an alias
        # Mapping (wrapped in AliasCanonicalizer). Applied to each extracted entity
        # before its node is created, so surface variants ("Apple Inc." / "Apple",
        # "the Fed" / "Federal Reserve") collapse to one shared-entity bridge. This
        # is the light stand-in for coreference; None (default) leaves entities
        # verbatim. Overridable per add_text/add_texts call.
        if canonicalizer is None and os.environ.get("REASONGRAPH_ENTITY_NORMALIZE", "").strip() not in ("", "0", "false", "off"):
            from reasongraph._canonical import EntityNormalizer
            canonicalizer = EntityNormalizer()
        self._canonicalizer = self._normalize_canonicalizer(canonicalizer)
        # Containment bridging: an entity that is a whole-word prefix of another ("malzeme" /
        # "malzeme eksikligi", "Sabah" / "Ahmet sabah" is not: same first word required) links
        # the new fact to the other entity node too, so inflected recurrences bridge. Looked up
        # by first word through an indexed prefix query, never a scan. Opt-in.
        if link_contained_entities is None:
            link_contained_entities = os.environ.get("REASONGRAPH_ENTITY_CONTAINMENT", "").strip() not in ("", "0", "false", "off")
        self.link_contained_entities = bool(link_contained_entities)
        # Span linking: by default two causal spans are tied when their embedding cosine
        # clears span_link_threshold. A span linker is a cross-encoder trained to answer
        # "do these two spans describe the same event?" (effect of one hop, cause of the
        # next), which recovers paraphrased hops (nominalisation vs clause) that cosine
        # misses; the tie is made when its logit clears span_link_logit. Either an object
        # with predict(pairs) -> logits, or a model id / path (hf://owner/repo or local).
        if span_linker is None and os.environ.get("REASONGRAPH_SPAN_LINKER", "").strip():
            span_linker = os.environ["REASONGRAPH_SPAN_LINKER"].strip()
        self._span_linker_spec = span_linker
        self._span_linker = None
        if span_link_logit is None:
            span_link_logit = float(os.environ.get("REASONGRAPH_SPAN_LINK_LOGIT", "-2.2"))
        self.span_link_logit = span_link_logit
        # A similarity linker REPLACES the embedder's own cosine, which costs links on pairs cosine
        # already agreed about. Give it a floor and it ADDS instead: cosine keeps every link it would
        # have made, and the linker may only add pairs whose cosine sits in [floor, threshold).
        # Measured: replacing triples root recall on rephrased chains but loses a point on chains that
        # were never broken; the floor is what removes that loss. None keeps the replacing behaviour.
        if span_link_floor is None:
            _floor = os.environ.get("REASONGRAPH_SPAN_LINK_FLOOR", "").strip()
            span_link_floor = float(_floor) if _floor else None
        self.span_link_floor = span_link_floor
        # How many near neighbours a span is compared against before the linker (or plain cosine)
        # judges them. This used to widen to 10 whenever a linker was configured. Measured on 341
        # cases, that was strictly worse on every axis: the wider shortlist floods the walk with
        # look-alikes, halving the gain on rephrased chains and turning a +1 on ordinary chains into
        # a -1. Six for everyone; raise it only with your own numbers.
        if span_link_top_k is None:
            _tk = os.environ.get("REASONGRAPH_SPAN_LINK_TOP_K", "").strip()
            span_link_top_k = int(_tk) if _tk else None
        self.span_link_top_k = span_link_top_k
        # A note that points at the one before it ("This broke checkout", "Dadurch stieg die Last")
        # states a cause the extractor cannot see, because it reads one sentence at a time. With
        # this on, such a sentence is linked to the fact that precedes it in the same batch.
        # Off by default: measured on generated cases it recovers a real and otherwise-lost link
        # with no cross-case damage, but its precision on ordinary traffic is not yet established.
        if resolve_back_references is None:
            resolve_back_references = os.environ.get(
                "REASONGRAPH_RESOLVE_BACK_REFERENCES", "").lower() in ("1", "true", "yes")
        self.resolve_back_references = resolve_back_references
        # Hub cap: an entity linked to thousands of facts ("Apple", "the company") would
        # turn every walk through it into a scan. A walk expands at most max_degree
        # neighbours of a node, the ones nearest to the question.
        if max_degree is None:
            max_degree = int(os.environ.get("REASONGRAPH_MAX_DEGREE", "64"))
        self.max_degree = max(1, int(max_degree))
        self._req_cache: dict | None = None   # see request_cache(): per-request metadata memo

    @staticmethod
    def _normalize_synthesizer(synthesizer):
        if synthesizer is None:
            return None
        if hasattr(synthesizer, "synthesize"):
            return synthesizer.synthesize
        if callable(synthesizer):
            return synthesizer
        raise TypeError(
            "synthesizer must be None, a callable(query, context) -> str, or an "
            "object with a synthesize(query, context) method"
        )

    @staticmethod
    def _normalize_canonicalizer(canonicalizer):
        """Coerce the canonicalizer arg to a callable(str) -> str, or None.

        A Mapping is a convenience for the common case (an alias map): it is
        wrapped in an ``AliasCanonicalizer`` so callers can pass a plain dict.
        """
        if canonicalizer is None:
            return None
        if isinstance(canonicalizer, Mapping):
            return AliasCanonicalizer(canonicalizer)
        if callable(canonicalizer):
            return canonicalizer
        raise TypeError(
            "canonicalizer must be None, a callable(str) -> str, or a Mapping of "
            "surface form -> canonical name"
        )

    def _resolve_canonicalizer(self, canonicalizer):
        """Per-call canonicalizer wins when given; else the graph default."""
        if canonicalizer is None:
            return self._canonicalizer
        return self._normalize_canonicalizer(canonicalizer)

    _CONTAIN_STOP = frozenset("the a an of and de der die das den het een el la los las le les du des del van von bir ve ile".split())

    async def _contained_entities(self, entities: list[str], batch_seen: set[str]) -> list[str]:
        """Existing entity nodes that contain, or are contained in, one of ``entities`` as a
        whole-word prefix (same first word). Short or stopword-led names are skipped."""
        out: list[str] = []
        mine = {e.lower() for e in entities}
        for entity in entities:
            words = entity.lower().split()
            if not words or len(entity) < 4 or words[0] in self._CONTAIN_STOP or len(words) > 4:
                continue
            try:
                candidates = list(await self.backend.entities_starting_with(words[0], limit=20))
            except NotImplementedError:
                return out
            candidates += [b for b in batch_seen if b.lower().split()[:1] == words[:1]]
            for cand in candidates:
                c = cand.lower()
                if c in mine or c == entity.lower():
                    continue
                cw = c.split()
                if cw[: len(words)] == words or words[: len(cw)] == cw:
                    if len(min(c, entity.lower(), key=len)) >= 4 and cand not in out:
                        out.append(cand)
        return out

    @staticmethod
    def _canonicalize(entities: list[str], canonicalizer: CanonicalizerFn) -> list[str]:
        """Map each entity to its canonical form, dropping empties and duplicates.

        Order is preserved (first occurrence wins). Two surface forms that map to
        the same canonical name collapse to a single entity for this text, so only
        one entity node and one edge are created for the pair.
        """
        seen: set[str] = set()
        canonical: list[str] = []
        for entity in entities:
            name = canonicalizer(entity)
            if name and name not in seen:
                seen.add(name)
                canonical.append(name)
        return canonical

    # -- Lifecycle --

    async def initialize(self) -> None:
        """Initialize the backend (create tables etc.)."""
        await self.backend.initialize()

    async def close(self) -> None:
        """Close the backend and release resources."""
        await self.backend.close()

    async def __aenter__(self) -> ReasonGraph:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # -- Core operations --

    async def add_nodes(
        self,
        nodes: list[tuple[str, str]],
        scopes: set[str] | list[str] | None = None,
    ) -> None:
        """Add nodes to the graph.

        Args:
            nodes: List of (content, type) tuples. Type is 'text' or 'entity'.
            scopes: Optional free-text scope tags to attach to every node.
                Existing nodes accumulate scopes (union), never lose them.
        """
        scope_set = set(scopes) if scopes else set()
        texts = [content for content, _ in nodes]
        embeddings = self.embeddings.encode_batch(texts)
        node_objs = [
            # Each node gets its own set copy; the backend may merge into it.
            Node(content=content, type=node_type, embedding=emb, scopes=set(scope_set))
            for (content, node_type), emb in zip(nodes, embeddings)
        ]
        await self.backend.insert_nodes(node_objs)

    async def add_edges(
        self, edges: list[tuple[str, str] | tuple[str, str, str | None]]
    ) -> None:
        """Add edges to the graph.

        Args:
            edges: List of ``(from_content, to_content)`` tuples, or
                ``(from_content, to_content, label)`` to type the edge (e.g.
                ``"causes"`` for a directed cause->effect link).
        """
        edge_objs = []
        for e in edges:
            f, t = e[0], e[1]
            label = e[2] if len(e) > 2 else None
            edge_objs.append(Edge(from_content=f, to_content=t, label=label))
        await self.backend.insert_edges(edge_objs)

    @staticmethod
    def _build_default_extractor():
        """Pick the default entity extractor by what is installed.

        Prefers ``GlinerExtractor`` (gliner_small-v2.5): fast, multilingual, and
        the highest entity recall in the benchmark. Falls back to
        ``GLiNER2Extractor`` (adds causal relations, English/European) and then
        ``NERExtractor`` (BERT NER). Pass an explicit ``extractor`` to override --
        e.g. ``GlinerExtractor("gliner-community/gliner_large-v2.5")`` for higher
        precision, or ``GLiNER2Extractor()`` when you want causal-relation edges.
        """
        try:
            import gliner as _gliner_check  # noqa: F401
            return GlinerExtractor()
        except ImportError:
            pass
        try:
            import gliner2 as _gliner2_check  # noqa: F401
            return GLiNER2Extractor()
        except ImportError:
            return NERExtractor()

    @staticmethod
    def _build_default_causal_extractor():
        """Build the default causal extractor, best backend first.

        Causal extraction is the headline feature, so it defaults ON. Preference:
        the span-pointer model (``CausalPointerExtractor``, ~0.70 F1 on CNC
        Subtask-2 dev -- the strongest available, beating a few-shot LLM baseline
        and the hybrid) when the optional ``causal-span-model`` package is
        installed; otherwise the ``HybridCausalExtractor`` (cue + relex, needs
        ``gliner``). Both are lazy (models load on first call), so building here is
        cheap. Returns ``None`` and warns once when neither backend is installed,
        so the drop is visible rather than silent.

        Environment: ``REASONGRAPH_CAUSAL_MODEL`` (HF repo id or local dir),
        ``REASONGRAPH_CAUSAL_GATE_THRESHOLD`` (0.5 = argmax gate, 1.0 = off),
        ``REASONGRAPH_CAUSAL_EMBED_GATE`` (path or ``hf://repo/file`` of an embedding-gate
        .joblib) and ``REASONGRAPH_CAUSAL_EMBED_GATE_THRESHOLD`` (default 0.9);
        ``REASONGRAPH_CAUSAL_ONNX`` (path or ``hf://owner/repo/file.onnx``: the pointer's
        ONNX export, 2-2.5x faster on CPU, same spans) and ``REASONGRAPH_CAUSAL_ONNX_THREADS``;
        ``REASONGRAPH_CAUSAL_TOKEN_GATE`` (dir or ``hf://owner/repo/subfolder`` of a
        fine-tuned causal/non-causal sequence classifier, takes precedence) and
        ``REASONGRAPH_CAUSAL_TOKEN_GATE_THRESHOLD`` (default 0.1).
        """
        try:
            import causal_span_model as _pointer_check  # noqa: F401
            import os
            kwargs = {}
            if os.environ.get("REASONGRAPH_CAUSAL_MODEL"):
                kwargs["model"] = os.environ["REASONGRAPH_CAUSAL_MODEL"]
            if os.environ.get("REASONGRAPH_CAUSAL_GATE_THRESHOLD"):
                kwargs["gate_threshold"] = float(os.environ["REASONGRAPH_CAUSAL_GATE_THRESHOLD"])
            if os.environ.get("REASONGRAPH_CAUSAL_EMBED_GATE"):
                kwargs["embed_gate"] = os.environ["REASONGRAPH_CAUSAL_EMBED_GATE"]
                if os.environ.get("REASONGRAPH_CAUSAL_EMBED_GATE_THRESHOLD"):
                    kwargs["embed_gate_threshold"] = float(os.environ["REASONGRAPH_CAUSAL_EMBED_GATE_THRESHOLD"])
            if os.environ.get("REASONGRAPH_CAUSAL_ONNX"):
                kwargs["onnx"] = os.environ["REASONGRAPH_CAUSAL_ONNX"]
                if os.environ.get("REASONGRAPH_CAUSAL_ONNX_THREADS"):
                    kwargs["onnx_threads"] = int(os.environ["REASONGRAPH_CAUSAL_ONNX_THREADS"])
            if os.environ.get("REASONGRAPH_CAUSAL_TOKEN_GATE"):
                kwargs["token_gate"] = os.environ["REASONGRAPH_CAUSAL_TOKEN_GATE"]
                if os.environ.get("REASONGRAPH_CAUSAL_TOKEN_GATE_THRESHOLD"):
                    kwargs["token_gate_threshold"] = float(os.environ["REASONGRAPH_CAUSAL_TOKEN_GATE_THRESHOLD"])
            return CausalPointerExtractor(**kwargs)
        except ImportError:
            pass
        try:
            import gliner as _gliner_check  # noqa: F401
        except ImportError:
            warnings.warn(
                "Causal extraction disabled: no causal extractor is available. "
                "Install causal-span-model for the best span-pointer model, "
                "reasongraph[gliner] (gliner>=0.2.27) for the hybrid, or pass "
                "causal_extractor=... .",
                stacklevel=2,
            )
            return None
        return HybridCausalExtractor()

    def _resolve_causal_extractor(self) -> CausalExtractorFn | None:
        """Return the causal callable to use, per the constructor's choice.

        None arg -> build (once, lazily) and cache the default hybrid; False ->
        disabled; a callable/object -> that (its ``extract_causal`` when present,
        so an object is not mistaken for an entity extractor).
        """
        arg = self._causal_extractor_arg
        if arg is False:
            return None
        if arg is None:
            if not hasattr(self, "_default_causal_extractor"):
                self._default_causal_extractor = self._build_default_causal_extractor()
            obj = self._default_causal_extractor
        else:
            obj = arg
        if obj is None:
            return None
        return obj.extract_causal if hasattr(obj, "extract_causal") else obj

    async def add_text(
        self,
        text: str,
        extractor: ExtractorFn | None = None,
        scopes: set[str] | list[str] | None = None,
        causal_extractor: CausalExtractorFn | None = None,
        causal: bool | None = None,
        dedup_threshold: float | None = None,
        dedup_scopes=None,
        dedup_entity_gate: bool = False,
        resolve_conflicts: bool | None = None,
        canonicalizer: CanonicalizerFn | Mapping[str, str] | None = None,
    ) -> list[str]:
        """Add text to the graph with automatic entity and causal extraction.

        Creates a text node for the input, extracts entities using the provided
        extractor, creates entity nodes, and links each entity to the text node.
        Causal relations are extracted by default (see ``add_texts``).

        Args:
            text: The text content to add.
            extractor: A callable(str) -> list[str] that extracts entity strings.
                Defaults to GlinerExtractor (gliner_small-v2.5) when `gliner` is
                installed, else GLiNER2Extractor, else NERExtractor.
            scopes: Optional free-text scope tags for the text and its entities.
            causal_extractor: Optional causal extractor override (see ``add_texts``).
            causal: True forces causal extraction (raises if unavailable), False
                disables it, None (default) runs it when an extractor is available.
            canonicalizer: Optional per-call entity canonicalizer (see ``add_texts``).

        Returns:
            List of extracted entity strings (canonical forms when a canonicalizer
            is in effect).
        """
        result = await self.add_texts(
            [text], extractor=extractor, causal_extractor=causal_extractor,
            scopes=scopes, causal=causal, dedup_threshold=dedup_threshold, dedup_scopes=dedup_scopes, dedup_entity_gate=dedup_entity_gate,
            resolve_conflicts=resolve_conflicts, canonicalizer=canonicalizer,
        )
        return result[0]

    async def add_texts(
        self,
        texts: list[str],
        extractor: ExtractorFn | None = None,
        causal_extractor: CausalExtractorFn | None = None,
        scopes: set[str] | list[str] | None = None,
        causal: bool | None = None,
        dedup_threshold: float | None = None,
        dedup_scopes=None,
        dedup_entity_gate: bool = False,
        resolve_conflicts: bool | None = None,
        canonicalizer: CanonicalizerFn | Mapping[str, str] | None = None,
        split: bool | None = None,
    ) -> list[list[str]]:
        """Add multiple texts with automatic entity and causal extraction.

        ``split``: split each text into sentences first and store one fact per sentence
        (``None`` = whenever a ``sentence_splitter`` is configured). The returned entity
        lists still line up with the input texts (union over a text's sentences).

        Processes texts in batch. Entity nodes that appear across multiple
        texts are shared (deduplicated by the backend upsert).

        When a causal_extractor is provided, each text is also analyzed for
        cause-effect relations. Cause and effect spans are added as text
        nodes with edges: cause_span -> original_text and
        effect_span -> original_text, plus cause_span -> effect_span.

        Args:
            texts: List of text strings to add.
            extractor: A callable(str) -> list[str] for entity extraction.
                Defaults to GlinerExtractor (gliner_small-v2.5) when `gliner` is
                installed, else GLiNER2Extractor (adds causal relations), else
                NERExtractor. Pass gliner_large-v2.5 for higher precision.
            causal_extractor: A callable(list[str]) -> list[dict] for
                cause-effect extraction. Each dict should have 'causal' (bool)
                and 'relations' (list of {'cause': str, 'effect': str}).
                Auto-enabled when the default GLiNER2Extractor is used.
            dedup_threshold: When set (e.g. 0.95), a text whose embedding cosine
                similarity to an existing fact is >= this value is treated as a
                near-duplicate: it is not added again, and any new ``scopes`` are
                unioned onto the existing fact. Prevents an evolving memory from
                accumulating paraphrased restatements. None (default) disables it.
                Dedup is against facts already in the graph, not within one batch.
            resolve_conflicts: When the graph has a ``conflict_resolver``, a new
                fact that contradicts existing ones soft-supersedes them (a
                "supersedes" edge drops them from default recall, keeping them
                auditable). None (default) resolves iff a resolver is configured;
                True requires one (raises otherwise); False skips resolution.
            canonicalizer: A callable(str) -> str (or an alias Mapping) applied to
                each extracted entity before its node is created, so surface
                variants collapse to one shared-entity bridge ("Apple Inc." /
                "Apple", "the Fed" / "Federal Reserve"). Per-call value overrides
                the graph default; None (default) uses the graph default (also
                None unless set on the constructor). Canonical entities are also
                what the returned entity lists contain.

        Returns:
            List of entity lists, one per input text (canonical forms when a
            canonicalizer is in effect). Skipped duplicates yield [].
        """
        use_split = self.sentence_splitter is not None if split is None else split
        if use_split:
            splitter = self.sentence_splitter
            if splitter is None:
                from reasongraph._split import RegexSplitter
                splitter = RegexSplitter()
            groups = [splitter.split(t) or [t] for t in texts]
            flat = [sent for g in groups for sent in g]
            per_fact = await self.add_texts(
                flat, extractor=extractor, causal_extractor=causal_extractor, scopes=scopes,
                causal=causal, dedup_threshold=dedup_threshold, dedup_scopes=dedup_scopes, dedup_entity_gate=dedup_entity_gate, resolve_conflicts=resolve_conflicts,
                canonicalizer=canonicalizer, split=False,
            )
            out: list[list[str]] = []
            k = 0
            for g in groups:
                seen: list[str] = []
                for _ in g:
                    for e in per_fact[k]:
                        if e not in seen:
                            seen.append(e)
                    k += 1
                out.append(seen)
            return out
        if extractor is None:
            if not hasattr(self, "_default_extractor"):
                self._default_extractor = self._build_default_extractor()
            extractor = self._default_extractor
        canonicalizer = self._resolve_canonicalizer(canonicalizer)

        # Causal extraction (the headline feature) runs by default. Resolution:
        # an explicit causal_extractor wins; else reuse the entity extractor's own
        # extract_causal when it has one (e.g. GLiNER2 -- no second model); else
        # fall back to the instance default causal extractor (the hybrid). Disable
        # per call with causal=False, or at construction with causal_extractor=False.
        # causal=True with nothing available raises, so the drop is never silent.
        disabled = causal is False or self._causal_extractor_arg is False
        if causal_extractor is None and not disabled:
            if hasattr(extractor, "extract_causal"):
                causal_extractor = extractor.extract_causal
            else:
                causal_extractor = self._resolve_causal_extractor()
        if causal is True and causal_extractor is None:
            raise ValueError(
                "causal=True but no causal extractor is available. Install "
                "reasongraph[gliner] (gliner>=0.2.27) or pass causal_extractor=... ."
            )

        # Semantic dedup (opt-in): drop texts that near-duplicate an existing
        # fact, unioning their scopes onto it instead of adding a paraphrase.
        skip: set[str] = set()
        pre_entities: dict[str, list] = {}   # extracted early for the gate, reused below
        if dedup_threshold is not None:
            for text in texts:
                if text in skip:
                    continue
                ents = None
                if dedup_entity_gate and extractor is not None:
                    ents = pre_entities[text] = list(extractor(text) or [])
                dup = await self._find_duplicate(text, dedup_threshold, scopes=dedup_scopes, entities=ents)
                if dup is not None:
                    skip.add(text)
                    if scopes:
                        await self.add_nodes([(dup, "text")], scopes=scopes)

        all_entities = []
        all_nodes = []
        all_edges = []
        active_texts = []

        # NER entity extraction (duplicates are skipped, yielding [] entities)
        batch_seen: set[str] = set()
        for text in texts:
            if text in skip:
                all_entities.append([])
                continue
            entities = pre_entities[text] if text in pre_entities else extractor(text)
            if canonicalizer is not None:
                entities = self._canonicalize(entities, canonicalizer)
            all_entities.append(entities)
            active_texts.append(text)
            all_nodes.append((text, "text"))
            for entity in entities:
                all_nodes.append((entity, "entity"))
                all_edges.append((entity, text))
            if self.link_contained_entities and entities:
                for other in await self._contained_entities(entities, batch_seen):
                    all_edges.append((other, text))
            batch_seen.update(entities)

        # Causal relation extraction (only on the non-duplicate texts)
        if causal_extractor is not None and active_texts:
            causal_results = causal_extractor(active_texts)
            for text, result in zip(active_texts, causal_results):
                if not result.get("causal"):
                    continue
                for rel in result.get("relations", []):
                    cause = rel.get("cause", "").strip()
                    effect = rel.get("effect", "").strip()
                    if not cause or not effect:
                        continue
                    # Add cause and effect as entity nodes so they serve as
                    # graph connectors but don't appear in query results.
                    all_nodes.append((cause, "entity"))
                    all_nodes.append((effect, "entity"))
                    # cause -> effect (typed causal link)
                    all_edges.append((cause, effect, "causes"))
                    # Both link back to the source sentence
                    all_edges.append((cause, text))
                    all_edges.append((effect, text))

        # A note that points back at the one before it states a cause the extractor cannot see:
        # it reads one sentence at a time, so "This broke checkout" has no visible cause. Link it
        # to the preceding fact, which is what the writer meant by "this".
        if self.resolve_back_references and len(active_texts) > 1:
            from reasongraph._backref import is_back_referenced_cause
            for prev, text in zip(active_texts, active_texts[1:]):
                if not is_back_referenced_cause(text):
                    continue
                if any(len(e) == 3 and e[2] == "causes" and e[1] in (text,) for e in all_edges):
                    continue
                all_edges.append((prev, text, "causes"))

        new_spans = [c for c, kind in all_nodes if kind == "entity"
                     and any(e[0] == c or e[1] == c for e in all_edges if len(e) == 3)]
        span_roles: dict[str, set[str]] = {}
        for e in all_edges:
            if len(e) == 3 and e[2] == "causes":
                span_roles.setdefault(e[0], set()).add("cause")
                span_roles.setdefault(e[1], set()).add("effect")
        if all_nodes:
            await self.add_nodes(all_nodes, scopes=scopes)
        if all_edges:
            await self.add_edges(all_edges)
        if self.span_link_threshold is not None and new_spans:
            await self._link_causal_spans(new_spans, self.span_link_threshold, span_roles)

        # Conflict resolution: soft-supersede existing facts the new ones contradict.
        do_resolve = (
            (self.conflict_resolver is not None and self.resolve_conflicts_by_default)
            if resolve_conflicts is None else resolve_conflicts
        )
        if do_resolve:
            if self.conflict_resolver is None:
                raise ValueError(
                    "resolve_conflicts=True but no conflict_resolver is configured. "
                    "Pass conflict_resolver=NLIConflictResolver() to ReasonGraph."
                )
            if active_texts:
                await self._resolve_conflicts(active_texts, scopes=scopes)

        return all_entities

    async def _neighbors(self, content: str, walk_scopes, embedding=None) -> list[dict]:
        """Neighbours for a walk step, capped at max_degree nearest to the query."""
        if embedding is not None:
            try:
                return await self.backend.nearest_neighbors(content, embedding, self.max_degree, walk_scopes)
            except NotImplementedError:
                pass
        neighbors = await self.backend.get_neighbors(content, walk_scopes)
        return neighbors[: self.max_degree] if len(neighbors) > self.max_degree else neighbors

    @contextlib.asynccontextmanager
    async def request_cache(self):
        """Memoise read-only metadata for the span of one request.

        One recall makes several discover/query passes over overlapping facts, and each pass asks the
        backend again for the same scopes, causal relations, timestamps and validity. Over a network
        those repeats are round trips. Inside this block each (kind, content) is fetched once; the
        cache is dropped at the end, so nothing is ever stale across requests. Reads only: writes and
        anything that changes facts are outside it.
        """
        outer = self._req_cache
        self._req_cache = {} if outer is None else outer
        try:
            yield
        finally:
            self._req_cache = outer

    async def _cached_map(self, kind: str, fetch, contents: list[str]) -> dict:
        """``fetch(missing) -> {content: value}``, served from the request cache when one is open."""
        cache = self._req_cache
        if cache is None:
            return await fetch(list(contents))
        store = cache.setdefault(kind, {})
        missing = [c for c in dict.fromkeys(contents) if c not in store]
        if missing:
            store.update(await fetch(missing))
            for c in missing:                      # a content the backend did not return: remember the gap
                store.setdefault(c, None)
        return {c: store[c] for c in contents if store.get(c) is not None}

    async def _scopes_of(self, contents) -> dict:
        return await self._cached_map("scopes", self.backend.get_scopes, list(contents))

    async def _causal_of(self, contents) -> dict:
        return await self._cached_map("causal", self.backend.get_causal_relations, list(contents))

    async def _created_of(self, contents) -> dict:
        return await self._cached_map("created", self.backend.get_created_at, list(contents))

    async def _validity_of(self, contents) -> dict:
        return await self._cached_map("validity", self.backend.get_validity, list(contents))

    async def _neighbors_many(self, contents: list[str], walk_scopes, embedding=None) -> dict[str, list[dict]]:
        """Neighbours for a whole walk level, capped per node, in one backend call where the
        backend supports it. A recall walks tens of nodes; per-node calls make recall latency a
        multiple of the round trip to the database."""
        if not contents:
            return {}
        if embedding is not None:
            try:
                return await self.backend.nearest_neighbors_many(contents, embedding, self.max_degree, walk_scopes)
            except NotImplementedError:
                pass
        got = await self.backend.get_neighbors_many(contents, walk_scopes)
        return {c: (ns[: self.max_degree] if len(ns) > self.max_degree else ns) for c, ns in got.items()}

    def _get_span_linker(self):
        """The span linker, loaded on first use.

        Deciding whether two spans describe the same event is a different job from finding relevant
        facts, so it can use a different model. Two shapes are accepted: a **cross-encoder**, scored
        pairwise against ``span_link_logit``, and a **bi-encoder**, which embeds each span once and
        compares by cosine against the ordinary span-link threshold. A bi-encoder is the cheaper
        shape (one pass per span, cached by the encoder) and is what a same-event similarity model
        trained on paraphrase pairs would be. Prefix a model id with ``bi:`` to load it as one.
        """
        if self._span_linker is not None or self._span_linker_spec is None:
            return self._span_linker
        spec = self._span_linker_spec
        if hasattr(spec, "predict") or hasattr(spec, "encode"):
            self._span_linker = spec
            return spec
        name = str(spec)
        if name.startswith("hf://"):
            name = name[len("hf://"):]
        bi = name.startswith("bi:")
        if bi:
            name = name[3:]
        try:
            if bi:
                from sentence_transformers import SentenceTransformer
                self._span_linker = SentenceTransformer(name)
            else:
                from sentence_transformers import CrossEncoder
                self._span_linker = CrossEncoder(name)
        except Exception as exc:  # a missing model must not break ingest: fall back to cosine
            warnings.warn(f"span linker {spec!r} could not be loaded ({exc}); using cosine linking")
            self._span_linker_spec = None
        return self._span_linker

    async def _link_causal_spans(self, spans: list[str], threshold: float, roles=None) -> None:
        """Tie each new causal span to existing causal spans it near-duplicates.

        Candidates come from vector search; only entity nodes that already take
        part in a ``causes`` edge qualify (a named entity like "Arizona" never
        becomes an alias of a span). The tie is an undirected ``same_as`` edge
        that :meth:`_causal_reach` crosses at no depth cost.

        With a span linker the tie is direction-aware: a hop continues where the
        effect of one relation is the cause of the next, so a new span is only
        compared with existing spans of the opposite role, scored as
        (effect, cause). Symmetric linking with the matcher tied same-role spans
        and reversed hops; measured: 48% causal chains symmetric vs 71%
        direction-aware vs 69% cosine.
        """
        edges: list[tuple] = []
        seen: set[tuple[str, str]] = set()
        linker = self._get_span_linker()
        roles = roles or {}
        for span in dict.fromkeys(spans):
            hits = await self.backend.knn_search(self.embeddings.encode(span), top_k=self.span_link_top_k or 6)
            others = [h["content"] for h in hits
                      if h.get("type") == "entity" and h["content"] != span]
            if not others:
                continue
            # the candidate's role comes from the direction of its causes edge
            other_roles: dict[str, set[str]] = {}
            for other in others:
                rs: set[str] = set()
                for n in await self.backend.get_neighbors(other):
                    if n.get("label") == "causes":
                        rs.add("cause" if n.get("direction") == "out" else "effect")
                if rs:
                    other_roles[other] = rs
            others = [o for o in others if o in other_roles]
            if not others:
                continue
            if linker is not None and hasattr(linker, "encode") and not hasattr(linker, "predict"):
                # A bi-encoder: its own cosine over the same direction-aware candidate set.
                my_roles = roles.get(span, set())
                cands = [o for o in others
                         if ("effect" in my_roles and "cause" in other_roles[o])
                         or ("cause" in my_roles and "effect" in other_roles[o])]
                if not cands:
                    continue
                floor = self.span_link_floor
                base = dict(zip(others, self.embeddings.score(span, others))) if floor is not None else {}
                try:
                    import numpy as _np
                    vecs = linker.encode([span, *cands])
                    q = _np.asarray(vecs[0], dtype=float)
                    qn = float(_np.linalg.norm(q)) or 1.0
                    scored = []
                    for o, v in zip(cands, vecs[1:]):
                        v = _np.asarray(v, dtype=float)
                        sim = float(q @ v / (qn * (float(_np.linalg.norm(v)) or 1.0)))
                        if floor is not None:
                            own = base.get(o, 0.0)
                            if own >= threshold:
                                sim = own              # cosine already agreed: keep its decision
                            elif own < floor:
                                sim = 0.0              # too far apart for the linker to be trusted
                        scored.append((o, sim))
                    if floor is not None:              # candidates cosine linked but the linker did not see
                        for o, own in base.items():
                            if own >= threshold and o not in dict(scored):
                                scored.append((o, own))
                    cut = threshold
                except Exception:
                    scored = list(zip(others, self.embeddings.score(span, others))); cut = threshold
            elif linker is not None:
                my_roles = roles.get(span, set())
                pairs: list[tuple[str, str, str]] = []      # (effect, cause, other)
                for other in others:
                    if "effect" in my_roles and "cause" in other_roles[other]:
                        pairs.append((span, other, other))
                    if "cause" in my_roles and "effect" in other_roles[other]:
                        pairs.append((other, span, other))
                if not pairs:
                    continue
                try:
                    logits = [float(x) for x in linker.predict([(e, c) for e, c, _ in pairs])]
                    scored = [(o, l) for (_, _, o), l in zip(pairs, logits)]
                    cut = self.span_link_logit
                except Exception:
                    scored = list(zip(others, self.embeddings.score(span, others))); cut = threshold
            else:
                scored = list(zip(others, self.embeddings.score(span, others))); cut = threshold
            for other, score in scored:
                if score < cut or (span, other) in seen or (other, span) in seen:
                    continue
                edges.append((span, other, "same_as"))
                seen.add((span, other))
        if edges:
            await self.add_edges(edges)

    async def _superseded(self, contents: list[str]) -> list[str]:
        """Return the subset of ``contents`` that have been retired (soft-superseded).

        A retired fact carries an ``invalid_at`` timestamp: it stays in the graph but
        drops out of default recall. This is a bounded validity lookup, not an edge
        scan, so it stays cheap on the query hot path.
        """
        validity = await self.backend.get_validity(contents)
        return [c for c in contents if validity.get(c) is not None]

    async def supersession_history(self, content: str) -> dict:
        """Audit trail for a fact from its ``"supersedes"`` edges.

        Returns ``{"supersedes": [...], "superseded_by": [...]}`` -- what this fact
        replaced, and what (if anything) has since replaced it. A non-empty
        ``superseded_by`` is why the fact is absent from default recall.
        """
        neighbors = await self.backend.get_neighbors(content)
        return {
            "supersedes": [n["content"] for n in neighbors
                           if n.get("label") == "supersedes" and n.get("direction") == "out"],
            "superseded_by": [n["content"] for n in neighbors
                              if n.get("label") == "supersedes" and n.get("direction") == "in"],
        }

    async def _resolve_conflicts(self, texts: list[str], scopes=None,
                                 candidates_k: int = 10) -> None:
        """Soft-supersede facts each new fact contradicts.

        Records a ``"supersedes"`` edge (provenance: which fact replaced which) and
        stamps the old fact's ``invalid_at`` (validity state: dropped from default
        recall, still time-travellable).

        Candidates are the ``candidates_k`` nearest facts that share at least one
        of the new fact's ``scopes`` (when given). Scoping matters in a shared
        graph: a tenant's write must never retire another tenant's fact.
        """
        batch = set(texts)
        scope_set = set(scopes) if scopes else None
        edges: list[tuple] = []
        retired: list[str] = []
        for text in texts:
            embedding = self.embeddings.encode(text)
            candidates = await self.backend.knn_search(embedding, top_k=candidates_k, scopes=scope_set)
            pool = [c["content"] for c in candidates
                    if c.get("type") == "text" and c["content"] not in batch]
            if not pool:
                continue
            already = set(await self._superseded(pool))
            pool = [c for c in pool if c not in already]
            if not pool:
                continue
            resolver = self.conflict_resolver
            if hasattr(resolver, "acontradictions"):
                found = await resolver.acontradictions(text, pool)
            else:  # sync resolvers (cross-encoder, blocking LLM call) run off the loop
                found = await asyncio.to_thread(resolver.contradictions, text, pool)
            for old in found:
                edges.append((text, old, "supersedes"))
                retired.append(old)
        if edges:
            await self.add_edges(edges)
        if retired:
            await self.backend.set_invalid(retired, datetime.now())

    async def _find_duplicate(self, text: str, threshold: float, scopes=None,
                              entities=None) -> str | None:
        """Return an existing text fact that near-duplicates ``text``, or None.

        Exact-content matches are left to the backend upsert (which unions
        scopes); only a distinct text node whose cosine similarity is >=
        ``threshold`` counts as a near-duplicate. ``scopes`` confines the search
        to nodes carrying one of those scopes (a tenant must never be merged
        into another tenant's wording). ``entities`` (the new text's entities)
        gates the merge: two sentences built on the same template score 0.9+
        even when they name different things ("Zurich ... strike for Thursday"
        vs "Frankfurt ... strike for Friday"), so the merge is refused unless
        every entity of the new text is already linked to the candidate.
        """
        embedding = self.embeddings.encode(text)
        candidates = await self.backend.knn_search(embedding, top_k=5, scopes=set(scopes) if scopes else None)
        hits = [c for c in candidates if c.get("type") == "text" and c["content"] != text]
        if not hits:
            return None
        if all(isinstance(c.get("score"), (int, float)) for c in hits):
            # The backend already computed the cosine similarity; no re-encoding.
            scores = [float(c["score"]) for c in hits]
        else:
            scores = self.embeddings.score(text, [c["content"] for c in hits])
        best = max(range(len(hits)), key=lambda i: scores[i])
        if scores[best] < threshold:
            return None
        dup = hits[best]["content"]
        if entities is not None:
            linked = {n["content"] for n in await self.backend.get_neighbors(dup) if n.get("type") == "entity"}
            if not set(entities) <= linked:
                return None
        return dup

    async def query(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 4,
        rerank_top_k: int = 4,
        search_mode: str = "embedding",
        rrf_k: int = 60,
        recency_weight: float = 0.0,
        scopes: set[str] | list[str] | None = None,
        isolate: bool | None = None,
        include_superseded: bool = False,
        as_of: datetime | None = None,
        walk_scopes: set[str] | list[str] | None = None,
    ) -> list[str]:
        """Query the graph with vector similarity and multi-hop traversal.

        Args:
            query: The search query text.
            top_k: Number of initial seeds from vector search.
            hops: Number of graph traversal hops.
            rerank_top_k: Number of results to keep after reranking at each hop.
            search_mode: 'embedding', 'keyword', or 'hybrid'.
            rrf_k: RRF smoothing constant for hybrid mode (default 60).
            recency_weight: In [0, 1]. When > 0, blends recency (by created_at)
                into reranking so newer facts outrank older contradicting ones.
                0 (default) leaves ranking unchanged.
            scopes: Optional free-text scope tags. When given, the initial
                seeds are drawn only from nodes carrying at least one of these
                scopes.
            walk_scopes: When given, the multi-hop walk is confined to nodes
                carrying at least one of these scopes, independently of
                ``scopes`` (which only picks the seeds). Lets a query seed from
                one session yet walk a wider boundary (e.g. one tenant's sessions
                but never another tenant's). Overrides ``isolate``.
            isolate: Controls whether traversal stays within ``scopes``. None
                (default) uses the graph's ``isolate_traversal`` setting. False
                lets the walk cross all scopes (shared-graph / cross-session
                discovery). True confines the walk to ``scopes`` so a query can
                only reach facts in its own tenant -- the multi-tenant setting.
                Has no effect without ``scopes``.
            include_superseded: When False (default) facts a newer fact has
                contradicted (soft-superseded) are dropped from the results. Set
                True to also return them. Only applies when a conflict_resolver is
                configured.
            as_of: Time-travel. When given, return only facts that were current at
                that moment -- created at or before ``as_of`` and not yet retired
                (``invalid_at`` after ``as_of``). Overrides ``include_superseded``.

        Returns:
            List of text-type node contents in relevance order.
        """
        if search_mode not in ("embedding", "keyword", "hybrid"):
            raise ValueError(f"search_mode must be 'embedding', 'keyword', or 'hybrid', got '{search_mode}'")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError(f"recency_weight must be in [0, 1], got {recency_weight}")

        scope_set = set(scopes) if scopes else None
        isolate = self.isolate_traversal if isolate is None else isolate
        walk_scopes = (
            set(walk_scopes) if walk_scopes
            else (scope_set if (isolate and scope_set) else None)
        )
        embedding = self.embeddings.encode_query(query)

        if search_mode == "embedding":
            seeds = await self.backend.knn_search(embedding, top_k, scopes=scope_set)
        elif search_mode == "keyword":
            seeds = await self.backend.hybrid_search(
                embedding, query, top_k, keyword_only=True, scopes=scope_set,
            )
        else:
            seeds = await self.backend.hybrid_search(
                embedding, query, top_k, rrf_k=rrf_k, scopes=scope_set,
            )

        visited: set[str] = set()
        results: list[dict[str, str]] = []

        for _ in range(hops):
            # Separate entity nodes from text nodes.  Entity nodes serve as
            # graph bridges -- we always traverse their edges -- but they
            # should not compete with text nodes for rerank budget.
            text_seeds = [s for s in seeds if s.get("type") == "text"]
            entity_seeds = [s for s in seeds if s.get("type") != "text"]

            # For recency-weighted ranking, attach created_at to the text seeds
            # before reranking (fetched only when the feature is enabled).
            if recency_weight > 0 and text_seeds:
                created = await self.backend.get_created_at(
                    [s["content"] for s in text_seeds]
                )
                for s in text_seeds:
                    s.setdefault("created_at", created.get(s["content"]))

            # Split text seeds by provenance: chain continuations (from
            # text->text edges) get priority access to the rerank budget,
            # bridge discoveries (from entity->text edges) fill remaining
            # slots.  On the first hop there is no provenance tag, so all
            # seeds go into the chain pool (they came from the initial
            # vector search, not from entity bridges).
            chain_pool = [s for s in text_seeds if s.get("_source", "chain") == "chain"]
            bridge_pool = [s for s in text_seeds if s.get("_source") == "bridge"]

            # One reranker call for both pools (scores are per query/doc pair, so
            # the order within each pool is the same as ranking them separately);
            # then apply the budget: chain continuations first, bridges fill up.
            pooled = self.embeddings.rerank(
                query, chain_pool + bridge_pool, len(chain_pool) + len(bridge_pool), recency_weight,
            )
            bridge_contents = {s["content"] for s in bridge_pool} - {s["content"] for s in chain_pool}
            ranked_chain = [s for s in pooled if s["content"] not in bridge_contents][:rerank_top_k]
            remaining_budget = max(0, rerank_top_k - len(ranked_chain))
            ranked_bridge = [s for s in pooled if s["content"] in bridge_contents][:remaining_budget]
            ranked = ranked_chain + ranked_bridge

            chain_next: list[dict[str, str]] = []
            entity_next: list[dict[str, str]] = []

            for seed in ranked:
                if seed["content"] in visited:
                    continue
                results.append(seed)
                visited.add(seed["content"])
                neighbors = await self._neighbors(seed["content"], walk_scopes, embedding)
                for n in neighbors:
                    if n["content"] not in visited:
                        if n["type"] == "text":
                            n["_source"] = "chain"
                            chain_next.append(n)
                        else:
                            entity_next.append(n)

            # Traverse entity edges transparently (no rerank cost)
            bridge_next: list[dict[str, str]] = []
            for seed in entity_seeds:
                if seed["content"] in visited:
                    continue
                visited.add(seed["content"])
                neighbors = await self._neighbors(seed["content"], walk_scopes, embedding)
                for n in neighbors:
                    if n["content"] not in visited:
                        if n["type"] == "text":
                            n["_source"] = "bridge"
                            bridge_next.append(n)
                        else:
                            entity_next.append(n)

            seeds = chain_next + bridge_next + entity_next

        texts = [node["content"] for node in results if node["type"] == "text"]
        if as_of is not None and texts:
            # Time-travel: keep only facts current at ``as_of``. Stored timestamps
            # are naive, so coerce an aware ``as_of`` to naive local time.
            if as_of.tzinfo is not None:
                as_of = as_of.astimezone().replace(tzinfo=None)
            created = await self._created_of(texts)
            validity = await self._validity_of(texts)
            texts = [t for t in texts if self._valid_at(t, as_of, created, validity)]
        elif not include_superseded and texts and self.conflict_resolver is not None:
            # Drop retired (soft-superseded) facts. Only a resolver retires facts,
            # so skip the validity lookup entirely when none is configured.
            superseded = set(await self._superseded(texts))
            texts = [t for t in texts if t not in superseded]
        return texts

    @staticmethod
    def _valid_at(content, as_of, created, validity) -> bool:
        cr = created.get(content)
        iv = validity.get(content)
        born = cr is None or datetime.fromisoformat(cr) <= as_of
        retired = iv is not None and datetime.fromisoformat(iv) <= as_of
        return born and not retired

    async def query_detailed(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 4,
        rerank_top_k: int = 4,
        search_mode: str = "embedding",
        rrf_k: int = 60,
        recency_weight: float = 0.0,
        scopes: set[str] | list[str] | None = None,
        isolate: bool | None = None,
        include_superseded: bool = False,
        as_of: datetime | None = None,
        walk_scopes: set[str] | list[str] | None = None,
    ) -> list[dict]:
        """Like :meth:`query` but return structured results instead of bare strings.

        Each result is ``{"content": str, "score": float, "created_at": str | None,
        "scopes": [str]}`` in the same relevance order as :meth:`query`. ``score``
        is the embedding cosine similarity to the query (in [-1, 1]), so callers
        can threshold on confidence, dedupe by content, budget tokens, or show
        "remembered on <date>" -- the things a bare ``list[str]`` cannot support.
        """
        contents = await self.query(
            query, top_k=top_k, hops=hops, rerank_top_k=rerank_top_k,
            search_mode=search_mode, rrf_k=rrf_k, recency_weight=recency_weight,
            scopes=scopes, isolate=isolate, include_superseded=include_superseded,
            as_of=as_of, walk_scopes=walk_scopes,
        )
        if not contents:
            return []
        created = await self._created_of(contents)
        scope_map = await self._scopes_of(contents)
        scores = self.embeddings.score(query, contents)
        return [
            {
                "content": content,
                "score": float(score),
                "created_at": created.get(content),
                "scopes": sorted(scope_map.get(content, set())),
            }
            for content, score in zip(contents, scores)
        ]

    async def discover(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 4,
        search_mode: str = "embedding",
        rrf_k: int = 60,
        scopes: set[str] | list[str] | None = None,
        max_results: int = 10,
        max_visited: int = 1000,
        isolate: bool | None = None,
        include_superseded: bool = False,
        walk_scopes: set[str] | list[str] | None = None,
        causal_hops: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Discover connection paths from a query into the graph.

        Like :meth:`query`, seeds are drawn from ``scopes`` (a knowledge
        session) and, by default, traversal crosses all scopes (that is the
        point of discovery). Set ``isolate=True`` (or the graph default) to
        confine the walk to ``scopes`` for multi-tenant safety. Unlike ``query``,
        this
        returns *how* each reached fact connects back to a seed: an alternating
        chain of facts and the entities that bridge them, each fact tagged with
        its scopes. A fact whose scopes do not overlap the query scope is a
        cross-session discovery (``cross_session=True``).

        Scales to large graphs: the breadth-first walk stops after
        ``max_visited`` nodes (bounding hub-entity blowup), scopes are fetched
        only for the discovered facts (not the whole graph), and when more
        facts are discovered than ``max_results`` they are reranked by relevance
        to the query so the most relevant connections are kept; smaller result
        sets stay in discovery-distance order (nearest first).

        Returns a list of dicts::

            {"content": str, "scopes": [str], "cross_session": bool,
             "causes": [{"cause": str, "effect": str}, ...],
             "path": [{"content": str, "scopes": [str]} | {"entity": str}, ...]}

        ``causes`` lists the directed cause->effect relations the fact asserts
        (from typed ``"causes"`` edges), surfacing causality first-class.

        ``causal_hops`` gives the walk separate budgets along the causal direction:
        ``(forward, backward)``, forward following cause->effect (consequences) and backward
        following effect->cause (what led here). Entity bridges are unaffected and total depth
        is still ``hops``. ``None``, the default, leaves the walk symmetric. "Why" is a backward
        question, so spending depth differently in each direction can reach a root cause a
        symmetric walk does not; which split wins is empirical, so measure before changing it.
        """
        scope_set = set(scopes) if scopes else None
        isolate = self.isolate_traversal if isolate is None else isolate
        walk_scopes = (
            set(walk_scopes) if walk_scopes
            else (scope_set if (isolate and scope_set) else None)
        )
        embedding = self.embeddings.encode_query(query)
        # Entity nodes share the index with facts and often outrank them for short
        # queries, so fetch a wider window and keep the first ``top_k`` facts; otherwise
        # a question could seed from one fact and return a single connection.
        fetch = max(top_k * 4, 20)
        if search_mode == "embedding":
            seeds = await self.backend.knn_search(embedding, fetch, scopes=scope_set)
        elif search_mode == "keyword":
            seeds = await self.backend.hybrid_search(
                embedding, query, fetch, keyword_only=True, scopes=scope_set,
            )
        elif search_mode == "hybrid":
            seeds = await self.backend.hybrid_search(
                embedding, query, fetch, rrf_k=rrf_k, scopes=scope_set,
            )
        else:
            raise ValueError(
                f"search_mode must be 'embedding', 'keyword', or 'hybrid', got '{search_mode}'"
            )

        # Seed only from text facts so every connection path is rooted at a fact
        # (entities bridge during traversal, they are not path roots).
        seeds = [s for s in seeds if s.get("type") == "text"][:top_k]

        # Breadth-first traversal tracking, for every node, the fact and entity
        # it was reached through. parent[c] = (prior_fact, bridging_entity, depth);
        # seeds have (None, None, 0). By default get_neighbors is unscoped, so the
        # walk crosses knowledge sessions; walk_scopes confines it when isolating.
        fwd_cap, back_cap = causal_hops if causal_hops else (None, None)
        spent: dict[str, tuple[int, int]] = {}   # (forward, backward) causal steps used to reach a node
        visited: set[str] = set()
        parent: dict[str, tuple] = {}
        order: list[str] = []  # discovered text facts, in BFS order
        frontier: deque = deque()
        for s in seeds:
            c = s["content"]
            if c in visited:
                continue
            visited.add(c)
            spent[c] = (0, 0)
            parent[c] = (None, None, 0)
            frontier.append((c, s.get("type", "text"), 0))
            if s.get("type") == "text":
                order.append(c)

        while frontier and len(visited) < max_visited:
            # Expand a whole level at once: one backend call per level, not per node.
            level = [frontier.popleft() for _ in range(len(frontier))]
            level = [(c, t, d) for (c, t, d) in level if d < hops]
            if not level:
                continue
            fetched = await self._neighbors_many([c for c, _, _ in level], walk_scopes, embedding)
            for content, ntype, depth in level:
                if len(visited) >= max_visited:
                    break
                for n in fetched.get(content, []):
                  if len(visited) >= max_visited:
                      break
                  nc, nt = n["content"], n["type"]
                  if nc in visited:
                      continue
                  used_f, used_b = spent.get(content, (0, 0))
                  if causal_hops and n.get("label") == "causes":
                      # 'out' means this node -> the neighbour, i.e. along cause->effect.
                      if n.get("direction") == "out":
                          if fwd_cap is not None and used_f >= fwd_cap:
                              continue
                          used_f += 1
                      else:
                          if back_cap is not None and used_b >= back_cap:
                              continue
                          used_b += 1
                  spent[nc] = (used_f, used_b)
                  visited.add(nc)
                  if ntype == "text" and nt == "entity":
                      # An entity bridge leaving this fact.
                      parent[nc] = (content, None, depth + 1)
                      frontier.append((nc, "entity", depth + 1))
                  elif ntype == "entity" and nt == "text":
                      # A fact reached through this entity; bridge back to the fact
                      # that led into the entity.
                      prior_fact = parent[content][0]
                      parent[nc] = (prior_fact, content, depth + 1)
                      frontier.append((nc, "text", depth + 1))
                      order.append(nc)
                  else:
                      # text->text (direct) or entity->entity (e.g. cause->effect) edge.
                      prior = content if nt == "text" else parent.get(content, (None,))[0]
                      parent[nc] = (prior, None, depth + 1)
                      frontier.append((nc, nt, depth + 1))
                      if nt == "text":
                          order.append(nc)

        # Scope lookup for the discovered facts only -- a bounded fetch keyed by
        # the reached contents, so it scales with the result set, not the graph.
        # Drop retired (soft-superseded) facts. Only a resolver retires facts, so
        # skip the validity lookup when none is configured.
        if not include_superseded and order and self.conflict_resolver is not None:
            superseded = set(await self._superseded(order))
            order = [c for c in order if c not in superseded]

        scope_map = await self._scopes_of(order)

        def reconstruct(content: str) -> list[dict]:
            steps: list[dict] = []
            cur = content
            guard = 0
            while cur is not None and guard < 128:
                guard += 1
                prior_fact, bridging_entity, _ = parent.get(cur, (None, None, 0))
                steps.append({"content": cur, "scopes": sorted(scope_map.get(cur, set()))})
                if bridging_entity:
                    steps.append({"entity": bridging_entity})
                cur = prior_fact
            return list(reversed(steps))

        # Causal relations each discovered fact asserts (bounded fetch keyed by
        # the reached facts), so causality is surfaced first-class in the output.
        causal_map = await self._causal_of(order)

        candidates: list[dict] = []
        for content in order:
            node_scopes = scope_map.get(content, set())
            candidates.append({
                "content": content,
                "scopes": sorted(node_scopes),
                "cross_session": bool(scope_set) and not (node_scopes & scope_set),
                "causes": causal_map.get(content, []),
                "path": reconstruct(content),
            })

        # When more facts were discovered than we return, rerank by relevance to
        # the query and keep the most relevant; otherwise the discovery-distance
        # order (nearest first) already fits and needs no cross-encoder.
        if len(candidates) > max_results:
            return self.embeddings.rerank(query, candidates, max_results)
        return candidates

    # -- causal chain tracing (walk the directed "causes" DAG) --

    async def _resolve_fact(self, content: str, scopes: set[str] | None = None) -> str | None:
        """Resolve free text to an existing text fact: exact match else nearest."""
        hits = await self.backend.knn_search(
            self.embeddings.encode(content), top_k=5, scopes=scopes
        )
        for h in hits:
            if h["content"] == content and h.get("type") == "text":
                return content
        for h in hits:
            if h.get("type") == "text":
                return h["content"]
        return None

    async def _causal_reach(
        self, start: list[str], edge_dir: str, *,
        blocked_edges: frozenset = frozenset(), max_depth: int,
        walk_scopes: set[str] | None, max_visited: int, bridge: bool = False,
    ) -> tuple[list[dict], set[str], set[str], set[str]]:
        """BFS over ``causes`` edges from ``start`` spans in ``edge_dir`` (``'out'``
        forward / ``'in'`` backward).

        Returns ``(chain, reached_facts, reached_targets, has_next)`` where ``chain``
        is one ``{cause, effect, depth}`` dict per traversed hop. ``blocked_edges``
        is a set of ``(span, other)`` traversal pairs that are neither recorded nor
        followed -- used by ``what_if`` for counterfactual pruning; empty (the
        default) reproduces the plain walk exactly.
        """
        visited_spans: set[str] = set(start)
        frontier: deque = deque((s, 0) for s in start)
        chain: list[dict] = []
        reached_facts: set[str] = set()
        reached_targets: set[str] = set()
        has_next: set[str] = set()

        while frontier and len(visited_spans) <= max_visited:
            span, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for n in await self.backend.get_neighbors(span, walk_scopes):
                if n.get("type") == "text":
                    reached_facts.add(n["content"])
                    continue
                if n.get("label") == "same_as":
                    # An alias of this span (same event, different wording): continue
                    # the walk from it at the same depth without recording a hop.
                    alias = n["content"]
                    if alias not in visited_spans and len(visited_spans) < max_visited:
                        visited_spans.add(alias)
                        frontier.append((alias, depth))
                    continue
                if n.get("label") != "causes" or n.get("direction") != edge_dir:
                    continue
                other = n["content"]
                if (span, other) in blocked_edges:
                    continue
                has_next.add(span)
                reached_targets.add(other)
                cause, effect = (span, other) if edge_dir == "out" else (other, span)
                chain.append({"cause": cause, "effect": effect, "depth": depth})
                if other not in visited_spans and len(visited_spans) < max_visited:
                    visited_spans.add(other)
                    frontier.append((other, depth + 1))
                    if bridge and edge_dir == "out":
                        # Same event, different wording, no same_as edge: continue from
                        # cause spans that share a content word with this effect.
                        for alias in await self._lexical_bridges(other, exclude=visited_spans):
                            if alias not in visited_spans and len(visited_spans) < max_visited:
                                visited_spans.add(alias)
                                frontier.append((alias, depth + 1))

        return chain, reached_facts, reached_targets, has_next

    async def _trace(
        self, content: str, direction: str, *, max_depth: int = 6,
        scopes=None, isolate: bool | None = None, include_superseded: bool = False,
        max_visited: int = 1000, walk_scopes=None, extra_start: list[str] | None = None,
        bridge: bool = False,
    ) -> dict:
        """Directional walk over ``causes`` edges from the fact nearest ``content``.

        ``direction='effects'`` walks forward (what this fact caused downstream);
        ``'causes'`` walks backward (what led to it). Returns ``origin`` (the seed
        fact), ``chain`` (one dict per causal hop in BFS-depth order, each resolved
        to the fact that asserted it, with scopes and a cross_session flag), and
        ``terminals`` (the leaf effects, or the root causes).
        """
        scope_set = set(scopes) if scopes else None
        resolved_isolate = self.isolate_traversal if isolate is None else isolate
        walk_scopes = (
            set(walk_scopes) if walk_scopes
            else (scope_set if (resolved_isolate and scope_set) else None)
        )

        origin = await self._resolve_fact(content, scope_set)
        if origin is None:
            return {"origin": None, "chain": [], "terminals": []}

        pairs = (await self._causal_of([origin])).get(origin, [])
        edge_dir = "out" if direction == "effects" else "in"
        start = [p["cause"] for p in pairs] if direction == "effects" else [p["effect"] for p in pairs]
        if extra_start:
            start = list(dict.fromkeys(start + [x for x in extra_start]))

        chain, reached_facts, reached_targets, has_next = await self._causal_reach(
            start, edge_dir, max_depth=max_depth, walk_scopes=walk_scopes,
            max_visited=max_visited, bridge=bridge,
        )
        terminals = sorted(reached_targets - has_next)

        # Map each hop to the fact that asserted it, and tag scopes / retirement.
        causal_map = await self._causal_of(sorted(reached_facts))
        reverse: dict[tuple, str] = {}
        for fact, prs in causal_map.items():
            for p in prs:
                reverse[(p["cause"], p["effect"])] = fact
        hop_facts = [reverse.get((h["cause"], h["effect"])) for h in chain]
        present = sorted({f for f in hop_facts if f})
        retired: set[str] = set()
        if not include_superseded and present and self.conflict_resolver is not None:
            retired = set(await self._superseded(present))
        scope_map = await self._scopes_of(present)

        out_chain = []
        for hop, fact in zip(chain, hop_facts):
            if fact in retired:
                continue
            node_scopes = scope_map.get(fact, set()) if fact else set()
            out_chain.append({
                "cause": hop["cause"], "effect": hop["effect"], "fact": fact,
                "depth": hop["depth"], "scopes": sorted(node_scopes),
                "cross_session": bool(scope_set) and not (node_scopes & scope_set),
            })
        return {"origin": origin, "chain": out_chain, "terminals": terminals}

    async def trace_effects(self, content: str, *, max_depth: int = 6, scopes=None,
                            isolate: bool | None = None, include_superseded: bool = False,
                            max_visited: int = 1000, walk_scopes=None) -> dict:
        """Forward causal walk: what the fact nearest ``content`` caused downstream."""
        return await self._trace(content, "effects", max_depth=max_depth, scopes=scopes,
                                  isolate=isolate, include_superseded=include_superseded,
                                  max_visited=max_visited, walk_scopes=walk_scopes)

    async def trace_causes(self, content: str, *, max_depth: int = 6, scopes=None,
                           isolate: bool | None = None, include_superseded: bool = False,
                           max_visited: int = 1000, walk_scopes=None) -> dict:
        """Backward causal walk: what led to the fact nearest ``content``."""
        return await self._trace(content, "causes", max_depth=max_depth, scopes=scopes,
                                 isolate=isolate, include_superseded=include_superseded,
                                 max_visited=max_visited, walk_scopes=walk_scopes)

    async def root_causes(self, content: str, **kwargs) -> list[str]:
        """The root cause spans behind ``content`` (backward-walk terminals)."""
        return (await self.trace_causes(content, **kwargs))["terminals"]

    async def _entity_names(self, fact: str) -> list[str]:
        """Entity-type neighbours of a fact (extracted entities and causal spans)."""
        names = []
        for n in await self.backend.get_neighbors(fact):
            if n.get("type") == "entity" and len(n["content"]) >= 3:
                names.append(n["content"])
        return names

    _BRIDGE_STOP = frozenset("""the a an and or but of to in on at for with by from as is are was were be been
    being this that these those it its into over under than then there their they them his her our your
    which while when where who whom whose what because since after before during also very more most
    such into onto about against between through above below each other some any all both few many
    much no nor not only own same so too can will just don should now der die das und oder den dem des
    ein eine einer eines einem einen ist sind war waren wird werden nicht auch mit von zu auf für aus
    bei nach über unter het een van en dat dit die deze niet ook met voor door naar bij uit over el la
    los las un una unos unas y o de del al que en con por para sin sobre es son fue fueron le les des
    du un une et ou que qui dans sur avec pour par sans est sont ve bir bu şu o ile için gibi da de
    ama veya değil""".split())

    bridge_min_score: float = 0.45

    @classmethod
    def _bridge_tokens(cls, text: str) -> set[str]:
        """Crude language-agnostic content tokens: lowercase words of 4+ letters minus
        stopwords, cut to a 5-char prefix so ``resolved``/``resolution`` match."""
        out = set()
        for w in re.findall(r"[^\W\d_]+", text):
            low = w.lower()
            if low in cls._BRIDGE_STOP:
                continue
            if len(low) >= 4 or (len(low) == 3 and w.isupper()):   # keep acronyms: UDP, CPU, GPU
                out.add(low[:5])
        return out

    async def _lexical_bridges(self, text: str, *, exclude: set[str] = frozenset(),
                               top_k: int = 15) -> list[str]:
        """Cause spans (of any fact) that share a content word with ``text`` and are
        semantically close to it. Used by :meth:`causal_chain` to continue a walk
        across facts that describe one event with different span boundaries, e.g.
        effect "the system throttles performance" -> cause "Throttling performance",
        or a plain root fact "CPU temperature exceeded 85°C" -> cause "the CPU
        temperature rises". Paraphrases without a shared word still need ``same_as``."""
        toks = self._bridge_tokens(text)
        if not toks:
            return []
        hits = await self.backend.knn_search(self.embeddings.encode(text), top_k=top_k)
        cands = [h for h in hits if h.get("type") == "entity" and h["content"] != text
                 and h["content"] not in exclude]
        if not cands:
            return []
        scores = self.embeddings.score(text, [h["content"] for h in cands])
        out: list[str] = []
        for h, score in zip(cands, scores):
            cand = h["content"]
            if score < self.bridge_min_score or not (toks & self._bridge_tokens(cand)):
                continue
            neighbours = await self.backend.get_neighbors(cand)
            if any(n.get("label") == "causes" and n.get("direction") == "out" for n in neighbours):
                out.append(cand)
        return out

    async def causal_chain(self, from_content: str, to_content: str, *, max_depth: int = 6,
                           scopes=None, isolate: bool | None = None,
                           include_superseded: bool = False, walk_scopes=None) -> list[dict] | None:
        """Directed causal hops linking ``from_content`` to ``to_content``, or None.

        Returns the ordered list of causal hops (as in ``trace_effects``' chain) if
        the fact nearest ``from_content`` causally leads to the one nearest
        ``to_content``; otherwise None.

        Facts rarely repeat each other's wording, so two bridges are applied:
        the walk also continues from cause spans (of other facts) that share a
        content word with the origin fact or with a reached effect span (see
        :meth:`_lexical_bridges`), and it counts as arrived when a hop's effect is
        a target span, a ``same_as`` alias of one, or mentions one of the target
        fact's entities.
        """
        scope_set = set(scopes) if scopes else None
        target = await self._resolve_fact(to_content, scope_set)
        if target is None:
            return None
        target_pairs = (await self._causal_of([target])).get(target, [])
        target_spans = {p["cause"] for p in target_pairs} | {p["effect"] for p in target_pairs}
        target_names = [n for n in await self._entity_names(target) if n not in target_spans]
        if not target_spans and not target_names:
            return None
        origin = await self._resolve_fact(from_content, scope_set)
        if origin is None:
            return None
        own = {p["cause"] for p in (await self._causal_of([origin])).get(origin, [])}
        own |= {p["effect"] for p in (await self._causal_of([origin])).get(origin, [])}
        seeds = await self._lexical_bridges(origin, exclude=own)
        traced = await self._trace(
            origin, "effects", max_depth=max_depth, scopes=scopes, isolate=isolate,
            include_superseded=include_superseded, walk_scopes=walk_scopes, extra_start=seeds,
            bridge=True,
        )
        # A chain exists if the forward walk *arrived at* the target fact: either it
        # traversed the target's own hop (its cause span was reached, so the hop
        # belongs to the target), or some hop's effect is a target span, an alias of
        # one, or names one of the target's entities. Checking a hop's cause alone
        # would count the origin's own spans (a fact whose cause is the target's
        # effect would wrongly "lead to" it).
        target_effects = {p["effect"] for p in target_pairs}
        for i, hop in enumerate(traced["chain"]):
            if hop.get("fact") == target:
                return traced["chain"][: i + 1]
            effect = hop["effect"]
            arrived = effect in target_effects
            if not arrived:
                aliases = {n["content"] for n in await self.backend.get_neighbors(effect)
                           if n.get("label") == "same_as"}
                arrived = bool(aliases & target_effects)
            if not arrived and target_names:
                low = effect.lower()
                arrived = any(name.lower() in low for name in target_names)
            if arrived:
                return traced["chain"][: i + 1]
        return None

    async def what_if(
        self, content: str, *, origin: str | None = None, direction: str = "effects",
        max_depth: int = 6, scopes=None, isolate: bool | None = None,
        include_superseded: bool = False, max_visited: int = 1000, walk_scopes=None,
    ) -> dict:
        """Counterfactual: if the fact nearest ``content`` were false, which downstream
        effects would collapse?

        Hypothetically prunes the resolved fact (no graph mutation) and re-runs the
        causal reachability walk from ``origin`` -- the pruned fact itself by default,
        or a distinct upstream fact if given. Only edges the pruned fact *solely*
        supports are removed; an edge another live fact also asserts stays, so its
        downstream survives. ``direction='effects'`` (default) walks forward;
        ``'causes'`` mirrors upstream (which cause spans become orphaned).

        Returns ``pruned`` (the resolved fact, echoed so the caller can verify what was
        pruned), ``origin``, ``pruned_edges`` (the solely-supported causal edges that
        were removed, as ``{cause, effect}``), ``collapsed`` (spans that lost all
        causal support -- each cited to a now-unsupported ``fact`` with ``scopes``,
        ``depth`` and a ``cross_session`` flag) and ``survived`` (spans directly
        downstream of a removed edge that a live alternate path still reaches).

        Superseded-supporter exclusion requires a configured ``conflict_resolver``
        (same convention as ``trace_effects``); without one, a retired duplicate
        assertion still counts as live support.
        """
        empty = {"pruned": None, "origin": None, "pruned_edges": [],
                 "collapsed": [], "survived": []}
        scope_set = set(scopes) if scopes else None
        resolved_isolate = self.isolate_traversal if isolate is None else isolate
        walk_scopes = (
            set(walk_scopes) if walk_scopes
            else (scope_set if (resolved_isolate and scope_set) else None)
        )

        pruned = await self._resolve_fact(content, scope_set)
        if pruned is None:
            return dict(empty)
        origin_fact = pruned if origin is None else await self._resolve_fact(origin, scope_set)
        if origin_fact is None:
            return {**empty, "pruned": pruned}

        edge_dir = "out" if direction == "effects" else "in"
        down_key = "effect" if direction == "effects" else "cause"

        def _oriented(pair):
            # (cause, effect) -> (span, other) traversal pair matching _causal_reach.
            return (pair["cause"], pair["effect"]) if direction == "effects" \
                else (pair["effect"], pair["cause"])

        def _to_ce(pair):
            # (span, other) traversal pair -> {cause, effect} for output.
            return {"cause": pair[0], "effect": pair[1]} if direction == "effects" \
                else {"cause": pair[1], "effect": pair[0]}

        pruned_pairs = (await self._causal_of([pruned])).get(pruned, [])
        pruned_oriented = {_oriented(p) for p in pruned_pairs}

        origin_pairs = (await self._causal_of([origin_fact])).get(origin_fact, [])
        start = ([p["cause"] for p in origin_pairs] if direction == "effects"
                 else [p["effect"] for p in origin_pairs])

        base_chain, base_facts, _, _ = await self._causal_reach(
            start, edge_dir, max_depth=max_depth, walk_scopes=walk_scopes,
            max_visited=max_visited,
        )
        baseline_spans = {h[down_key] for h in base_chain}

        if not pruned_oriented:
            return {"pruned": pruned, "origin": origin_fact, "pruned_edges": [],
                    "collapsed": [], "survived": []}

        # Sound support map: gather every live fact that asserts a pruned edge from the
        # pruned fact's OWN span neighborhoods, not the origin walk -- so an edge a
        # disjoint fact also asserts is never falsely reported as solely-pruned.
        span_pool: set[str] = {pruned}
        for span in {s for pair in pruned_oriented for s in pair}:
            for n in await self.backend.get_neighbors(span, walk_scopes):
                if n.get("type") == "text":
                    span_pool.add(n["content"])
        support_rels = await self._causal_of(sorted(span_pool))
        retired: set[str] = set()
        if not include_superseded and self.conflict_resolver is not None:
            retired = set(await self._superseded(sorted(span_pool)))
        support: dict[tuple, set[str]] = {}
        for fact, prs in support_rels.items():
            if fact in retired:
                continue
            for p in prs:
                support.setdefault(_oriented(p), set()).add(fact)

        solely_pruned = frozenset(
            e for e in pruned_oriented if support.get(e, set()) <= {pruned}
        )
        pruned_edges = [_to_ce(e) for e in sorted(solely_pruned)]
        if not solely_pruned:
            return {"pruned": pruned, "origin": origin_fact, "pruned_edges": [],
                    "collapsed": [], "survived": []}

        cf_chain, _, _, _ = await self._causal_reach(
            start, edge_dir, blocked_edges=solely_pruned, max_depth=max_depth,
            walk_scopes=walk_scopes, max_visited=max_visited,
        )
        cf_spans = {h[down_key] for h in cf_chain}

        collapsed_spans = baseline_spans - cf_spans
        directly_threatened = {other for (_span, other) in solely_pruned}
        survived = sorted(directly_threatened & cf_spans)

        # Cite each collapsed span to a fact that asserted its shallowest baseline hop.
        cite_rels = await self._causal_of(sorted(base_facts))
        reverse: dict[tuple, str] = {}
        for fact, prs in cite_rels.items():
            for p in prs:
                reverse[(p["cause"], p["effect"])] = fact
        by_span: dict[str, dict] = {}
        for h in base_chain:
            s = h[down_key]
            if s in collapsed_spans and (s not in by_span or h["depth"] < by_span[s]["depth"]):
                by_span[s] = h
        cited = sorted({reverse.get((h["cause"], h["effect"])) for h in by_span.values()} - {None})
        scope_map = await self._scopes_of(cited)

        collapsed = []
        for span in sorted(collapsed_spans):
            hop = by_span.get(span)
            fact = reverse.get((hop["cause"], hop["effect"])) if hop else None
            node_scopes = scope_map.get(fact, set()) if fact else set()
            collapsed.append({
                "span": span, "fact": fact, "depth": hop["depth"] if hop else 0,
                "scopes": sorted(node_scopes),
                "cross_session": bool(scope_set) and not (node_scopes & scope_set),
            })

        return {"pruned": pruned, "origin": origin_fact, "pruned_edges": pruned_edges,
                "collapsed": collapsed, "survived": survived}

    async def answer(
        self,
        query: str,
        *,
        use_discover: bool = True,
        top_k: int = 5,
        hops: int = 4,
        search_mode: str = "embedding",
        scopes: set[str] | list[str] | None = None,
        max_results: int = 10,
        walk_scopes: set[str] | list[str] | None = None,
    ) -> str:
        """Answer a query in logical free text via the configured synthesizer.

        Retrieves supporting facts -- with their connection paths when
        ``use_discover`` is True -- then rephrases them into a natural-language
        answer using the pluggable ``synthesizer`` (bring your own small model).
        Raises if no synthesizer was configured on the graph.
        """
        if self._synthesizer is None:
            raise RuntimeError(
                "No synthesizer configured. Pass synthesizer=<callable> to ReasonGraph()."
            )
        if use_discover:
            context = await self.discover(
                query, top_k=top_k, hops=hops, search_mode=search_mode,
                scopes=scopes, max_results=max_results, walk_scopes=walk_scopes,
            )
        else:
            facts = await self.query(
                query, top_k=top_k, hops=hops, search_mode=search_mode, scopes=scopes,
                walk_scopes=walk_scopes,
            )
            context = [{"content": f, "path": [{"content": f}]} for f in facts]

        result = self._synthesizer(query, context)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def load_dataset(self, name: str) -> None:
        """Load a built-in dataset into the graph.

        Args:
            name: Dataset name (e.g. 'syllogisms', 'causal', 'taxonomy').
        """
        from reasongraph.datasets import load_dataset as _load
        data = _load(name)
        if data["nodes"]:
            await self.add_nodes(
                [(n["content"], n["type"]) for n in data["nodes"]]
            )
        if data["edges"]:
            await self.add_edges(
                [(e["from"], e["to"]) for e in data["edges"]]
            )

    async def delete_stale(self) -> int:
        """Delete nodes not accessed within forget_after days."""
        return await self.backend.delete_stale_nodes(self.forget_after)

    async def maybe_forget(self) -> int:
        """Run ``delete_stale()`` at most once per ``forget_every`` seconds.

        Designed to be called liberally (e.g. after each write or at the end of
        a session), so regular-interval cleanup works without wiring an external
        scheduler. Actual sweeps are throttled by wall-clock time.

        Returns the number of nodes deleted on this call: 0 when ``forget_every``
        is ``None`` (feature disabled), when the throttle interval has not yet
        elapsed, or when nothing was stale.
        """
        if self.forget_every is None:
            return 0
        now = datetime.now()
        if self._last_forget is not None:
            if (now - self._last_forget).total_seconds() < self.forget_every:
                return 0
        self._last_forget = now
        return await self.delete_stale()

    async def _purge_orphan_entities(self, entities: list[str]) -> int:
        """Delete entity nodes that no longer link to any text fact.

        Used for complete erasure: after a fact is deleted, entities it introduced
        (names, places, emails) that no other fact references would otherwise stay
        resident and queryable. Entities still bridging another fact are kept.
        """
        orphans = [
            e for e in entities
            if not any(
                n["type"] == "text" for n in await self.backend.get_neighbors(e)
            )
        ]
        return await self.backend.delete_nodes(orphans) if orphans else 0

    async def delete(self, content: str, purge_orphans: bool = False) -> bool:
        """Delete a single node and its incident edges by exact content.

        Returns True if a node was deleted, False if no node matched. Shared
        entity nodes are not touched; only the named node and the edges
        incident to it are removed.

        When ``purge_orphans`` is True, entity nodes that linked only to the
        deleted fact (and now reference no other fact) are removed too -- required
        for complete erasure / right-to-be-forgotten, since extracted entities are
        often the PII. Entities still bridging another fact are kept. Defaults to
        False to preserve the standard behaviour of leaving entities resident.
        """
        entity_neighbors = (
            [n["content"] for n in await self.backend.get_neighbors(content)
             if n["type"] == "entity"]
            if purge_orphans else []
        )
        deleted = await self.backend.delete_nodes([content]) > 0
        if deleted and purge_orphans:
            await self._purge_orphan_entities(entity_neighbors)
        return deleted

    async def forget(self, scopes, *, purge_orphans: bool = True) -> dict:
        """Erase everything held under ``scopes`` (sessions, a tenant, a test run).

        A node that lives only in these scopes is deleted, with the entities that
        linked only to it when ``purge_orphans`` is set. A node also held elsewhere
        (the same sentence pushed by another session or tenant) merely loses these
        scopes and stays for its other holders. Returns ``{"deleted", "detached"}``.
        """
        scopes = set(scopes)
        if not scopes:
            return {"deleted": 0, "detached": 0}
        contents = await self.backend.nodes_in_scopes(scopes)
        held = await self.backend.get_scopes(contents)
        gone = [c for c in contents if held.get(c, set()) <= scopes]
        kept = [c for c in contents if c not in set(gone)]
        if kept:
            await self.backend.remove_scopes(kept, scopes)
        deleted = 0
        types: dict[str, str] = {}
        try:
            types = await self.backend.get_node_types(list(gone))
        except NotImplementedError:
            try:
                for n in await self.get_all_nodes():
                    if n.content in set(gone):
                        types[n.content] = getattr(n, "type", "text")
            except Exception:
                pass
        # text facts first (their orphan entities go with them), then leftover entity nodes
        for c in sorted(gone, key=lambda c: types.get(c, "text") != "text"):
            if await self.delete(c, purge_orphans=purge_orphans):
                deleted += 1
        return {"deleted": deleted, "detached": len(kept)}

    async def supersede(
        self,
        old_content: str,
        new_text: str,
        extractor: ExtractorFn | None = None,
        purge_orphans: bool = False,
    ) -> list[str]:
        """Replace a stale fact with a corrected one.

        Adds ``new_text`` (with entity extraction) and then deletes the node
        ``old_content``, so the superseded fact can no longer surface in
        queries. The new text is added before the old node is removed, so any
        entity shared between them survives and keeps bridging the graph.

        When ``old_content`` equals ``new_text`` the fact is re-asserted: it is
        (re)added and kept, never deleted, and a previously retired
        (soft-superseded) fact is revived (its ``invalid_at`` is cleared).

        Args:
            old_content: Exact content of the node to remove.
            new_text: The corrected text to add in its place.
            extractor: Optional entity extractor for the new text (defaults to
                the same extractor ``add_text`` uses).
            purge_orphans: When True, entities left orphaned by removing
                ``old_content`` (not referenced by the new text or any other
                fact) are deleted too -- complete erasure. Shared entities
                survive because the new text is added first.

        Returns:
            The entities extracted from ``new_text``.
        """
        entities = await self.add_text(new_text, extractor=extractor)
        if old_content != new_text:
            await self.delete(old_content, purge_orphans=purge_orphans)
        return entities

    async def get_all_nodes(
        self, scopes: set[str] | list[str] | None = None
    ) -> list[Node]:
        """Return all nodes in the graph, or only those in the given scopes."""
        return await self.backend.get_all_nodes(scopes=set(scopes) if scopes else None)

    async def get_all_edges(self) -> list[Edge]:
        """Return all edges in the graph."""
        return await self.backend.get_all_edges()

    # -- Sync convenience wrappers --

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot use sync methods from within a running event loop. "
                "Use the async methods directly instead."
            )
        return asyncio.run(coro)

    def initialize_sync(self) -> None:
        self._run(self.initialize())

    def close_sync(self) -> None:
        self._run(self.close())

    def add_nodes_sync(
        self, nodes: list[tuple[str, str]],
        scopes: set[str] | list[str] | None = None,
    ) -> None:
        self._run(self.add_nodes(nodes, scopes=scopes))

    def add_edges_sync(self, edges: list[tuple[str, str]]) -> None:
        self._run(self.add_edges(edges))

    def add_text_sync(
        self, text: str, extractor: ExtractorFn | None = None,
        scopes: set[str] | list[str] | None = None,
        causal_extractor: CausalExtractorFn | None = None,
        causal: bool | None = None,
        dedup_threshold: float | None = None,
        dedup_scopes=None,
        dedup_entity_gate: bool = False,
        resolve_conflicts: bool | None = None,
        canonicalizer: CanonicalizerFn | Mapping[str, str] | None = None,
    ) -> list[str]:
        return self._run(self.add_text(
            text, extractor, scopes, causal_extractor, causal,
            dedup_threshold=dedup_threshold, dedup_scopes=dedup_scopes, dedup_entity_gate=dedup_entity_gate, resolve_conflicts=resolve_conflicts,
            canonicalizer=canonicalizer,
        ))

    def add_texts_sync(
        self,
        texts: list[str],
        extractor: ExtractorFn | None = None,
        causal_extractor: CausalExtractorFn | None = None,
        scopes: set[str] | list[str] | None = None,
        causal: bool | None = None,
        dedup_threshold: float | None = None,
        dedup_scopes=None,
        dedup_entity_gate: bool = False,
        resolve_conflicts: bool | None = None,
        canonicalizer: CanonicalizerFn | Mapping[str, str] | None = None,
    ) -> list[list[str]]:
        return self._run(self.add_texts(
            texts, extractor, causal_extractor, scopes, causal,
            dedup_threshold=dedup_threshold, dedup_scopes=dedup_scopes, dedup_entity_gate=dedup_entity_gate, resolve_conflicts=resolve_conflicts,
            canonicalizer=canonicalizer,
        ))

    def query_sync(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 4,
        rerank_top_k: int = 4,
        search_mode: str = "embedding",
        rrf_k: int = 60,
        recency_weight: float = 0.0,
        scopes: set[str] | list[str] | None = None,
        isolate: bool | None = None,
        include_superseded: bool = False,
        as_of: datetime | None = None,
        walk_scopes: set[str] | list[str] | None = None,
    ) -> list[str]:
        return self._run(self.query(
            query, top_k, hops, rerank_top_k, search_mode, rrf_k, recency_weight,
            scopes, isolate, include_superseded, as_of, walk_scopes=walk_scopes,
        ))

    def query_detailed_sync(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 4,
        rerank_top_k: int = 4,
        search_mode: str = "embedding",
        rrf_k: int = 60,
        recency_weight: float = 0.0,
        scopes: set[str] | list[str] | None = None,
        isolate: bool | None = None,
        include_superseded: bool = False,
        as_of: datetime | None = None,
        walk_scopes: set[str] | list[str] | None = None,
    ) -> list[dict]:
        return self._run(self.query_detailed(
            query, top_k, hops, rerank_top_k, search_mode, rrf_k, recency_weight,
            scopes, isolate, include_superseded, as_of, walk_scopes=walk_scopes,
        ))

    def discover_sync(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 4,
        search_mode: str = "embedding",
        rrf_k: int = 60,
        scopes: set[str] | list[str] | None = None,
        max_results: int = 10,
        max_visited: int = 1000,
        isolate: bool | None = None,
        include_superseded: bool = False,
        walk_scopes: set[str] | list[str] | None = None,
    ) -> list[dict]:
        return self._run(self.discover(
            query, top_k, hops, search_mode, rrf_k, scopes, max_results,
            max_visited, isolate, include_superseded, walk_scopes=walk_scopes,
        ))

    def trace_effects_sync(self, content: str, **kwargs) -> dict:
        return self._run(self.trace_effects(content, **kwargs))

    def forget_sync(self, scopes, **kwargs) -> dict:
        return self._run(self.forget(scopes, **kwargs))

    def trace_causes_sync(self, content: str, **kwargs) -> dict:
        return self._run(self.trace_causes(content, **kwargs))

    def root_causes_sync(self, content: str, **kwargs) -> list[str]:
        return self._run(self.root_causes(content, **kwargs))

    def causal_chain_sync(self, from_content: str, to_content: str, **kwargs) -> list[dict] | None:
        return self._run(self.causal_chain(from_content, to_content, **kwargs))

    def what_if_sync(self, content: str, **kwargs) -> dict:
        return self._run(self.what_if(content, **kwargs))

    def answer_sync(
        self,
        query: str,
        *,
        use_discover: bool = True,
        top_k: int = 5,
        hops: int = 4,
        search_mode: str = "embedding",
        scopes: set[str] | list[str] | None = None,
        max_results: int = 10,
    ) -> str:
        return self._run(self.answer(
            query, use_discover=use_discover, top_k=top_k, hops=hops,
            search_mode=search_mode, scopes=scopes, max_results=max_results,
        ))

    def load_dataset_sync(self, name: str) -> None:
        self._run(self.load_dataset(name))

    def delete_stale_sync(self) -> int:
        return self._run(self.delete_stale())

    def maybe_forget_sync(self) -> int:
        return self._run(self.maybe_forget())

    def delete_sync(self, content: str, purge_orphans: bool = False) -> bool:
        return self._run(self.delete(content, purge_orphans=purge_orphans))

    def supersede_sync(
        self,
        old_content: str,
        new_text: str,
        extractor: ExtractorFn | None = None,
        purge_orphans: bool = False,
    ) -> list[str]:
        return self._run(self.supersede(
            old_content, new_text, extractor, purge_orphans=purge_orphans,
        ))

    def supersession_history_sync(self, content: str) -> dict:
        return self._run(self.supersession_history(content))

    def get_all_nodes_sync(self, scopes: set[str] | list[str] | None = None) -> list:
        return self._run(self.get_all_nodes(scopes=scopes))

    def get_all_edges_sync(self) -> list:
        return self._run(self.get_all_edges())
