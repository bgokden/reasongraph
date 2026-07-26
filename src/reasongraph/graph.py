from __future__ import annotations

import asyncio
import warnings
from collections import deque
from datetime import datetime

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
    ) -> None:
        self.backend = backend or MemoryBackend()
        self.embeddings = EmbeddingManager(
            embed_model=embed_model, rerank_model=rerank_model
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
        """
        try:
            import causal_span_model as _pointer_check  # noqa: F401
            return CausalPointerExtractor()
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
        resolve_conflicts: bool | None = None,
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

        Returns:
            List of extracted entity strings.
        """
        result = await self.add_texts(
            [text], extractor=extractor, causal_extractor=causal_extractor,
            scopes=scopes, causal=causal, dedup_threshold=dedup_threshold,
            resolve_conflicts=resolve_conflicts,
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
        resolve_conflicts: bool | None = None,
    ) -> list[list[str]]:
        """Add multiple texts with automatic entity and causal extraction.

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

        Returns:
            List of entity lists, one per input text. Skipped duplicates yield [].
        """
        if extractor is None:
            if not hasattr(self, "_default_extractor"):
                self._default_extractor = self._build_default_extractor()
            extractor = self._default_extractor

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
        if dedup_threshold is not None:
            for text in texts:
                if text in skip:
                    continue
                dup = await self._find_duplicate(text, dedup_threshold)
                if dup is not None:
                    skip.add(text)
                    if scopes:
                        await self.add_nodes([(dup, "text")], scopes=scopes)

        all_entities = []
        all_nodes = []
        all_edges = []
        active_texts = []

        # NER entity extraction (duplicates are skipped, yielding [] entities)
        for text in texts:
            if text in skip:
                all_entities.append([])
                continue
            entities = extractor(text)
            all_entities.append(entities)
            active_texts.append(text)
            all_nodes.append((text, "text"))
            for entity in entities:
                all_nodes.append((entity, "entity"))
                all_edges.append((entity, text))

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

        if all_nodes:
            await self.add_nodes(all_nodes, scopes=scopes)
        if all_edges:
            await self.add_edges(all_edges)

        # Conflict resolution: soft-supersede existing facts the new ones contradict.
        do_resolve = (self.conflict_resolver is not None
                      if resolve_conflicts is None else resolve_conflicts)
        if do_resolve:
            if self.conflict_resolver is None:
                raise ValueError(
                    "resolve_conflicts=True but no conflict_resolver is configured. "
                    "Pass conflict_resolver=NLIConflictResolver() to ReasonGraph."
                )
            if active_texts:
                await self._resolve_conflicts(active_texts)

        return all_entities

    async def _superseded(self, contents: list[str]) -> list[str]:
        """Return the subset of ``contents`` that a newer fact has superseded.

        A fact is superseded when another fact points at it via a ``"supersedes"``
        edge (soft-supersede): it stays in the graph but drops out of default recall.
        """
        out = []
        for content in contents:
            neighbors = await self.backend.get_neighbors(content)
            if any(n.get("label") == "supersedes" and n.get("direction") == "in"
                   for n in neighbors):
                out.append(content)
        return out

    async def _resolve_conflicts(self, texts: list[str]) -> None:
        """Add a ``"supersedes"`` edge from each new fact to the facts it contradicts."""
        batch = set(texts)
        edges: list[tuple] = []
        for text in texts:
            embedding = self.embeddings.encode(text)
            candidates = await self.backend.knn_search(embedding, top_k=5)
            pool = [c["content"] for c in candidates
                    if c.get("type") == "text" and c["content"] not in batch]
            if not pool:
                continue
            already = set(await self._superseded(pool))
            pool = [c for c in pool if c not in already]
            if not pool:
                continue
            for old in self.conflict_resolver.contradictions(text, pool):
                edges.append((text, old, "supersedes"))
        if edges:
            await self.add_edges(edges)

    async def _find_duplicate(self, text: str, threshold: float) -> str | None:
        """Return an existing text fact that near-duplicates ``text``, or None.

        Exact-content matches are left to the backend upsert (which unions
        scopes); only a distinct text node whose cosine similarity is >=
        ``threshold`` counts as a near-duplicate.
        """
        embedding = self.embeddings.encode(text)
        candidates = await self.backend.knn_search(embedding, top_k=5)
        others = [
            c["content"] for c in candidates
            if c.get("type") == "text" and c["content"] != text
        ]
        if not others:
            return None
        scores = self.embeddings.score(text, others)
        best = max(range(len(others)), key=lambda i: scores[i])
        return others[best] if scores[best] >= threshold else None

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

        Returns:
            List of text-type node contents in relevance order.
        """
        if search_mode not in ("embedding", "keyword", "hybrid"):
            raise ValueError(f"search_mode must be 'embedding', 'keyword', or 'hybrid', got '{search_mode}'")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError(f"recency_weight must be in [0, 1], got {recency_weight}")

        scope_set = set(scopes) if scopes else None
        isolate = self.isolate_traversal if isolate is None else isolate
        walk_scopes = scope_set if (isolate and scope_set) else None
        embedding = self.embeddings.encode(query)

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

            ranked_chain = self.embeddings.rerank(query, chain_pool, rerank_top_k, recency_weight)
            remaining_budget = max(0, rerank_top_k - len(ranked_chain))
            if remaining_budget > 0 and bridge_pool:
                ranked_bridge = self.embeddings.rerank(query, bridge_pool, remaining_budget, recency_weight)
            else:
                ranked_bridge = []
            ranked = ranked_chain + ranked_bridge

            chain_next: list[dict[str, str]] = []
            entity_next: list[dict[str, str]] = []

            for seed in ranked:
                if seed["content"] in visited:
                    continue
                results.append(seed)
                visited.add(seed["content"])
                neighbors = await self.backend.get_neighbors(seed["content"], walk_scopes)
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
                neighbors = await self.backend.get_neighbors(seed["content"], walk_scopes)
                for n in neighbors:
                    if n["content"] not in visited:
                        if n["type"] == "text":
                            n["_source"] = "bridge"
                            bridge_next.append(n)
                        else:
                            entity_next.append(n)

            seeds = chain_next + bridge_next + entity_next

        texts = [node["content"] for node in results if node["type"] == "text"]
        # Drop soft-superseded facts (only possible when a resolver is configured,
        # so this costs nothing for graphs that don't use conflict resolution).
        if not include_superseded and self.conflict_resolver is not None and texts:
            superseded = set(await self._superseded(texts))
            texts = [t for t in texts if t not in superseded]
        return texts

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
        )
        if not contents:
            return []
        created = await self.backend.get_created_at(contents)
        scope_map = await self.backend.get_scopes(contents)
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
        """
        scope_set = set(scopes) if scopes else None
        isolate = self.isolate_traversal if isolate is None else isolate
        walk_scopes = scope_set if (isolate and scope_set) else None
        embedding = self.embeddings.encode(query)
        if search_mode == "embedding":
            seeds = await self.backend.knn_search(embedding, top_k, scopes=scope_set)
        elif search_mode == "keyword":
            seeds = await self.backend.hybrid_search(
                embedding, query, top_k, keyword_only=True, scopes=scope_set,
            )
        elif search_mode == "hybrid":
            seeds = await self.backend.hybrid_search(
                embedding, query, top_k, rrf_k=rrf_k, scopes=scope_set,
            )
        else:
            raise ValueError(
                f"search_mode must be 'embedding', 'keyword', or 'hybrid', got '{search_mode}'"
            )

        # Seed only from text facts so every connection path is rooted at a fact
        # (entities bridge during traversal, they are not path roots).
        seeds = [s for s in seeds if s.get("type") == "text"]

        # Breadth-first traversal tracking, for every node, the fact and entity
        # it was reached through. parent[c] = (prior_fact, bridging_entity, depth);
        # seeds have (None, None, 0). By default get_neighbors is unscoped, so the
        # walk crosses knowledge sessions; walk_scopes confines it when isolating.
        visited: set[str] = set()
        parent: dict[str, tuple] = {}
        order: list[str] = []  # discovered text facts, in BFS order
        frontier: deque = deque()
        for s in seeds:
            c = s["content"]
            if c in visited:
                continue
            visited.add(c)
            parent[c] = (None, None, 0)
            frontier.append((c, s.get("type", "text"), 0))
            if s.get("type") == "text":
                order.append(c)

        while frontier and len(visited) < max_visited:
            content, ntype, depth = frontier.popleft()
            if depth >= hops:
                continue
            for n in await self.backend.get_neighbors(content, walk_scopes):
                if len(visited) >= max_visited:
                    break
                nc, nt = n["content"], n["type"]
                if nc in visited:
                    continue
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
        scope_map = await self.backend.get_scopes(order)

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
        causal_map = await self.backend.get_causal_relations(order)

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
                scopes=scopes, max_results=max_results,
            )
        else:
            facts = await self.query(
                query, top_k=top_k, hops=hops, search_mode=search_mode, scopes=scopes,
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

        When ``old_content`` equals ``new_text`` the call is a no-op
        replacement: the fact is (re)added and kept, never deleted.

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
    ) -> list[str]:
        return self._run(self.add_text(text, extractor, scopes, causal_extractor, causal))

    def add_texts_sync(
        self,
        texts: list[str],
        extractor: ExtractorFn | None = None,
        causal_extractor: CausalExtractorFn | None = None,
        scopes: set[str] | list[str] | None = None,
        causal: bool | None = None,
    ) -> list[list[str]]:
        return self._run(self.add_texts(texts, extractor, causal_extractor, scopes, causal))

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
    ) -> list[str]:
        return self._run(self.query(
            query, top_k, hops, rerank_top_k, search_mode, rrf_k, recency_weight,
            scopes, isolate,
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
    ) -> list[dict]:
        return self._run(self.query_detailed(
            query, top_k, hops, rerank_top_k, search_mode, rrf_k, recency_weight,
            scopes, isolate,
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
    ) -> list[dict]:
        return self._run(self.discover(
            query, top_k, hops, search_mode, rrf_k, scopes, max_results,
            max_visited, isolate,
        ))

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

    def delete_sync(self, content: str) -> bool:
        return self._run(self.delete(content))

    def supersede_sync(
        self,
        old_content: str,
        new_text: str,
        extractor: ExtractorFn | None = None,
    ) -> list[str]:
        return self._run(self.supersede(old_content, new_text, extractor))
