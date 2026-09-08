"""Env-driven entrypoint: build a MemoryService + FastAPI app from environment.

Serve it with the ASGI factory (no import-time model load until built)::

    uvicorn reasongraph.service.app:create_app_from_env --factory --host 0.0.0.0 --port 8000

or the console script::

    reasongraph-serve

Environment variables:
    REASONGRAPH_BACKEND        memory | sqlite | postgres        (default: memory)
    REASONGRAPH_DATABASE_URL   postgres URL, sqlite file path, or memory JSON path
    REASONGRAPH_EMBED_MODEL    embedding model name; prefix 'fastembed:' for ONNX
                               (default: the built-in SentenceTransformer)
    REASONGRAPH_RERANK_MODEL   reranker model name; prefix 'fastembed:' for a
                               CPU-fast ONNX cross-encoder, e.g.
                               'fastembed:Xenova/ms-marco-MiniLM-L-6-v2'
                               (default: the built-in torch CrossEncoder)
    REASONGRAPH_SYNTHESIZER    none | template | transformers    (default: template)
    REASONGRAPH_SYNTH_MODEL    instruct model for the transformers synthesizer
    REASONGRAPH_FORGET_AFTER   days; facts idle longer are droppable (default: 30)
    REASONGRAPH_FORGET_EVERY   seconds between auto-forget sweeps (unset: disabled)
    REASONGRAPH_ISOLATE        1/true to confine traversal to the query session
                               (multi-tenant); default off (cross-session discovery)
    REASONGRAPH_API_KEY        when set, data endpoints require it (Bearer/X-API-Key)
    REASONGRAPH_SPAN_LINK_THRESHOLD  cosine similarity (e.g. 0.85) above which a new
                               cause/effect span is tied to an existing causal span
                               so chains can cross facts with different wording
    REASONGRAPH_CAUSAL_MODEL / REASONGRAPH_CAUSAL_GATE_THRESHOLD /
    REASONGRAPH_CAUSAL_EMBED_GATE / REASONGRAPH_CAUSAL_EMBED_GATE_THRESHOLD  span-pointer
        model id, built-in gate threshold (1.0 = off), optional embedding-gate .joblib
        (path or hf://owner/repo/file) and its P(causal) cutoff (default 0.9).
    REASONGRAPH_SPLIT_SENTENCES  sat | sat:<model> | regex -> every push is split into
        sentences and stored one fact per sentence (clients can override per call with
        split=true/false). Unset = off. "sat" needs pip install reasongraph[split].
    REASONGRAPH_DEDUP_THRESHOLD  cosine similarity (e.g. 0.95) above which a new
                               fact is treated as a paraphrase of an existing one
                               (scopes are unioned, nothing new is added)
    REASONGRAPH_DEFER_EXTRACT  1/true to run entity/causal extraction in the
                               background so pushes return fast; default off
    REASONGRAPH_RESOLVE_CONFLICTS  1/true to soft-supersede facts a new push
                               contradicts (loads an NLI model); default off
    REASONGRAPH_CANONICALIZE   1/true to canonicalize extracted entities
                               (whitespace/case normalization + corporate-suffix
                               stripping) so surface variants bridge; default off
    REASONGRAPH_ALIASES        path to a JSON object of surface form -> canonical
                               name (e.g. {"the Fed": "Federal Reserve"}); implies
                               canonicalization on, layered over suffix stripping
    REASONGRAPH_HOST/PORT      bind address for the console script (0.0.0.0 / 8000)

Requires ``pip install reasongraph[service]`` plus the extras for the chosen
backend/embedder (e.g. ``postgres``, ``fastembed``, ``gliner``).
"""

from __future__ import annotations

import os
from typing import Mapping

from reasongraph.service.core import MemoryService
from reasongraph.service.http import create_app


def build_backend(env: Mapping[str, str] | None = None):
    env = env if env is not None else os.environ
    kind = env.get("REASONGRAPH_BACKEND", "memory").lower()
    url = env.get("REASONGRAPH_DATABASE_URL") or None

    if kind == "memory":
        from reasongraph.backends._memory import MemoryBackend
        return MemoryBackend(file_path=url)
    if kind == "sqlite":
        from reasongraph.backends._sqlite import SqliteBackend
        return SqliteBackend(db_path=url or ":memory:")
    if kind == "postgres":
        if not url:
            raise ValueError(
                "REASONGRAPH_BACKEND=postgres requires REASONGRAPH_DATABASE_URL"
            )
        from reasongraph.backends._postgres import PostgresBackend
        return PostgresBackend(url)
    raise ValueError(
        f"Unknown REASONGRAPH_BACKEND '{kind}' (memory | sqlite | postgres)"
    )


def build_embed_model(env: Mapping[str, str] | None = None):
    env = env if env is not None else os.environ
    name = env.get("REASONGRAPH_EMBED_MODEL")
    if not name:
        return None  # ReasonGraph uses its default SentenceTransformer
    prefix = "fastembed:"
    if name.startswith(prefix):
        from reasongraph import FastEmbedEmbedder
        return FastEmbedEmbedder(name[len(prefix):])
    return name


def build_rerank_model(env: Mapping[str, str] | None = None):
    env = env if env is not None else os.environ
    name = env.get("REASONGRAPH_RERANK_MODEL")
    if not name:
        return None  # ReasonGraph uses its default CrossEncoder reranker
    prefix = "fastembed:"
    if name.startswith(prefix):
        from reasongraph import FastEmbedReranker
        return FastEmbedReranker(name[len(prefix):])
    return name


def build_synthesizer(env: Mapping[str, str] | None = None):
    env = env if env is not None else os.environ
    kind = env.get("REASONGRAPH_SYNTHESIZER", "template").lower()
    if kind in ("none", ""):
        return None
    if kind == "template":
        from reasongraph import TemplateSynthesizer
        return TemplateSynthesizer()
    if kind == "transformers":
        from reasongraph import TransformersSynthesizer
        return TransformersSynthesizer(model=env.get("REASONGRAPH_SYNTH_MODEL") or None)
    raise ValueError(
        f"Unknown REASONGRAPH_SYNTHESIZER '{kind}' (none | template | transformers)"
    )


def build_canonicalizer(env: Mapping[str, str] | None = None):
    """Build an entity canonicalizer from env, or None when disabled.

    Enabled by ``REASONGRAPH_CANONICALIZE`` (suffix/whitespace normalization) or by
    providing ``REASONGRAPH_ALIASES`` (a JSON object of surface -> canonical name).
    An alias file implies canonicalization on, layered over suffix stripping.
    """
    env = env if env is not None else os.environ
    aliases_path = env.get("REASONGRAPH_ALIASES") or None
    enabled = _bool_env(env, "REASONGRAPH_CANONICALIZE", False)
    if not enabled and not aliases_path:
        return None
    aliases = None
    if aliases_path:
        import json
        with open(aliases_path, encoding="utf-8") as f:
            aliases = json.load(f)
        if not isinstance(aliases, dict):
            raise ValueError(
                "REASONGRAPH_ALIASES must be a JSON object of surface -> canonical name"
            )
        if not all(isinstance(v, str) for v in aliases.values()):
            raise ValueError(
                "REASONGRAPH_ALIASES values must be strings (canonical names)"
            )
    from reasongraph import AliasCanonicalizer
    return AliasCanonicalizer(aliases=aliases)


def _int_env(env: Mapping[str, str], key: str, default: int | None) -> int | None:
    value = env.get(key)
    return int(value) if value not in (None, "") else default


def _bool_env(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    value = env.get(key)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def build_service(env: Mapping[str, str] | None = None) -> MemoryService:
    """Construct a MemoryService (loading models) from environment variables."""
    env = env if env is not None else os.environ
    from reasongraph import ReasonGraph

    resolver = None
    if _bool_env(env, "REASONGRAPH_RESOLVE_CONFLICTS", False):
        from reasongraph import NLIConflictResolver
        resolver = NLIConflictResolver()

    graph = ReasonGraph(
        backend=build_backend(env),
        embed_model=build_embed_model(env),
        rerank_model=build_rerank_model(env),
        synthesizer=build_synthesizer(env),
        forget_after=_int_env(env, "REASONGRAPH_FORGET_AFTER", 30),
        forget_every=_int_env(env, "REASONGRAPH_FORGET_EVERY", None),
        isolate_traversal=_bool_env(env, "REASONGRAPH_ISOLATE", False),
        conflict_resolver=resolver,
        canonicalizer=build_canonicalizer(env),
        span_link_threshold=(float(env["REASONGRAPH_SPAN_LINK_THRESHOLD"])
                             if env.get("REASONGRAPH_SPAN_LINK_THRESHOLD") else None),
        sentence_splitter=(env.get("REASONGRAPH_SPLIT_SENTENCES") or None),
    )
    dedup = env.get("REASONGRAPH_DEDUP_THRESHOLD")
    return MemoryService(
        graph=graph,
        defer_extraction=_bool_env(env, "REASONGRAPH_DEFER_EXTRACT", False),
        dedup_threshold=float(dedup) if dedup not in (None, "") else None,
        split_sentences=bool(env.get("REASONGRAPH_SPLIT_SENTENCES")),
    )


def create_app_from_env(env: Mapping[str, str] | None = None):
    """ASGI factory: ``uvicorn reasongraph.service.app:create_app_from_env --factory``.

    Set ``REASONGRAPH_REQUIRE_AUTH=1`` in production so the app refuses to start
    with authentication disabled (i.e. when ``REASONGRAPH_API_KEY`` is empty) --
    a fail-closed guard against accidentally exposing an open service.
    """
    env = env if env is not None else os.environ
    api_key = env.get("REASONGRAPH_API_KEY") or None
    if api_key is None and _bool_env(env, "REASONGRAPH_REQUIRE_AUTH", False):
        raise RuntimeError(
            "REASONGRAPH_REQUIRE_AUTH is set but REASONGRAPH_API_KEY is empty; "
            "refusing to start with authentication disabled."
        )
    return create_app(build_service(env), api_key=api_key)


def main() -> None:
    """Console-script entrypoint (``reasongraph-serve``)."""
    import uvicorn

    uvicorn.run(
        "reasongraph.service.app:create_app_from_env",
        factory=True,
        host=os.environ.get("REASONGRAPH_HOST", "0.0.0.0"),
        port=int(os.environ.get("REASONGRAPH_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
