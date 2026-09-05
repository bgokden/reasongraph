"""Operation-latency benchmark for reasongraph memory.

Builds a synthetic graph of causal chains with shared bridge entities, then
measures the operations an agent hits constantly: write throughput and the read
latency of query / discover / trace_effects (p50/p95). Reports real numbers with
the default embedder+reranker; pass --fake for structure-only timing (no model
download, useful in CI or to isolate traversal cost from embedding cost).

Runs one configuration per process (loads heavy models). Not a pytest test.

Usage:
    uv run python tests/bench_speed.py                       # real models, memory backend
    uv run python tests/bench_speed.py --facts 500 --reads 50
    uv run python tests/bench_speed.py --backend sqlite
    uv run python tests/bench_speed.py --fake                # fast, no model download
"""

from __future__ import annotations

import argparse
import time

def _stable_hash(text):
    """Process-independent 64-bit hash (Python's hash() is randomized per run,
    which made fake embeddings and therefore test outcomes flaky)."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")



def _fake_encode(text):
    h = _stable_hash(text)
    return [(h >> i & 0xFF) / 255.0 for i in range(0, 384 * 8, 8)][:384]


def _fake_embed(x):
    return [_fake_encode(t) for t in x] if isinstance(x, list) else _fake_encode(x)


def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    if lo + 1 >= len(s):
        return s[lo]
    return s[lo] + (s[lo + 1] - s[lo]) * (k - lo)


def _make_backend(kind, url):
    if kind == "memory":
        from reasongraph.backends._memory import MemoryBackend
        return MemoryBackend()
    if kind == "sqlite":
        from reasongraph.backends._sqlite import SqliteBackend
        return SqliteBackend(url or ":memory:")
    if kind == "postgres":
        from reasongraph.backends._postgres import PostgresBackend
        return PostgresBackend(url or "postgresql:///reasongraph_bench")
    raise ValueError(f"unknown backend {kind}")


def _synthetic(chains, length):
    """Build (nodes, edges, heads): causal chains e{k}_0 -> e{k}_1 -> ... with a fact
    per hop and every chain sharing a 'hub' entity so discover has bridges."""
    nodes, edges, heads, fact_texts = [], [], [], []
    for k in range(chains):
        heads.append(f"In scenario {k}, event 0 caused event 1.")
        for j in range(length):
            fact = f"In scenario {k}, event {j} caused event {j + 1}."
            fact_texts.append(fact)
            cause, effect = f"e{k}_{j}", f"e{k}_{j + 1}"
            nodes += [(fact, "text"), (cause, "entity"), (effect, "entity")]
            edges += [(cause, effect, "causes"), (cause, fact), (effect, fact)]
        # a hub entity shared across all chains -> cross-chain discovery bridges
        nodes.append(("hub", "entity"))
        edges.append(("hub", f"In scenario {k}, event 0 caused event 1."))
    return nodes, edges, heads, fact_texts


async def _run(args):
    from reasongraph.graph import ReasonGraph

    backend = _make_backend(args.backend, args.url)
    graph = ReasonGraph(backend=backend, embed_model=_fake_embed if args.fake else None,
                        causal_extractor=False)
    if args.fake:
        graph.embeddings.encode = _fake_encode
        graph.embeddings.encode_batch = lambda xs: [_fake_encode(x) for x in xs]
        graph.embeddings.rerank = lambda q, r, k, recency_weight=0.0: r[:k]
    await graph.initialize()

    nodes, edges, heads, fact_texts = _synthetic(args.chains, args.length)
    n_facts = len(fact_texts)

    t0 = time.perf_counter()
    await graph.add_nodes(nodes)
    await graph.add_edges(edges)
    build_s = time.perf_counter() - t0

    async def _bench(label, factory, n):
        lat = []
        for i in range(n):
            t = time.perf_counter()
            await factory(i)
            lat.append((time.perf_counter() - t) * 1000)
        print(f"  {label:16s} n={n:<4d} p50={_pct(lat, 0.5):7.2f}ms  "
              f"p95={_pct(lat, 0.95):7.2f}ms  mean={sum(lat) / len(lat):7.2f}ms")

    print(f"\nreasongraph speed  backend={args.backend}  models={'fake' if args.fake else 'real'}")
    print(f"  build            facts={n_facts:<4d} nodes={len(nodes)} edges={len(edges)}  "
          f"{build_s * 1000:.0f}ms total  {n_facts / build_s:.0f} facts/s (embed+insert)")

    reads = min(args.reads, n_facts)
    # Warm the lazily-loaded cross-encoder on a real multi-result query so its
    # one-time load doesn't land in the timed p95.
    await graph.query(fact_texts[0], hops=args.hops)
    await _bench("query", lambda i: graph.query(fact_texts[i % n_facts], hops=args.hops), reads)
    await _bench("discover", lambda i: graph.discover(fact_texts[i % n_facts], hops=args.hops), reads)
    await _bench("trace_effects", lambda i: graph.trace_effects(heads[i % len(heads)]), reads)

    await graph.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="memory", choices=["memory", "sqlite", "postgres"])
    parser.add_argument("--url", default=None, help="sqlite path or postgres URL")
    parser.add_argument("--chains", type=int, default=20, help="number of causal chains")
    parser.add_argument("--length", type=int, default=10, help="facts per chain")
    parser.add_argument("--facts", type=int, default=None, help="target fact count (overrides chains*length)")
    parser.add_argument("--reads", type=int, default=30, help="query/discover/trace samples")
    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--fake", action="store_true", help="fake models (no download)")
    args = parser.parse_args(argv)
    if args.facts:
        args.length = max(2, args.facts // args.chains)

    import asyncio
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
