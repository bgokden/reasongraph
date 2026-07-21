# ReasonGraph

A graph-based reasoning library that discovers connections across independent documents through entity and causal extraction, embedding search, and multi-hop graph traversal.

[![PyPI version](https://img.shields.io/pypi/v/reasongraph?color=blue)](https://pypi.org/project/reasongraph/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/reasongraph?color=blue)](https://pypi.org/project/reasongraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Why ReasonGraph?

Standard RAG retrieves documents similar to your query. ReasonGraph discovers connections *between* documents that were written independently.

When you feed text into `add_texts()`, GLiNER2 automatically extracts **entities** and **cause-effect relations** that become nodes and edges in a graph. Documents that share entities or causal chains get connected -- even if they never reference each other. Multi-hop traversal then walks these connections to build reasoning chains that span multiple sources.

## Installation

```bash
pip install reasongraph[all]        # everything included
```

Or install only what you need:

```bash
pip install reasongraph             # core: in-memory backend, NER extraction, embeddings
pip install reasongraph[sqlite]     # + SQLite backend with sqlite-vec
pip install reasongraph[gliner2]    # + GLiNER2 entity + causal extraction (recommended)
pip install reasongraph[postgres]   # + PostgreSQL + pgvector backend
```

## Cross-Source Discovery

Two reports about different topics. Source A covers TSMC's semiconductor plant. Source B covers Arizona's water crisis. Neither mentions the other's subject.

```python
import asyncio
from reasongraph import ReasonGraph

source_a = [  # Tech industry report
    "TSMC announced plans to build a $40 billion semiconductor fabrication plant in Phoenix, Arizona.",
    "The Phoenix fab requires 10 million gallons of purified water daily to cool wafers during the chip etching process.",
    "TSMC signed a long-term supply agreement with Apple to manufacture next-generation M-series processors at the Arizona facility.",
    "Construction delays at the Phoenix site pushed first production to late 2025, raising concerns among TSMC's major customers.",
]

source_b = [  # Environmental report -- never mentions TSMC, semiconductors, or chips
    "Arizona declared a water emergency after Lake Mead dropped to its lowest level since the 1930s, threatening water supply for millions.",
    "The Arizona Department of Water Resources ordered mandatory water cuts for all industrial users in Maricopa County, where Phoenix is located.",
    "Intel paused expansion of its Chandler, Arizona chip plant citing water availability concerns and rising operational costs.",
    "Apple warned investors that component shortages from its Asian and North American suppliers could impact iPhone production timelines through 2026.",
]

async def main():
    async with ReasonGraph() as graph:
        await graph.add_texts(source_a)
        await graph.add_texts(source_b)
        results = await graph.query("How does the Arizona water crisis affect semiconductor manufacturing?")
        for i, text in enumerate(results, 1):
            source = "A" if text in source_a else "B"
            print(f"{i}. [Source {source}] {text}")

asyncio.run(main())
```

```
1. [Source B] Intel paused expansion of its Chandler, Arizona chip plant citing water availability concerns and rising operational costs.
2. [Source B] The Arizona Department of Water Resources ordered mandatory water cuts for all industrial users in Maricopa County, where Phoenix is located.
3. [Source A] The Phoenix fab requires 10 million gallons of purified water daily to cool wafers during the chip etching process.
4. [Source B] Arizona declared a water emergency after Lake Mead dropped to its lowest level since the 1930s.
5. [Source A] TSMC announced plans to build a $40 billion semiconductor fabrication plant in Phoenix, Arizona.
6. [Source A] TSMC signed a long-term supply agreement with Apple to manufacture M-series processors at the Arizona facility.
```

Results come from both sources. No single document contains this chain. Here is what happens under the hood:

**GLiNER2 extracts entities and causal relations from each text:**

| Text (abbreviated) | Entities | Causal relations |
|---------------------|----------|------------------|
| TSMC to build fab in Phoenix, Arizona... | TSMC, Phoenix, Arizona | -- |
| Phoenix fab requires 10M gallons water... | Phoenix | -- |
| TSMC supply agreement with Apple... | TSMC, Apple, Arizona | -- |
| Construction delays at Phoenix site... | TSMC, Phoenix | Construction delays -> first production |
| Arizona water emergency, Lake Mead... | Arizona, Lake Mead | Lake Mead dropped -> water emergency |
| Mandatory water cuts in Maricopa County... | Arizona Dept. of Water Resources, Phoenix, Maricopa County | -- |
| Intel paused Arizona chip plant... | Intel, Chandler, Arizona | -- |
| Apple warned of component shortages... | Apple | component shortages -> iPhone production timelines |

**Three entities appear in both sources, creating bridge nodes:**

| Bridge entity | Source A connections | Source B connections |
|---------------|---------------------|---------------------|
| Arizona | TSMC fab, TSMC-Apple deal | water emergency, Intel pause, water cuts |
| Phoenix | TSMC fab, water usage, delays | water cuts for industrial users |
| Apple | TSMC supply agreement | component shortage warning |

**The query traversal path:**

Water crisis query -> finds water-related texts from both sources via embeddings -> follows `Arizona` and `Phoenix` entity edges to discover TSMC's water-intensive fab -> follows `Apple` entity edge from TSMC supply agreement to Apple's component shortage warning. The causal relation `Lake Mead dropped -> water emergency` connects the environmental trigger to the industrial impact.

Full demo: `uv run python examples/cross_source_discovery.py`

## Quick Start

### Using a built-in dataset

```python
from reasongraph import ReasonGraph

graph = ReasonGraph()
graph.initialize_sync()
graph.load_dataset_sync("financial")

results = graph.query_sync("What caused the 2008 financial crisis?")
for i, text in enumerate(results, 1):
    print(f"{i}. {text}")

graph.close_sync()
```

Output -- a connected reasoning chain, not just keyword matches:

```
1. Lehman Brothers filed for bankruptcy in September 2008 after massive MBS losses.
2. Loose lending standards fueled a housing price bubble across the United States.
3. Lehman's collapse triggered a global credit freeze as interbank lending stopped.
4. Mortgage-backed securities built on subprime loans collapsed when defaults surged.
5. The U.S. government enacted TARP, a $700 billion bailout to stabilize the financial system.
6. Banks issued subprime mortgages to borrowers with poor credit histories.
```

### Async API

```python
import asyncio
from reasongraph import ReasonGraph

async def main():
    async with ReasonGraph() as graph:
        await graph.load_dataset("financial")
        results = await graph.query("What caused the 2008 crisis?")
        for text in results:
            print(text)

asyncio.run(main())
```

## Features

- **Cross-source discovery** -- connect facts across independent documents through shared entities and causal relations
- **Automatic extraction** -- GLiNER2 extracts entities and causal relations in one pass (falls back to BERT NER when gliner2 is not installed)
- **Hybrid search** -- combine embedding similarity, keyword (trigram) matching, or both
- **Multi-hop traversal** -- follow graph edges to discover connected reasoning chains
- **Cross-encoder reranking** -- rerank results at each hop with `ms-marco-MiniLM-L-6-v2`
- **Built-in datasets** -- load curated reasoning graphs for immediate use
- **Async-first** -- native async API with sync convenience wrappers
- **Pluggable backends** -- in-memory (zero-config default), SQLite, or PostgreSQL with pgvector

## Built-in Datasets

| Dataset | Description |
|---------|-------------|
| `syllogisms` | Classical syllogistic reasoning chains |
| `causal` | Cause-effect reasoning with entity annotations |
| `taxonomy` | Hierarchical concept taxonomy |
| `financial` | Financial crisis causal chains (2008 crisis, dot-com, inflation, eurozone) |
| `medical` | Medical causal chains (heart disease, diabetes, infectious disease, cancer) |
| `analysis_patterns` | Data analysis reasoning: scenario detection, technique selection, implementation patterns |

```python
graph.load_dataset_sync("financial")
```

## Search Modes

```python
# Pure embedding similarity (default)
results = graph.query_sync("credit freeze", search_mode="embedding")

# Pure keyword/trigram matching
results = graph.query_sync("credit freeze", search_mode="keyword")

# Hybrid: Reciprocal Rank Fusion of embedding + trigram rankings
results = graph.query_sync("credit freeze", search_mode="hybrid")

# Tune the RRF smoothing constant (default 60, lower = more weight to top ranks)
results = graph.query_sync("credit freeze", search_mode="hybrid", rrf_k=30)
```

## Entity and Causal Extraction

When `gliner2` is installed, `add_text()` / `add_texts()` automatically use GLiNER2 for both entity extraction and causal relation detection. Without `gliner2`, it falls back to BERT NER (entities only).

```python
from reasongraph import ReasonGraph, NERExtractor, GLiNER2Extractor

graph = ReasonGraph()
graph.initialize_sync()

# Default: GLiNER2 (entities + causal relations) if installed, else BERT NER
entities = graph.add_text_sync("Apple released the iPhone in 2007.")
print(entities)  # ['Apple', 'iPhone']

# Explicit: force BERT NER even if GLiNER2 is installed
entities = graph.add_text_sync("Apple released the iPhone in 2007.", extractor=NERExtractor())

# Explicit: GLiNER2 with custom entity types
gliner = GLiNER2Extractor(entity_types=["company", "product", "date"])
entities = graph.add_text_sync("Apple released the iPhone in 2007.", extractor=gliner)

# Conversational memory: ChatExtractor also captures preference/plan/topic,
# so "hard techno" or "visit" become bridgeable nodes -- not just people/places
from reasongraph import ChatExtractor
entities = graph.add_text_sync(
    "I love hard techno and plan to visit Berlin.", extractor=ChatExtractor()
)  # ['Berlin', 'hard techno', 'visit']

# Any callable works
entities = graph.add_text_sync("some text", extractor=lambda t: ["custom"])
```

## Fast inference (pure ONNX)

Every model slot is pluggable, so you can trade the PyTorch defaults for
CPU-optimized ONNX models. Measured on the 32-case mixed-domain eval:

```python
from reasongraph import ReasonGraph, FastEmbedEmbedder, FastEmbedReranker

graph = ReasonGraph(
    embed_model=FastEmbedEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
    rerank_model=FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2"),
)
```

- **Reranker → `Xenova/ms-marco-MiniLM-L-6-v2`**: the ONNX build of the default
  reranker, so scores (and eval quality) are identical, but cold start drops
  from ~2.4s to ~0.03s.
- **Embedder → `all-MiniLM-L6-v2` (ONNX)**: ~2.3x faster load, equal-or-better
  eval quality.
- **Full ONNX pipeline**: ~3x faster cold start and ~23% less RAM at
  equal-or-better quality; per-query latency rises (~12ms to ~100ms), a good
  trade when cold start and memory matter more than warm latency.
- Multilingual embedder (`paraphrase-multilingual-MiniLM-L12-v2`) is available
  as an option; it costs a few points of English quality.

Requires `pip install reasongraph[fastembed]`. Benchmark any configuration with
`tests/bench_pipeline.py`.

### Choosing an extractor

`add_text` / `add_texts` accept any extractor, so the entity model is a
measured choice (compare with `tests/bench_extractors.py` on bridge-entity
recall):

- **`GLiNER2Extractor`** (default) -- most flexible (zero-shot types + causal
  relations) and highest entity recall, but the heaviest (loads slowly, ~4.6 GB).
- **`OnnxTokenClassifierExtractor`** -- runs any BIO token-classification model
  exported to ONNX, decoding entities from the model's own `id2label`. Fast
  (~30 ms/call) and multilingual with a suitable model; the label scheme is the
  model's, so a specialized place model or a custom general NER both drop in
  with no code change.
- **`GlinerExtractor`** -- GLiNER v1 zero-shot with convert-and-cache ONNX
  inference (fast, flexible entity types; no causal). Defaults to
  `gliner-community/gliner_small-v2.5`, which on a 10-language WikiANN benchmark
  led on entity recall (**86%**, vs GLiNER2's 74%) at **~67 ms/call and ~2.2 GB**
  -- and unlike GLiNER2 it holds up on Korean/Arabic/Turkish/Russian. The
  checkpoint matters a lot: the older `urchade/gliner_multi-v2.1` scores ~12%,
  so pin the model and benchmark with `tests/bench_ner_multilingual.py`.

Size sweep (same WikiANN benchmark) -- bigger is not uniformly better:

| model | infer | RAM | recall | prec | F1 |
|---|---|---|---|---|---|
| `gliner_small-v2.5` | 67 ms | 2.2 GB | 86% | 73% | 79% |
| `gliner_medium-v2.5` | 73 ms | 2.7 GB | 84% | 75% | 79% |
| `gliner_large-v2.5` | 142 ms | 4.8 GB | 86% | 84% | 85% |
| GLiNER2 (default) | 250 ms | 4.8 GB | 74% | 84% | 79% |

Small ties large on recall; large's extra size buys precision (best F1). Medium is
dominated -- skip it. `large-v2.5` beats GLiNER2 outright (same precision, higher
recall, faster, far stronger on Arabic/Korean).

Running `gliner_small-v2.5` through ONNX (`GlinerExtractor(onnx=True)`) cuts
inference from ~67 ms to **~12 ms/call** with recall preserved -- the fastest
high-recall multilingual option (the conversion is cached on first use).

Rough guide: **`gliner_small-v2.5`** for the best speed/RAM at high recall (add
`onnx=True` for ~12 ms/call); **`gliner_large-v2.5`** for the best overall quality
and a strict upgrade over GLiNER2 on multilingual; **GLiNER2** only when you need
its causal-relation extraction; **place ONNX** for the fastest location-heavy path.

## Scopes

Scopes are free-text tags on facts (`"user-alice"`, `"topic-economy"`, `"session-42"`)
-- **not partitions**. The graph stays shared: a fact can carry several scopes at
once, and multi-hop reasoning follows shared entities across every scope. A scope
on `query()` only narrows where the search *seeds*; traversal still reaches
connected facts in other scopes.

```python
await graph.add_texts(alice_facts, scopes=["user-alice"])
await graph.add_texts(economy_facts, scopes=["topic-economy"])
# One fact can belong to several scopes at once
await graph.add_texts(shared, scopes=["user-alice", "topic-economy"])

# Seeds come from user-alice; reasoning still bridges into topic-economy facts
results = await graph.query("Will it get harder to afford a home?", scopes=["user-alice"])
```

Adding the same content under a new scope unions the tags (never drops the old
ones). Full demo: `uv run python examples/scoped_reasoning.py`

## Backends

By default, `ReasonGraph()` uses a pure Python in-memory backend (`MemoryBackend`). This works everywhere with zero dependencies beyond numpy. For persistence, pass a file path to save/load as JSON:

```python
from reasongraph import ReasonGraph, MemoryBackend

# In-memory only (default)
graph = ReasonGraph()

# In-memory with JSON file persistence (loads on init, saves on close)
graph = ReasonGraph(backend=MemoryBackend(file_path="graph.json"))
```

### SQLite Backend

For larger graphs or concurrent access, use the SQLite backend with `sqlite-vec` for vector search. Requires `pip install reasongraph[sqlite]`.

```python
from reasongraph import ReasonGraph
from reasongraph.backends import SqliteBackend

graph = ReasonGraph(backend=SqliteBackend(db_path="graph.db"))
```

### PostgreSQL Backend

```python
from reasongraph import ReasonGraph
from reasongraph.backends import PostgresBackend

graph = ReasonGraph(backend=PostgresBackend(database_url="postgresql://user:pass@localhost/db"))
```

Requires `pip install reasongraph[postgres]` and the `pgvector` + `pg_trgm` extensions enabled on your database.

## Evaluation: Mixed-Domain Reasoning

We evaluate reasoning quality by loading all 6 built-in datasets into a single graph (~130 text nodes, ~104 entity nodes, ~280 edges) and testing whether the library can trace the correct causal chains, syllogistic proofs, taxonomic hierarchies, and data analysis patterns -- without being distracted by unrelated facts from other domains.

32 test cases simulate agent-style queries like *"I need to understand what caused the 2008 financial crisis"*, *"How does insulin resistance lead to kidney failure?"*, or *"I have two numeric columns, check if related"* and check whether the returned reasoning chain matches the expected ground truth.

**Per-domain results (hybrid search, `top_k=5`, `hops=4`, `rerank_top_k=4`):**

| Domain | Cases | Chain Completeness | Recall@5 | Precision@5 | Domain Accuracy |
|--------|------:|--------------------|----------|-------------|-----------------|
| Causal | 5 | 100% | 100% | 92% | 100% |
| Financial | 6 | 100% | 82% | 60% | 100% |
| Medical | 5 | 100% | 92% | 76% | 92% |
| Syllogisms | 5 | 100% | 100% | 92% | 85% |
| Taxonomy | 3 | 100% | 83% | 53% | 92% |
| Analysis Patterns | 8 | 96% | 75% | 45% | 96% |
| **Overall** | **32** | **99%** | **88%** | **68%** | **95%** |

32/32 cases pass (>= 50% chain completeness). Split reranking gives chain continuations (text-to-text edges) priority over bridge discoveries (entity-to-text edges), keeping traversal focused.

**Search mode comparison:**

| Mode | Chain Completeness | Recall@5 | Precision@5 | Domain Accuracy |
|------|-------------------|----------|-------------|-----------------|
| Embedding | 99% | 88% | 68% | 95% |
| Keyword | 0% | 0% | 0% | 0% |
| Hybrid | 99% | 88% | 68% | 95% |

Keyword-only mode scores 0% because the eval queries are natural language questions that don't substring-match the dataset's declarative statements. This is expected -- keyword search is designed for known-term lookups, not question answering.

Reproduce: `uv run python tests/eval_financial_reasoning.py`

## API Reference

### `ReasonGraph(backend=None, embed_model=None, rerank_model=None, forget_after=30, forget_every=None)`

| Method | Description |
|--------|-------------|
| `add_nodes(nodes)` | Add `(content, type)` tuples to the graph |
| `add_edges(edges)` | Add `(from, to)` content edges |
| `add_text(text, extractor=None, scopes=None)` | Add text with automatic entity extraction, tagged with optional `scopes` |
| `add_texts(texts, extractor=None, causal_extractor=None, scopes=None)` | Batch add with entity + causal extraction (auto-enabled with GLiNER2) |
| `query(query, top_k=5, hops=4, rerank_top_k=4, search_mode="embedding", rrf_k=60, recency_weight=0.0, scopes=None)` | Search and traverse the graph; `recency_weight` in [0,1] blends recency into ranking; `scopes` narrows the seeds (traversal still crosses scopes) |
| `load_dataset(name)` | Load a built-in dataset |
| `delete_stale()` | Remove nodes not accessed within `forget_after` days |
| `maybe_forget()` | Throttled `delete_stale()`: sweeps at most once per `forget_every` seconds (no-op when `forget_every` is `None`) |
| `delete(content)` | Remove a single node and its incident edges by exact content |
| `supersede(old_content, new_text, extractor=None)` | Replace a stale fact: add `new_text`, then delete `old_content` |
| `get_all_nodes(scopes=None)` / `get_all_edges()` | Inspect graph contents (nodes optionally filtered by scope) |

All methods are async. Sync variants are available with a `_sync` suffix (e.g. `query_sync`).

`embed_model` accepts a model name (`str`), a `SentenceTransformer`, or any
object/callable that encodes text. The encoder must take a `str` (returning one
vector) or a `list[str]` (returning one vector per text); numpy, torch, or list
outputs are all accepted. This lets a host reuse an embedder it already runs
instead of loading a second stack:

```python
def encode(text_or_texts):
    # reuse your own embedding library; return list[float] or list[list[float]]
    ...

graph = ReasonGraph(embed_model=encode)
```

## License

MIT
