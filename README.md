# reasongraph

A graph-based reasoning library with embedding search, multi-hop traversal, and automatic entity/causal extraction.

[![PyPI version](https://img.shields.io/pypi/v/reasongraph)](https://pypi.org/project/reasongraph/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/reasongraph)](https://pypi.org/project/reasongraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Installation

```bash
pip install reasongraph[all]        # everything included
```

Or install only what you need:

```bash
pip install reasongraph             # core: SQLite backend, NER extraction, embeddings
pip install reasongraph[gliner2]    # + GLiNER2 entity + causal extraction (recommended)
pip install reasongraph[postgres]   # + PostgreSQL + pgvector backend
```

## Quick Start

```python
from reasongraph import ReasonGraph

graph = ReasonGraph()
graph.initialize_sync()

# Add text with automatic entity + causal extraction
graph.add_text_sync("Lehman Brothers filed for bankruptcy in September 2008.")
graph.add_text_sync("The Federal Reserve cut interest rates to near zero.")

# Query with embedding search + multi-hop graph traversal
results = graph.query_sync("What caused the 2008 financial crisis?")
for text in results:
    print(text)

graph.close_sync()
```

Or use the async API with a context manager:

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

- **Automatic extraction** -- GLiNER2 extracts entities and causal relations in one pass (falls back to BERT NER when gliner2 is not installed)
- **Hybrid search** -- combine embedding similarity, keyword (trigram) matching, or both
- **Multi-hop traversal** -- follow graph edges to discover connected reasoning chains
- **Cross-encoder reranking** -- rerank results at each hop with `ms-marco-MiniLM-L-6-v2`
- **Built-in datasets** -- load curated reasoning graphs for immediate use
- **Async-first** -- native async API with sync convenience wrappers
- **Pluggable backends** -- SQLite (zero-config default) or PostgreSQL with pgvector

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

# Any callable works
entities = graph.add_text_sync("some text", extractor=lambda t: ["custom"])
```

## PostgreSQL Backend

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

### `ReasonGraph(backend=None, embed_model=None, rerank_model=None, forget_after=30)`

| Method | Description |
|--------|-------------|
| `add_nodes(nodes)` | Add `(content, type)` tuples to the graph |
| `add_edges(edges)` | Add `(from, to)` content edges |
| `add_text(text, extractor=None)` | Add text with automatic entity extraction |
| `add_texts(texts, extractor=None, causal_extractor=None)` | Batch add with entity + causal extraction (auto-enabled with GLiNER2) |
| `query(query, top_k=5, hops=4, rerank_top_k=4, search_mode="embedding", rrf_k=60)` | Search and traverse the graph |
| `load_dataset(name)` | Load a built-in dataset |
| `delete_stale()` | Remove nodes not accessed within `forget_after` days |
| `get_all_nodes()` / `get_all_edges()` | Inspect graph contents |

All methods are async. Sync variants are available with a `_sync` suffix (e.g. `query_sync`).

## License

MIT
