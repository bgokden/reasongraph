# reasongraph

A graph-based reasoning library with embedding search, multi-hop traversal, and automatic entity extraction.

[![PyPI version](https://img.shields.io/pypi/v/reasongraph)](https://pypi.org/project/reasongraph/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/reasongraph)](https://pypi.org/project/reasongraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Installation

```bash
pip install reasongraph[all]        # everything included
```

Or install only what you need:

```bash
pip install reasongraph             # core: SQLite backend, NER, embeddings
pip install reasongraph[postgres]   # + PostgreSQL + pgvector backend
pip install reasongraph[gliner2]    # + GLiNER2 entity extraction
```

## Quick Start

```python
from reasongraph import ReasonGraph

graph = ReasonGraph()
graph.initialize_sync()

# Add text with automatic NER entity extraction
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

- **Hybrid search** -- combine embedding similarity, keyword (trigram) matching, or both
- **Multi-hop traversal** -- follow graph edges to discover connected reasoning chains
- **Automatic NER extraction** -- extract entities from text using `dslim/bert-base-NER` (default) or GLiNER2
- **Causal relation extraction** -- detect cause-effect pairs with CausaLMiner
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

```python
graph.load_dataset_sync("financial")
```

## Search Modes

```python
# Pure embedding similarity (default)
results = graph.query_sync("credit freeze", search_mode="embedding")

# Pure keyword/trigram matching
results = graph.query_sync("credit freeze", search_mode="keyword")

# Hybrid: weighted combination (embedding_weight controls the balance)
results = graph.query_sync("credit freeze", search_mode="hybrid", embedding_weight=0.7)
```

## Entity Extraction

```python
from reasongraph import ReasonGraph, NERExtractor, GLiNER2Extractor

graph = ReasonGraph()
graph.initialize_sync()

# Default: BERT NER (dslim/bert-base-NER)
entities = graph.add_text_sync("Apple released the iPhone in 2007.")
print(entities)  # ['Apple', 'iPhone']

# Custom: GLiNER2 with configurable entity types
gliner = GLiNER2Extractor(labels=["company", "product", "date"])
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

We evaluate reasoning quality by loading all 4 built-in datasets into a single graph (75 text nodes, 76 entity nodes, 157 edges) and testing whether the library can trace the correct causal chains, syllogistic proofs, and taxonomic hierarchies -- without being distracted by unrelated facts from other domains.

15 test cases simulate agent-style queries like *"I need to understand what caused the 2008 financial crisis"* or *"Is Socrates mortal? What is the logical reasoning?"* and check whether the returned reasoning chain matches the expected ground truth.

**Per-domain results (hybrid search, 3 hops):**

| Domain | Cases | Chain Completeness | Recall@5 | Precision@5 | Domain Accuracy |
|--------|------:|--------------------|----------|-------------|-----------------|
| Financial | 6 | 77% | 71% | 65% | 100% |
| Causal | 3 | 89% | 89% | 80% | 100% |
| Syllogisms | 3 | 100% | 100% | 92% | 92% |
| Taxonomy | 3 | 72% | 72% | 64% | 83% |
| **Overall** | **15** | **83%** | **80%** | **73%** | **95%** |

All 15/15 cases pass (>= 50% chain completeness). 8 out of 15 cases achieve 100% chain completeness. Domain accuracy of 95% means queries almost never return results from the wrong knowledge domain.

**Search mode comparison:**

| Mode | Chain Completeness | Recall@5 | Precision@5 | Domain Accuracy |
|------|-------------------|----------|-------------|-----------------|
| Embedding | 80% | 80% | 74% | 95% |
| Keyword | 89% | 82% | 62% | 82% |
| Hybrid | 83% | 80% | 73% | 95% |

Reproduce: `uv run python tests/eval_financial_reasoning.py`

## API Reference

### `ReasonGraph(backend=None, embed_model=None, rerank_model=None, forget_after=30)`

| Method | Description |
|--------|-------------|
| `add_nodes(nodes)` | Add `(content, type)` tuples to the graph |
| `add_edges(edges)` | Add `(from, to)` content edges |
| `add_text(text, extractor=None)` | Add text with automatic entity extraction |
| `add_texts(texts, extractor=None, causal_extractor=None)` | Batch add with NER + optional causal extraction |
| `query(query, top_k=5, hops=2, search_mode="embedding")` | Search and traverse the graph |
| `load_dataset(name)` | Load a built-in dataset |
| `delete_stale()` | Remove nodes not accessed within `forget_after` days |
| `get_all_nodes()` / `get_all_edges()` | Inspect graph contents |

All methods are async. Sync variants are available with a `_sync` suffix (e.g. `query_sync`).

## License

MIT
