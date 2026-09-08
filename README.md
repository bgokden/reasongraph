# ReasonGraph

A graph-based **memory for AI agents**: it ingests facts, auto-extracts entities and cause->effect relations, and discovers connections across independent documents *and* across agent sessions -- with conflict resolution, time-travel, causal tracing, and counterfactuals.

[![PyPI version](https://img.shields.io/pypi/v/reasongraph?color=blue)](https://pypi.org/project/reasongraph/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/reasongraph?color=blue)](https://pypi.org/project/reasongraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Why ReasonGraph?

Standard RAG retrieves documents similar to your query. ReasonGraph is a persistent, updatable memory that discovers connections *between* facts that were written independently.

When you feed text into `add_texts()`, ReasonGraph automatically extracts **entities** (via GLiNER) and **cause-effect relations** (via a dedicated causal model) that become nodes and typed edges in a graph. Facts that share entities or causal chains get connected -- even if they never reference each other. Multi-hop traversal then walks these connections to build reasoning chains that span multiple sources.

On top of retrieval it works as agent memory: **scopes/sessions** (agents discover into each other's memory through shared entities), **contradiction resolution** (a new fact soft-supersedes what it contradicts), **time-travel** (`query(as_of=...)`), **causal tracing** (`trace_effects` / `root_causes` / `causal_chain`), **counterfactuals** (`what_if`), and a shippable **MemoryService** over HTTP and MCP.

**Zero config, strong defaults.** `ReasonGraph()` picks the best available entity extractor, causal model, embedder, and reranker automatically -- the eval numbers below come from these defaults. For the SOTA causal model (~0.70 F1) use `pip install reasongraph[causal]` and the graph uses it automatically. The configuration sections are optional depth, not required reading.

## Use it in 60 seconds

**Claude Code / Cursor / any MCP client, hosted (EU, no LLM in the loop):**

```bash
claude mcp add --transport http memory https://memory.primaxiom.ai/mcp \
  --header "Authorization: Bearer rgm_YOUR_KEY"
```

**Python, in-process:**

```bash
pip install "reasongraph[all]"
```

```python
from reasongraph import ReasonGraph

graph = ReasonGraph()
graph.initialize_sync()
graph.add_texts_sync(["TSMC is building a chip fab in Phoenix, Arizona.",
                      "Arizona ordered water cuts for industrial users in Maricopa County."])
print(graph.discover_sync("water and chips"))   # a path: water cuts -> Arizona -> TSMC fab
```

**Any language, over HTTP (self-hosted or hosted):**

```bash
curl -X POST https://memory.primaxiom.ai/sessions/notes/memory \
  -H "Authorization: Bearer rgm_YOUR_KEY" -H "Content-Type: application/json" \
  -d '{"text": "Apple sources M-series chips from TSMC in Arizona."}'
```

Ready-to-copy agents (Groq/OpenAI-compatible research agent, two agents sharing one
memory, Claude Code with persistent memory, LangGraph) live in
[`examples/agents/`](examples/agents/).

### Hosted: ReasonGraph Cloud

[memory.primaxiom.ai](https://memory.primaxiom.ai) runs this library as a service: sign in,
get a free key (10k requests a month), remote MCP endpoint, browser console and playground.
Extraction runs with small models on servers PrimAxiom operates (currently in the EU); facts are
only sent to an LLM provider if you ask for a synthesized answer. Early access.

## Installation

```bash
pip install reasongraph[all]        # everything included
```

Or install only what you need:

```bash
pip install reasongraph             # core: in-memory backend, NER extraction, embeddings
pip install reasongraph[gliner]     # + GLiNER entity extraction + hybrid causal (default, recommended)
pip install reasongraph[causal]     # + SOTA span-pointer causal model (~0.70 F1) + hybrid fallback
pip install reasongraph[gliner2]    # + GLiNER2 alternative (single model does entities + causal)
pip install reasongraph[sqlite]     # + SQLite backend with sqlite-vec
pip install reasongraph[postgres]   # + PostgreSQL + pgvector backend
pip install reasongraph[service]    # + HTTP + MCP memory service
pip install reasongraph[fastembed]  # + pure-ONNX embedder / reranker (faster cold start)
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

**ReasonGraph extracts entities and causal relations from each text** (requires an entity+causal extractor, e.g. `pip install reasongraph[gliner]` or `[all]`)**:**

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
- **Automatic extraction** -- entities (GLiNER `gliner_small-v2.5` by default) and cause->effect relations (a dedicated span-pointer / hybrid causal model) are extracted on add, both on by default; falls back to GLiNER2 then BERT NER when `gliner` is not installed
- **Agent memory** -- scopes/sessions with cross-session discovery, contradiction resolution (soft-supersede), time-travel (`as_of`), semantic dedup, and auto-forget
- **Causal reasoning** -- trace downstream effects, root causes, and directed causal paths; ask counterfactual `what_if`
- **Hybrid search** -- combine embedding similarity, keyword (trigram) matching, or both
- **Multi-hop traversal** -- follow graph edges to discover connected reasoning chains
- **Cross-encoder reranking** -- rerank results at each hop with `ms-marco-MiniLM-L-6-v2`
- **Memory service** -- ready HTTP + MCP server so agents share and query memory
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

The default (`embedding`) is the best general choice and matches `hybrid` on the eval
below; keyword is for known-term lookups. You rarely need to change this.

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

Entity extraction and causal extraction are **two independent, both-on-by-default** capabilities. `add_text()` / `add_texts()` use **`gliner_small-v2.5`** for entities (fast, multilingual, highest entity recall) when `gliner` is installed, falling back to GLiNER2 then BERT NER. Override per call with the `extractor` argument -- e.g. `gliner_large-v2.5` for higher precision.

```python
from reasongraph import ReasonGraph, NERExtractor, GLiNER2Extractor

graph = ReasonGraph()
graph.initialize_sync()

# Default: GLiNER gliner_small-v2.5 for entities (+ the default causal model),
# falling back to GLiNER2 then BERT NER
entities = graph.add_text_sync("Apple released the iPhone in 2007.")
print(entities)  # ['Apple', 'iPhone']

# Explicit: force BERT NER even if a GLiNER model is installed
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

### Causal reasoning (default on)

Causality is the headline feature, so causal extraction runs **by default**
(opt out per call with `causal=False`). Directed cause->effect relations become
first-class **typed edges** (`label="causes"`) in the graph, distinct from
anonymous entity bridges, and `discover()` returns them per fact:

```python
graph.add_text_sync("Heavy rainfall caused severe flooding.")
# -> typed edge  heavy rainfall --causes--> severe flooding

for fact in graph.discover_sync("flooding"):
    print(fact["content"], fact["causes"])  # [{'cause': 'Heavy rainfall', 'effect': 'severe flooding'}]
```

### Causal chain tracing

Because cause->effect edges are directed and first-class, you can **walk the causal
graph** -- something a flat vector store cannot do. Trace downstream impact, trace
back to root causes, or find a directed causal path between two facts:

```python
graph.add_texts_sync([
    "Heavy rainfall caused flooding.",
    "Flooding caused power outages.",
    "Power outages caused hospital disruptions.",
])

graph.trace_effects_sync("Heavy rainfall caused flooding.")["terminals"]
# e.g. -> ['hospital disruptions']       # downstream impact

graph.trace_causes_sync("Power outages caused hospital disruptions.")["terminals"]
# e.g. -> ['rainfall']                   # upstream causes (same as root_causes_sync)

graph.causal_chain_sync("Heavy rainfall caused flooding.",
                        "Power outages caused hospital disruptions.")
# -> ordered causal hops, each cited to the fact that asserted it
```

The exact spans depend on the causal extractor; a hop chains when one fact's effect
span matches the next fact's cause span. Each hop is tagged with the fact that
asserts it, its scopes, and a `cross_session` flag; with a `conflict_resolver`
configured, retired (superseded) facts are skipped by default (`include_superseded=True`
keeps them). Tunable with `max_depth` (default 6) and `max_visited` (default 1000).
Exposed to agents as the `trace_memory` MCP tool and the `/trace` HTTP endpoint.

### Counterfactual: what breaks if a fact were false

Because the causal edges are first-class, you can ask the inverse of a trace:
**if one fact were false, which downstream effects collapse?** `what_if` prunes a
fact hypothetically (no graph mutation), re-walks reachability, and reports which
effect spans lost **all** causal support versus which **survived** via an alternate
path. Only edges the pruned fact *solely* supports are removed -- an effect another
fact also explains still stands.

```python
graph.what_if_sync("Flooding caused power outages.")
# {
#   'pruned': 'Flooding caused power outages.',
#   'origin': 'Flooding caused power outages.',     # walk start (== pruned unless origin= given)
#   'pruned_edges': [{'cause': 'flooding', 'effect': 'power outages'}],
#   'collapsed': [                                 # lost their only causal path
#       {'span': 'power outages', 'fact': 'Flooding caused power outages.', 'depth': 0, ...},
#       {'span': 'hospital disruptions', 'fact': 'Power outages caused hospital disruptions.', 'depth': 1, ...},
#   ],
#   'survived': [],                                # spans an alternate path rescued
# }
```

Pass `origin=` to measure collapse relative to an upstream fact, or
`direction='causes'` to see which upstream causes become orphaned. Exposed as the
`what_if_memory` MCP tool and the `/what_if` HTTP endpoint.

The default causal extractor picks the **best available** backend. When the
`causal-span-model` package is installed it uses the **span-pointer model**
(`CausalPointerExtractor`): a fine-tuned mDeBERTa-v3 that scores **~0.70 F1** on the
Causal News Corpus Subtask-2 official scorer -- beating the 0.627 organizer baseline,
the hybrid, and a few-shot LLM baseline (~0.24-0.41). It is trained on English but
multilingual at inference (script-aware segmentation, verified on es/fr/de/pt/tr/ru/ar
and zh/ja) and has a built-in causal gate, so it returns nothing on non-causal text.

The built-in gate can be replaced by a **decoupled embedding gate**: a small
classifier on sentence embeddings (train one with `scripts/train_embed_gate.py` in
causal-span-model; it saves a `.joblib`). It costs a millisecond per sentence, is
retrained on any negative mix without touching the span heads, and on our causal
eval it gave fewer, more precise edges than the built-in gate. Pass a local path or
an `hf://owner/repo/file.joblib` reference:

```python
ReasonGraph(causal_extractor=CausalPointerExtractor(
    model="Berk/causal-span-pointer-v2", gate_threshold=1.0,          # built-in gate off
    embed_gate="hf://Berk/causal-span-pointer-v2/embed_gate_mlp.joblib",
    embed_gate_threshold=0.9))                                        # keep P(causal) >= 0.9
```

`causal_chain` also bridges facts that phrase one event differently ("the system
throttles performance" -> "Throttling performance", or a plain root fact whose
words reappear in the next cause span), so directed chains survive wording changes
even without `span_link_threshold`.

Otherwise it falls back to the **hybrid** (`HybridCausalExtractor`): a fast,
model-free multilingual **cue pass** handles explicit and reversed phrasing with
correct direction, and sentences with no causal connective (implicit causality)
fall through to **`gliner-relex-multi`** (Apache-2.0, mDeBERTa, ~100 languages).
On a four-regime probe set (explicit / multilingual / implicit / reversed) the
hybrid reached **100% directed-pair recall vs 61-79%** for either part alone --
each covers the other's blind spot -- and most sentences never touch the model,
so the average cost is low. Reproduce with `tests/bench_causal_extractors.py`.

```python
from reasongraph import CausalPointerExtractor, HybridCausalExtractor, GlinerRelexExtractor

ReasonGraph()                                          # best available (pointer if installed, else hybrid)
ReasonGraph(causal_extractor=CausalPointerExtractor())  # force the span-pointer model
ReasonGraph(causal_extractor=HybridCausalExtractor())  # force the hybrid
ReasonGraph(causal_extractor=GlinerRelexExtractor())   # relex model only
ReasonGraph(causal_extractor=False)                    # disable causal extraction
```

`pip install reasongraph[causal]` installs both the pointer model
(`causal-span-model`) and the hybrid (`gliner`), so the graph uses the SOTA pointer
by default and falls back to the hybrid automatically. If neither is available the
default warns once rather than silently dropping causality; `add_text(..., causal=True)`
raises when no causal extractor can be resolved.

### Sentence splitting at ingest

Every model in the pipeline is trained on single sentences, so a paragraph pushed as one
fact hurts entities, causal spans and retrieval alike (on our 39-case causal eval, chain
recall drops from 79% to 10%). Pass a splitter and each text becomes one fact per sentence:

```python
ReasonGraph(sentence_splitter="sat")          # Segment-any-Text, 85 languages: pip install reasongraph[split]
ReasonGraph(sentence_splitter="regex")        # dependency-free fallback (punctuation + newlines)
graph.add_texts([paragraph], split=True)      # or per call; split=False keeps a text whole
```

The service reads `REASONGRAPH_SPLIT_SENTENCES=sat|regex`; pushes accept `split: true/false`.

## Deep memory integration: the memory loop

No tools, no prompts to write: wrap any chat model and every exchange becomes memory, and
whatever is relevant comes back by itself before the next call.

```python
from reasongraph import ReasonGraph, MemoryLoop

graph = ReasonGraph()                                  # or your Postgres-backed graph
loop = MemoryLoop(graph, session="support-chat", max_facts=8)

history = [{"role": "user", "content": "Why did the Rotterdam warehouse lose power?"}]
reply, context = loop.chat_sync(call_model, history, system="You are a careful assistant.")
# call_model is any fn(messages) -> str: OpenAI-compatible, Claude, Ollama, llama.cpp
# context.facts  -> what was recalled (with sources and cause->effect links)
# context.roots  -> for why-questions, the root cause(s) the chain walks back to;
#                   they are spelled out in the injected block so a small model
#                   answers with the root, not only the nearest cause
# the question and the reply are now remembered in "support-chat"
```

`loop.messages(history)` returns the message list with the recalled facts injected as a
system message, if you want to call the model yourself; `loop.observe(user, assistant)`
stores an exchange. Options: `max_facts` / `max_chars` (context budget), `min_score`
(no unrelated filler), `rerank_min` (an optional cross-encoder cutoff on top of it:
cosine cannot tell "same topic" from "answers this", the reranker can; -4 with the
default reranker), `extend_query` (when a why-question's chain ends in a root cause
no recalled fact states, one more targeted query fetches the plain fact behind it),
`observe_user` / `observe_assistant`, `redact` (a function that
drops or rewrites text before it is stored), `resolve_conflicts`. The hosted service
exposes the same loop as `POST /chat`. Example agent: `examples/agents/memory_loop_agent.py`.

## Fast inference (optional, pure ONNX)

The defaults already deliver the eval quality below; this is purely a
speed/memory optimization. Every model slot is pluggable, so you can trade the
PyTorch defaults for CPU-optimized ONNX models at equal-or-better quality.
Measured on the 32-case mixed-domain eval:

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

### Choosing an extractor (optional)

You don't need to choose -- the default (`gliner_small-v2.5` for entities plus the
default causal model) is the recommended, benchmarked setup. This section is the
evidence behind that default and the alternatives for special cases; swap the entity
model with the `extractor` argument if you have a specific need (reproduce the numbers
with `tests/bench_extractors.py`):

- **`GlinerExtractor`** (default) -- GLiNER v1 zero-shot with convert-and-cache ONNX
  inference (fast, flexible entity types; entities only -- causal relations come from
  the separate default causal model). Defaults to `gliner-community/gliner_small-v2.5`,
  which on a 10-language WikiANN benchmark led on entity recall (**86%**, vs GLiNER2's
  74%) at **~67 ms/call and ~2.2 GB** -- and unlike GLiNER2 it holds up on
  Korean/Arabic/Turkish/Russian. The checkpoint matters a lot: the older
  `urchade/gliner_multi-v2.1` scores ~12%, so pin the model and benchmark with
  `tests/bench_ner_multilingual.py`.
- **`GLiNER2Extractor`** -- a single model that does entity types **and** causal
  relations in one pass. Reach for it when you want one model for both, but it is the
  heaviest (loads slowly, ~4.6 GB) and lower on multilingual entity recall.
- **`OnnxTokenClassifierExtractor`** -- runs any BIO token-classification model
  exported to ONNX, decoding entities from the model's own `id2label`. Fast
  (~30 ms/call) and multilingual with a suitable model; the label scheme is the
  model's, so a specialized place model or a custom general NER both drop in
  with no code change.

Size sweep (same WikiANN benchmark) -- bigger is not uniformly better:

| model | infer | RAM | recall | prec | F1 |
|---|---|---|---|---|---|
| `gliner_small-v2.5` | 67 ms | 2.2 GB | 86% | 73% | 79% |
| `gliner_medium-v2.5` | 73 ms | 2.7 GB | 84% | 75% | 79% |
| `gliner_large-v2.5` | 142 ms | 4.8 GB | 86% | 84% | 85% |
| `knowledgator/gliner-x-base` | 151 ms | 4.2 GB | 87% | 79% | 83% |
| GLiNER2 | 250 ms | 4.8 GB | 74% | 84% | 79% |

Small ties large on recall; large's extra size buys precision (best F1). Medium is
dominated -- skip it. `large-v2.5` beats GLiNER2 outright (same precision, higher
recall, faster, far stronger on Arabic/Korean). `knowledgator/gliner-x-base`
(20+ languages) edges recall/precision above `small-v2.5` but needs `stanza` +
`langdetect` (with per-language models fetched at runtime), runs ~7x slower, and
is no better on the WikiANN Chinese reconstruction -- so `small-v2.5` stays the
default; reach for `x-base` only when precision matters more than latency.

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
ones). For a hard boundary where a query can only reach its own scope's facts, pass
`isolate=True` (see [Multi-tenant and production](#multi-tenant-and-production)).
Full demo: `uv run python examples/scoped_reasoning.py`

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

### `ReasonGraph(backend=None, embed_model=None, rerank_model=None, forget_after=30, forget_every=None, synthesizer=None, causal_extractor=None, isolate_traversal=False, conflict_resolver=None)`

- `causal_extractor`: `None` builds the best available causal extractor lazily (the span-pointer model when `causal-span-model` is installed, else the hybrid); `False` disables causal extraction; a callable/object with `extract_causal` uses it.
- `isolate_traversal`: graph-wide default for whether a scoped query confines traversal to its scopes (multi-tenant). Off keeps cross-scope discovery; override per call with `query(..., isolate=...)`.
- `conflict_resolver`: enables contradiction resolution (soft-supersede) on write and retired-fact filtering on read (see [Multi-tenant and production](#multi-tenant-and-production)).

**Ingest**

| Method | Description |
|--------|-------------|
| `add_nodes(nodes, scopes=None)` | Add `(content, type)` tuples to the graph |
| `add_edges(edges)` | Add `(from, to)` or `(from, to, label)` content edges (label e.g. `"causes"`) |
| `add_text(text, extractor=None, scopes=None, causal_extractor=None, causal=None, dedup_threshold=None, resolve_conflicts=None)` | Add text with entity + causal extraction; `causal=False` disables, `True` forces (raises if unavailable); `dedup_threshold` drops near-duplicates (unioning scopes); `resolve_conflicts` soft-supersedes contradicted facts when a `conflict_resolver` is set |
| `add_texts(texts, extractor=None, causal_extractor=None, scopes=None, causal=None, dedup_threshold=None, resolve_conflicts=None)` | Batch form of `add_text` (causal on by default) |

**Retrieve**

| Method | Description |
|--------|-------------|
| `query(query, top_k=5, hops=4, rerank_top_k=4, search_mode="embedding", rrf_k=60, recency_weight=0.0, scopes=None, isolate=None, include_superseded=False, as_of=None)` | Search and traverse the graph. `recency_weight` in [0,1] blends recency into ranking; `scopes` narrows the seeds (traversal still crosses scopes unless `isolate=True`); `as_of=<datetime>` time-travels to what was current then; `include_superseded=True` keeps retired facts |
| `query_detailed(...)` | Same signature as `query`, but returns `{content, score, created_at, scopes}` per hit for thresholding/dedup |
| `discover(query, top_k=5, hops=4, search_mode="embedding", rrf_k=60, scopes=None, max_results=10, max_visited=1000, isolate=None, include_superseded=False)` | Like `query`, but returns *connection paths* -- how each fact links back to a seed via bridging entities, tagged with scopes, flagging cross-session links, and listing each fact's directed `causes` relations. The walk stops after `max_visited` nodes |
| `answer(query, use_discover=True, top_k=5, hops=4, search_mode="embedding", scopes=None, max_results=10)` | Rephrase the retrieved facts/paths into logical free text via the pluggable `synthesizer` |

**Causal reasoning**

| Method | Description |
|--------|-------------|
| `trace_effects(content, max_depth=6, scopes=None, isolate=None, include_superseded=False, max_visited=1000)` | Forward causal walk: `{origin, chain, terminals}` for downstream impact |
| `trace_causes(content, ...)` | Backward causal walk: what led to `content` |
| `root_causes(content, ...)` | The root cause spans behind `content` (backward-walk terminals) |
| `causal_chain(from_content, to_content, max_depth=6, scopes=None, isolate=None, include_superseded=False)` | Ordered causal hops linking two facts, or `None` |
| `what_if(content, origin=None, direction="effects", max_depth=6, scopes=None, isolate=None, include_superseded=False, max_visited=1000)` | Counterfactual: prune a fact and report `collapsed` vs `survived` downstream spans |

**Update, forget, temporal**

| Method | Description |
|--------|-------------|
| `delete(content, purge_orphans=False)` | Remove a node and its incident edges by exact content; `purge_orphans=True` also removes entities left dangling |
| `supersede(old_content, new_text, extractor=None, purge_orphans=False)` | Replace a stale fact: add `new_text`, then delete `old_content` |
| `supersession_history(content)` | Audit `{supersedes, superseded_by}` for a fact |
| `delete_stale()` | Remove nodes not accessed within `forget_after` days |
| `maybe_forget()` | Throttled `delete_stale()`: sweeps at most once per `forget_every` seconds (no-op when `forget_every` is `None`) |

**Datasets and inspection**

| Method | Description |
|--------|-------------|
| `load_dataset(name)` | Load a built-in dataset |
| `get_all_nodes(scopes=None)` / `get_all_edges()` | Inspect graph contents (nodes optionally filtered by scope) |

Lifecycle is `initialize()` / `close()`, or use `async with ReasonGraph() as graph:`. All methods are async; every one has a `_sync` twin with the same parameters (e.g. `query_sync`, `what_if_sync`, `add_text_sync`).

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

## Agent memory service

A ready service turns reasongraph into shared, discoverable memory for many
agents. **Knowledge sessions are scopes**: an agent pushes memory into its
session, and a query seeds from that session but traversal crosses all sessions
-- so agents **discover connections into each other's memory** through shared
entities. `pip install reasongraph[service]`.

```python
from reasongraph.service import MemoryService
from reasongraph.backends import PostgresBackend

service = MemoryService(backend=PostgresBackend("postgresql:///memory"),
                        synthesizer=my_small_llm)   # synthesizer is optional

await service.push("research-bot", "TSMC is building a chip fab in Arizona.")
await service.push("news-bot", "Arizona declared a water emergency.")

# research-bot discovers news-bot's fact via the shared 'Arizona' entity
paths = await service.discover("Arizona", session="research-bot")
answer = await service.answer("Arizona", session="research-bot")   # logical free text
```

Expose it over **HTTP** (`reasongraph.service.http.create_app`) or **MCP**
(`reasongraph.service.mcp_server.create_mcp`) -- the HTTP `query`/`discover`
endpoints take a `synthesize` flag that adds the free-text `answer`. The demo
`uv run python examples/agent_memory_service.py` pushes an economy / supply-chain
/ energy / health / policy world across five agent sessions and shows a markets
query reaching a Taiwan drought and a chip fab recorded by other agents.

**HTTP endpoints**: `POST /sessions/{session}/memory` (and `/batch`; pass
`resolve_conflicts: true` to check nearby facts for contradictions), `/query` (supports
`as_of` and `include_superseded` for time-travel), `/discover`, `/causal_chain`,
`/supersede`, `/delete`, `/history`, `/trace`, `/what_if`, `/forget`, `GET /sessions`,
`/stats`, plus `/health` and `/ready` probes.

**MCP tools**: `push_memory`, `query_memory`, `query_memory_detailed`,
`discover_connections`, `causal_chain_memory`, `trace_memory`, `what_if_memory`, `answer`,
`update_memory`, `delete_memory`, `memory_history`, `forget_stale`, `list_sessions`.

### Synthesizers

`answer()` and the `synthesize` flag rephrase retrieved facts and their
connection paths into logical free text. The core stays model-free -- pass any
`callable(query, context) -> str`, or one of the shipped adapters:

```python
from reasongraph import TemplateSynthesizer, PromptSynthesizer, TransformersSynthesizer

ReasonGraph(synthesizer=TemplateSynthesizer())            # deterministic, no model
ReasonGraph(synthesizer=PromptSynthesizer(my_generate))   # bring any LLM: generate(prompt) -> str
ReasonGraph(synthesizer=TransformersSynthesizer())        # local small LLM (Qwen2.5-0.5B-Instruct)
```

`PromptSynthesizer` builds the prompt (question + facts + cross-session bridges)
and calls your `generate` (sync or async); `TransformersSynthesizer` runs a small
instruct model locally on the torch/transformers stack already pulled in by
`sentence-transformers`.

### Deploy

An env-driven entrypoint wires the backend, embedder, and synthesizer from
environment variables. Run the Postgres-backed stack with Docker:

```bash
docker compose up --build          # Postgres (pgvector) + the service on :8000
curl localhost:8000/stats
```

Or serve directly (`pip install reasongraph[service]` adds the `reasongraph-serve`
console script and the ASGI factory):

```bash
REASONGRAPH_BACKEND=postgres \
REASONGRAPH_DATABASE_URL=postgresql:///memory \
REASONGRAPH_SYNTHESIZER=template \
reasongraph-serve            # or: uvicorn reasongraph.service.app:create_app_from_env --factory
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `REASONGRAPH_BACKEND` | `memory` | `memory` \| `sqlite` \| `postgres` |
| `REASONGRAPH_DATABASE_URL` | -- | Postgres URL, sqlite path, or memory JSON path |
| `REASONGRAPH_EMBED_MODEL` | built-in | Model name; prefix `fastembed:` for pure-ONNX |
| `REASONGRAPH_SYNTHESIZER` | `template` | `none` \| `template` \| `transformers` |
| `REASONGRAPH_SYNTH_MODEL` | built-in | Instruct model for the `transformers` synthesizer |
| `REASONGRAPH_FORGET_AFTER` / `REASONGRAPH_FORGET_EVERY` | `30` / off | Auto-forget window (days) and sweep interval (seconds). When the interval is set the service runs the sweep on a background task. |
| `REASONGRAPH_ISOLATE` | off | Confine traversal to the query session (multi-tenant). Off keeps cross-session discovery. |
| `REASONGRAPH_RESOLVE_CONFLICTS` | off | Enable contradiction resolution (soft-supersede) with the default NLI resolver. |
| `REASONGRAPH_API_KEY` | -- | When set, data endpoints require it (`Authorization: Bearer` or `X-API-Key`); `/health` and `/ready` stay open. |
| `REASONGRAPH_DEFER_EXTRACT` | off | Run entity/causal extraction in a background worker (off the event loop) so pushes return immediately. |
| `REASONGRAPH_SPAN_LINK_THRESHOLD` | off | Cosine threshold (e.g. `0.85`) above which a new cause/effect span is tied (`same_as`) to an existing causal span, so `trace_*` / `causal_chain` cross facts that phrase the same event differently. |
| `REASONGRAPH_CAUSAL_MODEL` | `berk/causal-span-pointer-mdeberta` | HF repo id or local dir of the span-pointer model. |
| `REASONGRAPH_CAUSAL_GATE_THRESHOLD` | `0.5` | Built-in gate: P(non-causal) above which the pointer abstains; `1.0` turns it off. |
| `REASONGRAPH_CAUSAL_EMBED_GATE` | off | Path or `hf://owner/repo/file.joblib` of an embedding-gate classifier; texts under `REASONGRAPH_CAUSAL_EMBED_GATE_THRESHOLD` (default `0.9`) get no relations. |
| `REASONGRAPH_DEDUP_THRESHOLD` | off | Cosine threshold (e.g. `0.95`) above which a pushed fact is treated as a paraphrase of an existing one: scopes are unioned, nothing new is stored. |
| `REASONGRAPH_HOST` / `REASONGRAPH_PORT` | `0.0.0.0` / `8000` | Bind address and port for `reasongraph-serve`. |

Use a persistent backend (PostgresBackend) for real multi-agent concurrency.

### Multi-tenant and production

By default the graph is shared: a scoped query seeds from its session but the walk
crosses sessions, which is the cross-session discovery feature. For a
confidentiality boundary between tenants, turn on **traversal isolation** so a query
can only reach its own session's facts:

```python
ReasonGraph(isolate_traversal=True)               # graph-wide default
graph.query("...", scopes={"tenant-a"}, isolate=True)   # or per query
```

Other production controls:

- **Auth**: `create_app(service, api_key="...")` (or `REASONGRAPH_API_KEY`) gates every
  data endpoint; the MCP server exposes the same tools.
- **Structured results**: `query_detailed(...)` (and the HTTP `detailed` flag /
  `query_memory_detailed` MCP tool) return `{content, score, created_at, scopes}` for
  thresholding, dedup, and "remembered on <date>".
- **Erasure**: `delete(content, purge_orphans=True)` / `supersede(..., purge_orphans=True)`
  also remove entities left dangling by the deletion (right-to-be-forgotten); shared
  entities survive. Exposed as `delete_memory` / `update_memory` MCP tools.
- **Semantic dedup**: `add_text(..., dedup_threshold=0.95)` drops near-duplicate
  restatements instead of accumulating them, unioning scopes onto the kept fact.
- **Contradiction resolution**: `ReasonGraph(conflict_resolver=NLIConflictResolver())`
  (or `REASONGRAPH_RESOLVE_CONFLICTS=1`) soft-supersedes facts a new one contradicts
  -- a `supersedes` edge drops the old fact from default `query`/`discover` recall
  while keeping it auditable and retrievable via `include_superseded=True`;
  `supersession_history(fact)` shows what replaced what. The resolver is pluggable
  (any object with `contradictions(new, candidates)`): the `NLIConflictResolver`
  cross-encoder needs no LLM but can over-flag complements (it treats "works in
  Munich" vs "lives in Berlin" as a conflict), while `LLMConflictResolver(generate)`
  brings any LLM and is more precise. Soft-supersede is deliberately reversible:
  re-asserting a fact revives it, and `query(..., as_of=<datetime>)` **time-travels**
  to what was current at that moment (each fact carries a `created_at`/`invalid_at`
  validity interval). Exposed to agents as the `memory_history` MCP tool and
  `/history` endpoint.
- **Health**: `/health` (liveness) and `/ready` (readiness) for orchestration probes.
- **Postgres** creates an HNSW cosine index, so vector search is index-accelerated
  rather than a sequential scan.
- **Speed**: `tests/bench_speed.py` measures write throughput and read latency
  (query / discover / trace, p50/p95) at a configurable graph size and backend
  (`--fake` for model-free timing). Indicative real-model, in-memory numbers:
  query ~14ms, discover ~7ms, causal trace ~3ms p50.

## License

MIT
