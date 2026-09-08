from reasongraph._split import SentenceSplitter, SaTSplitter, RegexSplitter
__version__ = "0.6.6"

import os as _os

# Suppress noisy transformers/tqdm output (load reports, progress bars, pipeline hints)
# unless the user has explicitly configured verbosity.
if "TRANSFORMERS_VERBOSITY" not in _os.environ:
    _os.environ["TRANSFORMERS_VERBOSITY"] = "error"
if "TQDM_DISABLE" not in _os.environ:
    _os.environ["TQDM_DISABLE"] = "1"

from reasongraph._types import Node, Edge
from reasongraph._canonical import AliasCanonicalizer
from reasongraph._extraction import (
    NERExtractor,
    GLiNER2Extractor,
    ChatExtractor,
    OnnxTokenClassifierExtractor,
    GlinerExtractor,
    GlinerRelexExtractor,
    HybridCausalExtractor,
    CausalPointerExtractor,
)
from reasongraph.loop import MemoryLoop, ContextBlock
from reasongraph._conflict import ConflictResolver, NLIConflictResolver, LLMConflictResolver, FineTunedConflictResolver
from reasongraph._fastembed import FastEmbedEmbedder, FastEmbedReranker
from reasongraph._synthesizers import (
    TemplateSynthesizer,
    PromptSynthesizer,
    TransformersSynthesizer,
)
from reasongraph.graph import ReasonGraph
from reasongraph.backends._memory import MemoryBackend
from reasongraph.datasets import load_dataset

__all__ = [
    "ReasonGraph", "Node", "Edge",
    "MemoryBackend",
    "NERExtractor", "GLiNER2Extractor", "ChatExtractor",
    "OnnxTokenClassifierExtractor", "GlinerExtractor",
    "GlinerRelexExtractor", "HybridCausalExtractor", "CausalPointerExtractor",
    "ConflictResolver", "NLIConflictResolver", "LLMConflictResolver", "FineTunedConflictResolver", "MemoryLoop", "ContextBlock", "SentenceSplitter", "SaTSplitter", "RegexSplitter",
    "AliasCanonicalizer",
    "FastEmbedEmbedder", "FastEmbedReranker",
    "TemplateSynthesizer", "PromptSynthesizer", "TransformersSynthesizer",
    "load_dataset",
]
