__version__ = "0.3.1"

import os as _os

# Suppress noisy transformers/tqdm output (load reports, progress bars, pipeline hints)
# unless the user has explicitly configured verbosity.
if "TRANSFORMERS_VERBOSITY" not in _os.environ:
    _os.environ["TRANSFORMERS_VERBOSITY"] = "error"
if "TQDM_DISABLE" not in _os.environ:
    _os.environ["TQDM_DISABLE"] = "1"

from reasongraph._types import Node, Edge
from reasongraph._extraction import (
    NERExtractor,
    GLiNER2Extractor,
    ChatExtractor,
    OnnxTokenClassifierExtractor,
    GlinerExtractor,
)
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
    "FastEmbedEmbedder", "FastEmbedReranker",
    "TemplateSynthesizer", "PromptSynthesizer", "TransformersSynthesizer",
    "load_dataset",
]
