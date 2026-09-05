"""Local stdio MCP memory server for Claude Code / Claude Desktop / Cursor.

    claude mcp add memory -- python local_mcp_server.py

Persists to ~/.reasongraph/memory.sqlite (override with REASONGRAPH_DATABASE_URL).
Deps: pip install "reasongraph[service,gliner,fastembed,sqlite]"
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from reasongraph import FastEmbedEmbedder, ReasonGraph, TemplateSynthesizer
from reasongraph.backends import SqliteBackend
from reasongraph.service import MemoryService
from reasongraph.service.mcp_server import create_mcp


def main() -> None:
    db = os.environ.get("REASONGRAPH_DATABASE_URL")
    if not db:
        path = Path.home() / ".reasongraph"
        path.mkdir(exist_ok=True)
        db = str(path / "memory.sqlite")
    graph = ReasonGraph(
        backend=SqliteBackend(db),
        embed_model=FastEmbedEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        synthesizer=TemplateSynthesizer(),
    )
    service = MemoryService(graph=graph, defer_extraction=True)
    asyncio.run(service.initialize())
    create_mcp(service).run()  # stdio


if __name__ == "__main__":
    main()
