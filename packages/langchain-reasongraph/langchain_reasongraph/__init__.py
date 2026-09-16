"""LangChain integration for ReasonGraph.

    pip install langchain-reasongraph

    from reasongraph import ReasonGraph
    from langchain_reasongraph import ReasonGraphRetriever

    graph = ReasonGraph(); graph.initialize_sync()
    graph.add_texts_sync(["Redis runs on the same node as Elasticsearch.", ...])
    ReasonGraphRetriever(target=graph, k=4).invoke("why is checkout slow in the morning")

``target`` is a local ``ReasonGraph`` or a ``reasongraph.MemoryClient`` pointed at a
hosted service. The retriever, memory classes and tools live in the ``reasongraph``
package (``reasongraph.integrations.langchain``); this package is the ``langchain-*``
distribution LangChain's integration index expects, with the standard tests.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from reasongraph.integrations.langchain import (
    ReasonGraphChatMessageHistory,
    ReasonGraphMemory,
    ReasonGraphRetriever as _BaseRetriever,
    _connections,
    _is_client,
    _to_document,
    memory_tools,
    with_memory,
)

__all__ = ["ReasonGraphRetriever", "ReasonGraphMemory", "ReasonGraphChatMessageHistory",
           "with_memory", "memory_tools"]
__version__ = "0.1.0"


class ReasonGraphRetriever(_BaseRetriever):
    """Facts the graph connects to the question, as LangChain documents.

    ``k`` is the number of documents returned (LangChain's convention): the facts the
    graph walk connects to the question come first, with the names and cause->effect
    links they were reached through in ``metadata``; when the walk finds fewer than
    ``k``, the nearest facts by similarity fill the rest (``metadata["via"]`` empty).
    """

    k: int = 4

    @staticmethod
    def _merge(connections: list[dict], facts: list, k: int) -> list[Document]:
        docs = [_to_document(c) for c in connections]
        seen = {d.page_content for d in docs}
        for f in facts:
            if len(docs) >= k:
                break
            text = f if isinstance(f, str) else f.get("content") or f.get("text") or ""
            if text and text not in seen:
                docs.append(Document(page_content=text, metadata={"sources": [], "via": [], "causes": [], "cross_session": False}))
                seen.add(text)
        return docs[:k]

    def _fetch(self, query: str, k: int) -> list[Document]:
        connections = _connections(self.target, query, self.session, self.top_k, self.hops, k)
        facts: list = []
        if len(connections) < k:
            if _is_client(self.target):
                facts = self.target.recall(query, session=self.session, top_k=k)
            else:
                facts = self.target.query_sync(query, top_k=k, scopes={self.session} if self.session else None)
        return self._merge(connections, facts, k)

    async def _afetch(self, query: str, k: int) -> list[Document]:
        if _is_client(self.target):           # the HTTP client is sync; keep the loop free
            import asyncio
            return await asyncio.to_thread(self._fetch, query, k)
        scopes = {self.session} if self.session else None
        found = await self.target.discover(query, top_k=self.top_k, hops=self.hops, max_results=k, scopes=scopes)
        connections = list(found if isinstance(found, list) else found.get("connections", []))
        facts = await self.target.query(query, top_k=k, scopes=scopes) if len(connections) < k else []
        return self._merge(connections, facts, k)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
                                k: int | None = None, **kwargs: Any) -> list[Document]:
        return self._fetch(query, self.k if k is None else k)

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun,
                                       k: int | None = None, **kwargs: Any) -> list[Document]:
        return await self._afetch(query, self.k if k is None else k)
