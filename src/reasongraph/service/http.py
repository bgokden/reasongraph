"""FastAPI HTTP transport for the memory service.

    from reasongraph.service import MemoryService
    from reasongraph.service.http import create_app
    from reasongraph.backends import PostgresBackend

    service = MemoryService(backend=PostgresBackend("postgresql:///memory"), synthesizer=my_llm)
    app = create_app(service)          # uvicorn reasongraph.service.http:app

Requires ``pip install reasongraph[service]``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from reasongraph.service.core import MemoryService


class PushIn(BaseModel):
    text: str


class PushManyIn(BaseModel):
    texts: list[str]


class QueryIn(BaseModel):
    query: str
    session: str | None = None
    hops: int = 4
    top_k: int = 5
    search_mode: str = "embedding"
    synthesize: bool = False


class DiscoverIn(BaseModel):
    query: str
    session: str | None = None
    hops: int = 4
    top_k: int = 5
    max_results: int = 10
    synthesize: bool = False


class SupersedeIn(BaseModel):
    session: str
    old_text: str
    new_text: str


def create_app(service: MemoryService) -> FastAPI:
    """Build a FastAPI app over a MemoryService (opens/closes it with the app)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.initialize()
        yield
        await service.close()

    app = FastAPI(title="reasongraph memory service", lifespan=lifespan)

    async def _synth_or_400(coro):
        try:
            return await coro
        except RuntimeError as e:  # no synthesizer configured
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/sessions/{session}/memory")
    async def push(session: str, body: PushIn):
        return await service.push(session, body.text)

    @app.post("/sessions/{session}/memory/batch")
    async def push_many(session: str, body: PushManyIn):
        return await service.push_many(session, body.texts)

    @app.post("/query")
    async def query(body: QueryIn):
        facts = await service.query(
            body.query, session=body.session, hops=body.hops,
            top_k=body.top_k, search_mode=body.search_mode,
        )
        resp: dict = {"facts": facts}
        if body.synthesize:
            resp["answer"] = await _synth_or_400(service.answer(
                body.query, session=body.session, use_discover=False,
                hops=body.hops, top_k=body.top_k,
            ))
        return resp

    @app.post("/discover")
    async def discover(body: DiscoverIn):
        connections = await service.discover(
            body.query, session=body.session, hops=body.hops,
            top_k=body.top_k, max_results=body.max_results,
        )
        resp: dict = {"connections": connections}
        if body.synthesize:
            resp["answer"] = await _synth_or_400(service.answer(
                body.query, session=body.session, use_discover=True,
                hops=body.hops, top_k=body.top_k,
            ))
        return resp

    @app.post("/supersede")
    async def supersede(body: SupersedeIn):
        return await service.supersede(body.session, body.old_text, body.new_text)

    @app.post("/forget")
    async def forget():
        return await service.forget()

    @app.get("/sessions")
    async def sessions():
        return {"sessions": await service.list_sessions()}

    @app.get("/stats")
    async def stats():
        return await service.stats()

    return app
