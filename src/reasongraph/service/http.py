"""FastAPI HTTP transport for the memory service.

    from reasongraph.service import MemoryService
    from reasongraph.service.http import create_app
    from reasongraph.backends import PostgresBackend

    service = MemoryService(backend=PostgresBackend("postgresql:///memory"), synthesizer=my_llm)
    app = create_app(service)          # uvicorn reasongraph.service.http:app

Requires ``pip install reasongraph[service]``.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from reasongraph.service.core import MemoryService

logger = logging.getLogger(__name__)


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
    recency_weight: float = 0.0
    isolate: bool | None = None
    detailed: bool = False
    synthesize: bool = False


class DiscoverIn(BaseModel):
    query: str
    session: str | None = None
    hops: int = 4
    top_k: int = 5
    max_results: int = 10
    search_mode: str = "embedding"
    isolate: bool | None = None
    synthesize: bool = False


class SupersedeIn(BaseModel):
    session: str
    old_text: str
    new_text: str
    purge_orphans: bool = False


class DeleteIn(BaseModel):
    text: str
    purge_orphans: bool = False


def _api_key_dependency(api_key: str | None):
    """Bearer / X-API-Key check. A no-op when ``api_key`` is None (dev default)."""
    async def check(
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(None),
    ) -> None:
        if api_key is None:
            return
        provided = x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            provided = authorization[len("bearer "):]
        if provided != api_key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    return check


def create_app(service: MemoryService, api_key: str | None = None) -> FastAPI:
    """Build a FastAPI app over a MemoryService (opens/closes it with the app).

    When ``api_key`` is set, every data endpoint requires it via an
    ``Authorization: Bearer <key>`` or ``X-API-Key`` header; ``/health`` and
    ``/ready`` stay open for load balancers. When None (default), auth is off.

    If the underlying graph has ``forget_every`` set, a background task runs the
    throttled forget sweep on that interval (otherwise the setting would never
    fire).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.initialize()
        app.state.ready = True
        app.state.forget_task = None
        interval = getattr(service.graph, "forget_every", None)
        if interval:
            async def _sweeper() -> None:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await service.maybe_forget()
                    except Exception:  # keep the loop alive across transient errors
                        logger.exception("scheduled forget sweep failed")

            app.state.forget_task = asyncio.create_task(_sweeper())
        yield
        task = app.state.forget_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await service.close()

    app = FastAPI(title="reasongraph memory service", lifespan=lifespan)
    app.state.ready = False
    router = APIRouter(dependencies=[Depends(_api_key_dependency(api_key))])

    async def _synth_or_400(coro):
        try:
            return await coro
        except RuntimeError as e:  # no synthesizer configured
            raise HTTPException(status_code=400, detail=str(e))

    # -- liveness / readiness (unauthenticated, for orchestration probes) --

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="starting up")
        return {"ready": True}

    # -- data endpoints (require the API key when one is configured) --

    @router.post("/sessions/{session}/memory")
    async def push(session: str, body: PushIn):
        return await service.push(session, body.text)

    @router.post("/sessions/{session}/memory/batch")
    async def push_many(session: str, body: PushManyIn):
        return await service.push_many(session, body.texts)

    @router.post("/query")
    async def query(body: QueryIn):
        facts = await service.query(
            body.query, session=body.session, hops=body.hops,
            top_k=body.top_k, search_mode=body.search_mode,
            recency_weight=body.recency_weight, isolate=body.isolate,
            detailed=body.detailed,
        )
        resp: dict = {"facts": facts}
        if body.synthesize:
            resp["answer"] = await _synth_or_400(service.answer(
                body.query, session=body.session, use_discover=False,
                hops=body.hops, top_k=body.top_k,
            ))
        return resp

    @router.post("/discover")
    async def discover(body: DiscoverIn):
        connections = await service.discover(
            body.query, session=body.session, hops=body.hops,
            top_k=body.top_k, max_results=body.max_results,
            search_mode=body.search_mode, isolate=body.isolate,
        )
        resp: dict = {"connections": connections}
        if body.synthesize:
            resp["answer"] = await _synth_or_400(service.answer(
                body.query, session=body.session, use_discover=True,
                hops=body.hops, top_k=body.top_k,
            ))
        return resp

    @router.post("/supersede")
    async def supersede(body: SupersedeIn):
        return await service.supersede(
            body.session, body.old_text, body.new_text,
            purge_orphans=body.purge_orphans,
        )

    @router.post("/delete")
    async def delete(body: DeleteIn):
        return await service.delete(body.text, purge_orphans=body.purge_orphans)

    @router.post("/forget")
    async def forget():
        return await service.forget()

    @router.get("/sessions")
    async def sessions():
        return {"sessions": await service.list_sessions()}

    @router.get("/stats")
    async def stats():
        return await service.stats()

    app.include_router(router)
    return app
