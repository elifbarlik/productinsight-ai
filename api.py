"""REST API:  uvicorn api:app --reload

    GET  /health
    POST /search   {query, k, mode, use_filter}   -> ranked products, no LLM
    POST /chat     {query, history[], k}          -> grounded LLM answer + products
"""
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.factory import build_rag
from src.logging_utils import get_logger
from src.rag_core import ProductRAG
from src.retriever import SearchHit

log = get_logger("api")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=20)
    mode: Optional[Literal["bm25", "dense", "hybrid"]] = None
    use_filter: bool = True


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    history: List[Turn] = Field(default_factory=list, max_length=20)
    k: int = Field(5, ge=1, le=20)


def _hit_to_json(hit: SearchHit) -> dict:
    return {**hit.product.to_dict(), "score": round(hit.score, 5), "sources": hit.sources}


def _parsed_to_json(parsed) -> dict:
    return {
        "semantic_query": parsed.semantic_query,
        "filters": parsed.describe(),
        "colors": parsed.colors,
        "categories": parsed.categories,
        "gender": parsed.gender,
        "min_price": parsed.min_price,
        "max_price": parsed.max_price,
        "parser": parsed.parser,
    }


def create_app(rag: Optional[ProductRAG] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag = rag or build_rag()
        yield

    app = FastAPI(title="ProductInsight AI", version="2.0", lifespan=lifespan)

    def get_rag(request: Request) -> ProductRAG:
        return request.app.state.rag

    @app.get("/health")
    def health(request: Request):
        r = get_rag(request)
        return {
            "status": "ok",
            "products": len(r.retriever.products),
            "default_mode": r.retriever.mode,
            "dense_available": r.retriever.dense_available,
            "llm_enabled": r.llm is not None,
        }

    @app.post("/search")
    def search(body: SearchRequest, request: Request):
        r = get_rag(request)
        try:
            result = r.retriever.search(
                body.query, k=body.k, mode=body.mode, use_filter=body.use_filter
            )
        except RuntimeError as exc:  # e.g. dense mode requested but unavailable
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "query": body.query,
            "mode": result.mode,
            "parsed": _parsed_to_json(result.parsed),
            "results": [_hit_to_json(h) for h in result.hits],
            "timings_ms": result.timings,
        }

    @app.post("/chat")
    def chat(body: ChatRequest, request: Request):
        r = get_rag(request)
        history = [t.model_dump() for t in body.history]
        resp = r.answer(body.query, history=history, k=body.k)
        return {
            "message": resp.message,
            "standalone_query": resp.standalone_query,
            "products": [_hit_to_json(h) for h in resp.hits],
            "parsed": _parsed_to_json(resp.search.parsed),
            "grounded": resp.grounded,
            "llm_used": resp.llm_used,
            "timings_ms": {**resp.search.timings, **resp.timings},
        }

    return app


app = create_app()
