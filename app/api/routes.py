"""
app/api/routes.py
──────────────────
FastAPI router factory.

Production retrieval constants are defined here and imported by app.py
for the startup banner so both always agree on the same values.

Per-request logging emits:
  [req:<id>] query | model=qwen2.5:1.5b route=HYBRID question=...
  [req:<id>] hits  | chunks=[id1, id2, ...]
  [req:<id>] done  | answer_preview=...
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Generator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService

log = logging.getLogger(__name__)

# ── Production retrieval constants ────────────────────────────────────────────
# These values are used by every web request and are exported for the startup
# banner in app.py.  Only change them here; app.py will pick up the update.
RETRIEVAL_DENSE_K: int = 15   # Qdrant vector candidates
RETRIEVAL_BM25_K: int = 15    # BM25 lexical candidates
RETRIEVAL_LIMIT: int = 12     # Final fused hits passed to the generation layer
_RETRIEVAL_ROUTE: str = f"HYBRID:dense(k={RETRIEVAL_DENSE_K})+BM25(k={RETRIEVAL_BM25_K})+RRF→top{RETRIEVAL_LIMIT}"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    # limit is ignored by the production hybrid path; kept for API compatibility.
    # The server always uses RETRIEVAL_LIMIT (12) for production quality.
    limit: int = Field(default=RETRIEVAL_LIMIT, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    answer: str
    hits: list[dict]
    # Diagnostic fields visible in non-streaming responses
    model: str
    retrieval_route: str


def _ndjson(obj: dict) -> str:
    """Serialise a dict to a newline-terminated JSON string."""
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _clean_hit(h: dict) -> dict:
    """Return a client-safe hit dict, dropping raw vectors."""
    payload = h.get("payload") or h.get("metadata") or {}
    text = h.get("text") or payload.get("text", "")
    return {
        "id": h.get("id") or h.get("chunk_id", ""),
        "chunk_id": h.get("chunk_id") or h.get("id", ""),
        "score": round(h.get("score", 0.0), 4),
        "fused_score": round(h.get("fused_score", 0.0), 6),
        "dense_rank": h.get("dense_rank"),
        "bm25_rank": h.get("bm25_rank"),
        "section": payload.get("heading_path_text") or payload.get("section", {}),
        "intent_sphere": payload.get("intent_sphere", ""),
        "chunk_type": payload.get("chunk_type", ""),
        "plan_tier": payload.get("plan_tier", ""),
        "text_preview": text[:200],
    }


def make_router(
    retrieval_service: RetrievalService,
    generation_service: GenerationService,
) -> APIRouter:
    router = APIRouter()
    _model = settings.OLLAMA_MODEL

    @router.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": "1.0.0",
            "model": _model,
            "retrieval_route": _RETRIEVAL_ROUTE,
        }

    @router.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        req_id = uuid.uuid4().hex[:8]
        log.info(
            "[req:%s] query | model=%s route=%s question=%r",
            req_id, _model, _RETRIEVAL_ROUTE, request.question[:100],
        )

        # Always use production limits — ignore client-supplied limit
        hits = retrieval_service.search(
            request.question,
            limit=RETRIEVAL_LIMIT,
            dense_k=RETRIEVAL_DENSE_K,
            bm25_k=RETRIEVAL_BM25_K,
        )

        chunk_ids = [h.get("chunk_id") or h.get("id", "") for h in hits]
        log.info("[req:%s] hits  | count=%d chunks=%s", req_id, len(hits), chunk_ids)

        answer = generation_service.answer(request.question, hits)
        log.info("[req:%s] done  | answer_preview=%r", req_id, answer[:120])

        return QueryResponse(
            question=request.question,
            answer=answer,
            hits=[_clean_hit(h) for h in hits],
            model=_model,
            retrieval_route=_RETRIEVAL_ROUTE,
        )

    @router.post("/query/stream")
    def query_stream(request: QueryRequest) -> StreamingResponse:
        req_id = uuid.uuid4().hex[:8]
        t_start = time.monotonic()

        log.info(
            "[req:%s] query/stream | model=%s route=%s question=%r",
            req_id, _model, _RETRIEVAL_ROUTE, request.question[:100],
        )

        try:
            # Always use production limits — ignore client-supplied limit
            hits = retrieval_service.search(
                request.question,
                limit=RETRIEVAL_LIMIT,
                dense_k=RETRIEVAL_DENSE_K,
                bm25_k=RETRIEVAL_BM25_K,
            )
            chunk_ids = [h.get("chunk_id") or h.get("id", "") for h in hits]
            log.info(
                "[req:%s] hits  | count=%d chunks=%s",
                req_id, len(hits), chunk_ids,
            )
        except Exception as exc:
            log.error("[req:%s] retrieval failed: %s", req_id, exc)
            hits = []

            def retrieval_error_stream() -> Generator[str, None, None]:
                yield _ndjson({"type": "meta", "hits": [], "model": _model, "retrieval_route": _RETRIEVAL_ROUTE})
                yield _ndjson({"type": "error", "message": "Retrieval failed. Please try again."})
                yield _ndjson({"type": "done", "done_reason": "retrieval_error", "stats": {}})

            return StreamingResponse(
                retrieval_error_stream(),
                media_type="application/x-ndjson",
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
            )

        clean_hits = [_clean_hit(h) for h in hits]
        full_answer_parts: list[str] = []

        def event_generator() -> Generator[str, None, None]:
            done_reason = "unknown"
            stats: dict = {}
            errored = False

            try:
                log.info("[req:%s] stream start", req_id)

                for chunk in generation_service.answer_stream(request.question, hits):
                    full_answer_parts.append(chunk)
                    yield _ndjson({"type": "token", "content": chunk})

                llm = generation_service.llm
                if hasattr(llm, "last_stream_stats") and llm.last_stream_stats:
                    s = llm.last_stream_stats
                    done_reason = s.done_reason or "stop"
                    stats = {
                        "prompt_tokens": s.prompt_eval_count,
                        "completion_tokens": s.eval_count,
                        "eval_ms": s.eval_duration_ms,
                        "total_ms": s.total_duration_ms,
                    }
                else:
                    done_reason = "stop"

                elapsed = round((time.monotonic() - t_start) * 1000, 1)
                full_answer = "".join(full_answer_parts)
                log.info(
                    "[req:%s] done  | model=%s done_reason=%s "
                    "completion_tokens=%s elapsed_ms=%s answer_preview=%r",
                    req_id,
                    _model,
                    done_reason,
                    stats.get("completion_tokens", "?"),
                    elapsed,
                    full_answer[:120],
                )

            except Exception as exc:
                errored = True
                done_reason = "error"
                log.error("[req:%s] stream error: %s", req_id, exc, exc_info=True)
                yield _ndjson({"type": "error", "message": str(exc)})

            finally:
                yield _ndjson({
                    "type": "meta",
                    "hits": clean_hits,
                    "model": _model,
                    "retrieval_route": _RETRIEVAL_ROUTE,
                })
                yield _ndjson({
                    "type": "done",
                    "done_reason": done_reason,
                    "stats": stats,
                })
                log.info("[req:%s] done event emitted | errored=%s", req_id, errored)

        return StreamingResponse(
            event_generator(),
            media_type="application/x-ndjson",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "X-Request-Id": req_id,
            },
        )

    return router
