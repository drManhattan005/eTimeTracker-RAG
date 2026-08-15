"""
app/api/routes.py
──────────────────
FastAPI router factory with lightweight in-memory session history.

Design:
- Retrieval remains current-turn focused.
- Conversation history is used only for follow-up resolution and answer grounding.
- Short/ambiguous follow-ups can be rewritten against recent turns before retrieval.
- Session state is in-memory only and resets on server restart.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Generator
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService

log = logging.getLogger(__name__)

RETRIEVAL_DENSE_K: int = 15
RETRIEVAL_BM25_K: int = 15
RETRIEVAL_LIMIT: int = 12
_RETRIEVAL_ROUTE: str = (
    f"HYBRID:dense(k={RETRIEVAL_DENSE_K})+BM25(k={RETRIEVAL_BM25_K})+RRF→top{RETRIEVAL_LIMIT}"
)

_HISTORY_MAX_TURNS = 8
_SESSION_STORE: dict[str, list[dict[str, str]]] = defaultdict(list)
_SESSION_LOCK = threading.Lock()
_LEAK_PATTERNS = [
    r"\n\s*Question\s*:",
    r"\n\s*Answer\s*:",
    r"\n\s*Q\s*:",
    r"\n\s*A\s*:",
]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(default=RETRIEVAL_LIMIT, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    effective_question: str
    answer: str
    hits: list[dict]
    model: str
    retrieval_route: str
    session_id: str


class SessionResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=200)


def _ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _clean_hit(h: dict) -> dict:
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


def _trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(history) <= _HISTORY_MAX_TURNS * 2:
        return history
    return history[-(_HISTORY_MAX_TURNS * 2) :]


def _get_session_history(session_id: str) -> list[dict[str, str]]:
    with _SESSION_LOCK:
        return list(_SESSION_STORE.get(session_id, []))


def _append_session_turn(session_id: str, role: Literal["user", "assistant"], content: str) -> None:
    with _SESSION_LOCK:
        history = _SESSION_STORE[session_id]
        history.append({"role": role, "content": content})
        _SESSION_STORE[session_id] = _trim_history(history)


def _reset_session(session_id: str) -> None:
    with _SESSION_LOCK:
        _SESSION_STORE.pop(session_id, None)


def _sanitize_model_output(text: str) -> tuple[str, bool]:
    cleaned = text or ""

    for pattern in _LEAK_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return cleaned[: match.start()].rstrip(), True

    return cleaned.rstrip(), False


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
            "version": "1.1.0",
            "model": _model,
            "retrieval_route": _RETRIEVAL_ROUTE,
        }

    @router.post("/session/reset")
    def reset_session(request: SessionResetRequest) -> dict:
        _reset_session(request.session_id)
        return {"ok": True, "session_id": request.session_id}

    @router.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        req_id = uuid.uuid4().hex[:8]
        history = _get_session_history(request.session_id)
        effective_question = generation_service.prepare_effective_query(
            question=request.question,
            history=history,
        )

        log.info(
            "[req:%s] query | model=%s route=%s session=%s question=%r effective=%r",
            req_id,
            _model,
            _RETRIEVAL_ROUTE,
            request.session_id,
            request.question[:100],
            effective_question[:140],
        )

        hits = retrieval_service.search(
            effective_question,
            limit=RETRIEVAL_LIMIT,
            dense_k=RETRIEVAL_DENSE_K,
            bm25_k=RETRIEVAL_BM25_K,
        )

        chunk_ids = [h.get("chunk_id") or h.get("id", "") for h in hits]
        log.info("[req:%s] hits  | count=%d chunks=%s", req_id, len(hits), chunk_ids)

        answer = generation_service.answer(
            question=request.question,
            retrieved_chunks=hits,
            history=history,
        )

        _append_session_turn(request.session_id, "user", request.question)
        _append_session_turn(request.session_id, "assistant", answer)

        return QueryResponse(
            question=request.question,
            effective_question=effective_question,
            answer=answer,
            hits=[_clean_hit(h) for h in hits],
            model=_model,
            retrieval_route=_RETRIEVAL_ROUTE,
            session_id=request.session_id,
        )

    @router.post("/query/stream")
    def query_stream(request: QueryRequest) -> StreamingResponse:
        req_id = uuid.uuid4().hex[:8]
        t_start = time.monotonic()
        history = _get_session_history(request.session_id)
        effective_question = generation_service.prepare_effective_query(
            question=request.question,
            history=history,
        )

        log.info(
            "[req:%s] query/stream | model=%s route=%s session=%s question=%r effective=%r",
            req_id,
            _model,
            _RETRIEVAL_ROUTE,
            request.session_id,
            request.question[:100],
            effective_question[:140],
        )

        try:
            hits = retrieval_service.search(
                effective_question,
                limit=RETRIEVAL_LIMIT,
                dense_k=RETRIEVAL_DENSE_K,
                bm25_k=RETRIEVAL_BM25_K,
            )
            chunk_ids = [h.get("chunk_id") or h.get("id", "") for h in hits]
            log.info("[req:%s] hits  | count=%d chunks=%s", req_id, len(hits), chunk_ids)
        except Exception as exc:
            log.exception("[req:%s] retrieval_error | %s", req_id, exc)

            def retrieval_error_stream() -> Generator[str, None, None]:
                yield _ndjson(
                    {
                        "type": "error",
                        "error": "retrieval_failed",
                        "message": "Could not retrieve relevant context.",
                    }
                )

            return StreamingResponse(
                retrieval_error_stream(),
                media_type="application/x-ndjson",
            )

        def event_stream() -> Generator[str, None, None]:
            try:
                result = generation_service.answer_or_abstain(
                    question=request.question,
                    retrieved_chunks=hits,
                    history=history,
                )
                response_type = str(result.get("type", "")).upper()

                yield _ndjson(
                    {
                        "type": "meta",
                        "response_type": response_type,
                        "model": _model,
                        "retrieval_route": _RETRIEVAL_ROUTE,
                        "session_id": request.session_id,
                        "effective_question": effective_question,
                        "hits": [_clean_hit(h) for h in hits],
                    }
                )

                final_answer = ""

                if response_type == "ABSTAIN":
                    final_answer = result["answer"]
                    yield _ndjson({"type": "token", "token": final_answer})
                else:
                    streamed_text = ""
                    emitted_text = ""

                    for token in generation_service.answer_stream(
                        question=request.question,
                        retrieved_chunks=hits,
                        history=history,
                    ):
                        if not token:
                            continue

                        streamed_text += token
                        cleaned_text, was_truncated = _sanitize_model_output(streamed_text)

                        if len(cleaned_text) > len(emitted_text):
                            delta = cleaned_text[len(emitted_text):]
                            if delta:
                                emitted_text = cleaned_text
                                yield _ndjson({"type": "token", "token": delta})

                        if was_truncated:
                            log.warning(
                                "[req:%s] output_sanitized | leak_pattern_detected=true",
                                req_id,
                            )
                            break

                    final_answer = emitted_text.strip() or result["answer"]

                _append_session_turn(request.session_id, "user", request.question)
                _append_session_turn(request.session_id, "assistant", final_answer)

                yield _ndjson({"type": "done", "response_type": response_type})

            except Exception as exc:
                log.exception("[req:%s] stream_error | %s", req_id, exc)
                yield _ndjson(
                    {
                        "type": "error",
                        "error": "generation_failed",
                        "message": "Could not generate a response.",
                    }
                )
            finally:
                elapsed = time.monotonic() - t_start
                log.info("[req:%s] stream_done | elapsed=%.3fs", req_id, elapsed)

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
        )

    return router
