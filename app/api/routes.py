"""
app/api/routes.py
──────────────────
FastAPI router factory with lightweight in-memory session history.

Design:
- Keeps only the last 3 prior turns plus the current question path.
- Retrieval and generation both receive the same trimmed history.
- Session state is in-memory only and resets on server restart.
"""

from __future__ import annotations

import json
import logging
import re
import threading
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
from app.services.token_budget import (
    estimate_messages_tokens,
    estimate_tokens,
    trim_messages_to_budget,
)

log = logging.getLogger(__name__)

RETRIEVAL_DENSE_K: int = 15
RETRIEVAL_BM25_K: int = 15
RETRIEVAL_LIMIT: int = 12
_RETRIEVAL_ROUTE: str = (
    f"HYBRID:dense(k={RETRIEVAL_DENSE_K})+BM25(k={RETRIEVAL_BM25_K})+RRF→top{RETRIEVAL_LIMIT}"
)

_HISTORY_MAX_TURNS = 3
_SESSION_STORE: dict[str, list[dict[str, str]]] = defaultdict(list)
_SESSION_BUDGET_STORE: dict[str, dict] = {}
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
    session_tokens_used: int
    session_tokens_total: int
    session_tokens_remaining: int
    session_blocked: bool = False


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
    max_messages = _HISTORY_MAX_TURNS * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def _get_session_history(session_id: str) -> list[dict[str, str]]:
    with _SESSION_LOCK:
        return list(_SESSION_STORE.get(session_id, []))


def _append_session_turn(session_id: str, role: Literal["user", "assistant"], content: str) -> None:
    with _SESSION_LOCK:
        history = _SESSION_STORE[session_id]
        history.append({"role": role, "content": content})
        trimmed = _trim_history(history)
        trimmed, _, _ = trim_messages_to_budget(
            trimmed,
            settings.SESSION_SOFT_TURN_BUDGET,
        )
        _SESSION_STORE[session_id] = trimmed


def _reset_session(session_id: str) -> None:
    with _SESSION_LOCK:
        _SESSION_STORE.pop(session_id, None)
        _SESSION_BUDGET_STORE.pop(session_id, None)


def _sanitize_model_output(text: str) -> tuple[str, bool]:
    cleaned = text or ""

    for pattern in _LEAK_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return cleaned[: match.start()].rstrip(), True

    return cleaned.rstrip(), False


def _history_for_model(history: list[dict[str, str]]) -> list[dict[str, str]]:
    trimmed = _trim_history(history)
    trimmed, _, _ = trim_messages_to_budget(
        trimmed,
        settings.SESSION_SOFT_TURN_BUDGET,
    )
    return trimmed


def _session_tokens_used(session_id: str) -> int:
    with _SESSION_LOCK:
        budget = _SESSION_BUDGET_STORE.get(session_id, {})
        return int(budget.get("tokens_used", 0))


def _session_budget_fields(tokens_used: int) -> dict:
    total = settings.SESSION_LIFETIME_MAX_TOKENS
    used = min(max(0, tokens_used), total)
    return {
        "session_tokens_used": used,
        "session_tokens_total": total,
        "session_tokens_remaining": max(0, total - used),
        "session_blocked": used >= total,
    }


def _is_materially_different(question: str, effective_question: str) -> bool:
    normalized_question = re.sub(r"\s+", " ", question.strip().lower())
    normalized_effective = re.sub(r"\s+", " ", effective_question.strip().lower())
    return bool(normalized_effective and normalized_effective != normalized_question)


def _request_usage_delta(
    question: str,
    effective_question: str,
    budget: dict,
    answer: str,
) -> int:
    total = estimate_tokens(question)
    if _is_materially_different(question, effective_question):
        total += estimate_tokens(effective_question)
    total += int(budget.get("final_prompt_estimate_tokens", 0) or 0)
    total += estimate_tokens(answer)
    return total


def _add_session_usage(session_id: str, delta_tokens: int) -> dict:
    with _SESSION_LOCK:
        current = int(_SESSION_BUDGET_STORE.get(session_id, {}).get("tokens_used", 0))
        next_used = current + max(0, delta_tokens)
        fields = _session_budget_fields(next_used)
        _SESSION_BUDGET_STORE[session_id] = {
            "tokens_used": fields["session_tokens_used"],
            "tokens_total": fields["session_tokens_total"],
            "tokens_remaining": fields["session_tokens_remaining"],
            "session_blocked": fields["session_blocked"],
        }
        return fields


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
            "version": "1.2.0",
            "model": _model,
            "retrieval_route": _RETRIEVAL_ROUTE,
            "history_turns": _HISTORY_MAX_TURNS,
            "session_budget": {
                "max_tokens": settings.SESSION_MAX_TOKENS,
                "max_input_tokens": settings.SESSION_MAX_INPUT_TOKENS,
                "max_output_tokens": settings.SESSION_MAX_OUTPUT_TOKENS,
                "soft_turn_budget": settings.SESSION_SOFT_TURN_BUDGET,
                "lifetime_max_tokens": settings.SESSION_LIFETIME_MAX_TOKENS,
                "enforce_by": settings.session_enforce_by,
            },
        }

    @router.post("/session/reset")
    def reset_session(request: SessionResetRequest) -> dict:
        _reset_session(request.session_id)
        return {"ok": True, "session_id": request.session_id}

    @router.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        req_id = uuid.uuid4().hex[:8]
        current_budget = _session_budget_fields(_session_tokens_used(request.session_id))

        if current_budget["session_blocked"]:
            log.info(
                "[req:%s] session blocked | session=%s used=%d total=%d",
                req_id,
                request.session_id,
                current_budget["session_tokens_used"],
                current_budget["session_tokens_total"],
            )
            return QueryResponse(
                question=request.question,
                effective_question=request.question,
                answer=settings.SESSION_LIMIT_REACHED_MESSAGE,
                hits=[],
                model=_model,
                retrieval_route=_RETRIEVAL_ROUTE,
                session_id=request.session_id,
                **current_budget,
            )

        raw_history = _get_session_history(request.session_id)
        history = _history_for_model(raw_history)
        log.info(
            "[req:%s] session budget start | session=%s raw_messages=%d "
            "kept_messages=%d session_tokens=%d",
            req_id,
            request.session_id,
            len(raw_history),
            len(history),
            estimate_messages_tokens(history),
        )

        effective_question = generation_service.prepare_effective_query(
            question=request.question,
            history=history,
        )

        hits = retrieval_service.search(
            query=effective_question,
            history=history,
            limit=request.limit,
            dense_k=RETRIEVAL_DENSE_K,
            bm25_k=RETRIEVAL_BM25_K,
        )

        result = generation_service.answer_or_abstain(
            question=request.question,
            effective_question=effective_question,
            retrieved_chunks=hits,
            history=history,
        )
        answer = result["answer"]

        cleaned_answer, leaked = _sanitize_model_output(answer)
        if leaked:
            log.warning(
                "[req:%s] stripped leaked prompt markers from /query response",
                req_id,
            )

        _append_session_turn(request.session_id, "user", request.question)
        _append_session_turn(request.session_id, "assistant", cleaned_answer)
        usage_delta = _request_usage_delta(
            request.question,
            effective_question,
            result.get("budget", {}),
            cleaned_answer,
        )
        session_budget = _add_session_usage(request.session_id, usage_delta)

        log.info(
            "[req:%s] query complete | session=%s question=%r effective=%r "
            "hits=%d final_prompt_tokens=%s usage_delta=%d session_used=%d/%d",
            req_id,
            request.session_id,
            request.question,
            effective_question,
            len(hits),
            result.get("budget", {}).get("final_prompt_estimate_tokens"),
            usage_delta,
            session_budget["session_tokens_used"],
            session_budget["session_tokens_total"],
        )

        return QueryResponse(
            question=request.question,
            effective_question=effective_question,
            answer=cleaned_answer,
            hits=[_clean_hit(h) for h in hits],
            model=_model,
            retrieval_route=_RETRIEVAL_ROUTE,
            session_id=request.session_id,
            **session_budget,
        )

    @router.post("/query/stream")
    def query_stream(request: QueryRequest) -> StreamingResponse:
        req_id = uuid.uuid4().hex[:8]
        current_budget = _session_budget_fields(_session_tokens_used(request.session_id))

        if current_budget["session_blocked"]:
            log.info(
                "[req:%s] stream session blocked | session=%s used=%d total=%d",
                req_id,
                request.session_id,
                current_budget["session_tokens_used"],
                current_budget["session_tokens_total"],
            )

            def blocked_stream() -> Generator[str, None, None]:
                yield _ndjson(
                    {
                        "type": "meta",
                        "session_id": request.session_id,
                        "question": request.question,
                        "effective_question": request.question,
                        "retrieval_route": _RETRIEVAL_ROUTE,
                        "model": _model,
                        "hits": [],
                        **current_budget,
                    }
                )
                yield _ndjson(
                    {
                        "type": "done",
                        "answer": settings.SESSION_LIMIT_REACHED_MESSAGE,
                        **current_budget,
                    }
                )

            return StreamingResponse(
                blocked_stream(),
                media_type="application/x-ndjson",
            )

        raw_history = _get_session_history(request.session_id)
        history = _history_for_model(raw_history)
        log.info(
            "[req:%s] session budget start | session=%s raw_messages=%d "
            "kept_messages=%d session_tokens=%d",
            req_id,
            request.session_id,
            len(raw_history),
            len(history),
            estimate_messages_tokens(history),
        )

        effective_question = generation_service.prepare_effective_query(
            question=request.question,
            history=history,
        )

        hits = retrieval_service.search(
            query=effective_question,
            history=history,
            limit=request.limit,
            dense_k=RETRIEVAL_DENSE_K,
            bm25_k=RETRIEVAL_BM25_K,
        )

        log.info(
            "[req:%s] stream start | session=%s question=%r effective=%r",
            req_id,
            request.session_id,
            request.question,
            effective_question,
        )

        def event_stream() -> Generator[str, None, None]:
            yield _ndjson(
                {
                    "type": "meta",
                    "session_id": request.session_id,
                    "question": request.question,
                    "effective_question": effective_question,
                    "retrieval_route": _RETRIEVAL_ROUTE,
                    "model": _model,
                    "hits": [_clean_hit(h) for h in hits],
                    **current_budget,
                }
            )

            result = generation_service.answer_or_abstain(
                question=request.question,
                effective_question=effective_question,
                retrieved_chunks=hits,
                history=history,
            )
            answer = result["answer"]

            cleaned_answer, leaked = _sanitize_model_output(answer)
            if leaked:
                log.warning(
                    "[req:%s] stripped leaked prompt markers from /query/stream response",
                    req_id,
                )

            _append_session_turn(request.session_id, "user", request.question)
            _append_session_turn(request.session_id, "assistant", cleaned_answer)
            usage_delta = _request_usage_delta(
                request.question,
                effective_question,
                result.get("budget", {}),
                cleaned_answer,
            )
            session_budget = _add_session_usage(request.session_id, usage_delta)

            log.info(
                "[req:%s] stream complete | session=%s hits=%d "
                "final_prompt_tokens=%s usage_delta=%d session_used=%d/%d",
                req_id,
                request.session_id,
                len(hits),
                result.get("budget", {}).get("final_prompt_estimate_tokens"),
                usage_delta,
                session_budget["session_tokens_used"],
                session_budget["session_tokens_total"],
            )

            yield _ndjson({"type": "done", "answer": cleaned_answer, **session_budget})

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
        )

    return router
