"""
app/services/retrieval.py
──────────────────────────
Hybrid retrieval service combining dense semantic search (Qdrant) and
lexical search (BM25Okapi) using Reciprocal Rank Fusion (RRF).

Adds lightweight multi-turn query shaping for a small-model bot:
- only the last 3 prior turns are considered
- current question remains primary
- commercial/plan/pricing queries get an authoritative boost query
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from app.config import settings
from app.domain.ports import Embedder, VectorStore
from app.infrastructure.bm25 import BM25Retriever

log = logging.getLogger(__name__)

_MAX_POSSIBLE_RRF = 1.0 / (60 + 1) + 1.0 / (60 + 1)

_COMMERCIAL_KEYWORDS = {
    "plan",
    "plans",
    "pricing",
    "price",
    "prices",
    "commercial",
    "support",
    "enterprise",
    "business",
    "starter",
    "which plan",
    "what plan",
    "employee",
    "employees",
    "per employee",
    "dedicated account manager",
    "sla",
    "custom integrations",
    "multi-tenant admin",
}

_PRONOUN_FOLLOWUP_TOKENS = {
    "it",
    "that",
    "this",
    "they",
    "them",
    "those",
    "these",
    "he",
    "she",
    "its",
    "their",
    "there",
    "same",
    "one",
    "ones",
}

_SHORT_FOLLOWUP_PREFIXES = (
    "what about",
    "how about",
    "and ",
    "what if",
    "for that",
    "for this",
    "does that",
    "does it",
    "can it",
    "is that",
    "is it",
    "which one",
    "which plan",
)


class RetrievalService:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunks_file: Path | str | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.rrf_k = rrf_k

        if chunks_file is None:
            chunks_file = Path(settings.CHUNKS_FILE)
        else:
            chunks_file = Path(chunks_file)

        if not chunks_file.exists():
            log.warning(
                "[retrieval] chunks file not found at %s. BM25 will fail if queried.",
                chunks_file,
            )
            self.bm25_retriever: BM25Retriever | None = None
        else:
            log.info("[retrieval] initializing BM25 retriever from %s", chunks_file)
            self.bm25_retriever = BM25Retriever.from_jsonl(chunks_file)

    def build_search_query(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        query = (query or "").strip()
        history = history or []

        if not query:
            return ""

        recent_turns = self._recent_turns(history, limit=3)
        recent_user_text = " ".join(
            turn["content"] for turn in recent_turns if turn.get("role") == "user"
        ).strip()

        if self._looks_commercial(query):
            base = query
            if recent_user_text and self._is_followup(query):
                base = f"{recent_user_text} {query}".strip()

            authoritative = (
                "PLAN RULES AUTHORITATIVE "
                "Starter 99 per employee per month up to 100 employees "
                "Business 249 per employee per month up to 1000 employees "
                "Enterprise custom pricing unlimited employees "
                "more than 1000 employees Enterprise "
                "101 to 1000 employees Business "
                "100 employees or fewer Starter if listed capabilities fit "
                "Dedicated Account Manager SLA Guarantee Custom Integrations Multi-tenant Admin Enterprise only"
            )
            return f"{base}\n{authoritative}".strip()

        if recent_user_text and self._is_followup(query):
            return f"{recent_user_text} {query}".strip()

        return query

    def search(
        self,
        query: str,
        limit: int = 12,
        dense_k: int = 15,
        bm25_k: int = 15,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid retrieval over the complete corpus using RRF fusion.
        """
        effective_query = self.build_search_query(query=query, history=history)
        query_vector = self.embedder.embed([effective_query])[0]
        raw_dense = self.vector_store.search(query_vector=query_vector, limit=dense_k)

        dense_candidates: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(raw_dense, start=1):
            cid = item.get("id") or item.get("payload", {}).get("chunk_id", "")
            dense_candidates[cid] = {
                "dense_rank": rank,
                "dense_score": float(item.get("score", 0.0)),
                "payload": item.get("payload", {}),
                "text": item.get("payload", {}).get("text", ""),
            }

        bm25_candidates: dict[str, dict[str, Any]] = {}
        if self.bm25_retriever is not None:
            raw_bm25 = self.bm25_retriever.search(effective_query, top_k=bm25_k)
            for item in raw_bm25:
                cid = item.get("chunk_id", "")
                bm25_candidates[cid] = {
                    "bm25_rank": item.get("bm25_rank"),
                    "bm25_score": float(item.get("bm25_score", 0.0)),
                    "payload": item.get("metadata", {}),
                    "text": item.get("text", ""),
                }

        all_cids = set(dense_candidates.keys()) | set(bm25_candidates.keys())
        fused_results: list[dict[str, Any]] = []

        for cid in all_cids:
            d_info = dense_candidates.get(cid)
            b_info = bm25_candidates.get(cid)

            d_rank = d_info["dense_rank"] if d_info else None
            d_score = d_info["dense_score"] if d_info else 0.0

            b_rank = b_info["bm25_rank"] if b_info else None
            b_score = b_info["bm25_score"] if b_info else 0.0

            rrf_score = 0.0
            if d_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + d_rank)
            if b_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + b_rank)

            payload = (d_info or b_info or {}).get("payload", {})
            text = (d_info or b_info or {}).get("text", "")

            norm_score = min(1.0, rrf_score / _MAX_POSSIBLE_RRF)

            fused_results.append(
                {
                    "id": cid,
                    "chunk_id": cid,
                    "text": text,
                    "score": round(norm_score, 4),
                    "fused_score": round(rrf_score, 6),
                    "dense_rank": d_rank,
                    "dense_score": round(d_score, 4),
                    "bm25_rank": b_rank,
                    "bm25_score": round(b_score, 4),
                    "metadata": payload,
                    "payload": payload,
                }
            )

        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused_results[:limit]

    @staticmethod
    def _recent_turns(history: list[dict[str, str]], limit: int = 3) -> list[dict[str, str]]:
        if not history:
            return []
        return history[-limit:]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _looks_commercial(self, query: str) -> bool:
        q = self._normalize(query)
        return any(keyword in q for keyword in _COMMERCIAL_KEYWORDS)

    def _is_followup(self, query: str) -> bool:
        q = self._normalize(query)
        if len(q.split()) <= 8:
            if any(token in q.split() for token in _PRONOUN_FOLLOWUP_TOKENS):
                return True
            if any(q.startswith(prefix) for prefix in _SHORT_FOLLOWUP_PREFIXES):
                return True
            if "?" in q and len(q.split()) <= 6:
                return True
        return False
