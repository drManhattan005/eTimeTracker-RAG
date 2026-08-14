"""
app/services/retrieval.py
──────────────────────────
Hybrid retrieval service combining dense semantic search (Qdrant) and
lexical search (BM25Okapi) using Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import settings
from app.domain.ports import Embedder, VectorStore
from app.infrastructure.bm25 import BM25Retriever

log = logging.getLogger(__name__)

# Max possible RRF score for rank 1 in dense and rank 1 in BM25 (k=60)
_MAX_POSSIBLE_RRF = 1.0 / (60 + 1) + 1.0 / (60 + 1)  # ~0.0327868


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

    def search(
        self,
        query: str,
        limit: int = 12,
        dense_k: int = 15,
        bm25_k: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid retrieval over the complete corpus using RRF fusion.
        """
        # ── 1. Dense retrieval ───────────────────────────────────────────────
        query_vector = self.embedder.embed([query])[0]
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

        # ── 2. Lexical BM25 retrieval ─────────────────────────────────────────
        bm25_candidates: dict[str, dict[str, Any]] = {}
        if self.bm25_retriever is not None:
            raw_bm25 = self.bm25_retriever.search(query, top_k=bm25_k)
            for item in raw_bm25:
                cid = item.get("chunk_id", "")
                bm25_candidates[cid] = {
                    "bm25_rank": item.get("bm25_rank"),
                    "bm25_score": float(item.get("bm25_score", 0.0)),
                    "payload": item.get("metadata", {}),
                    "text": item.get("text", ""),
                }

        # ── 3. RRF Fusion ────────────────────────────────────────────────────
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

            fused_results.append({
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
            })

        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)

        return fused_results[:limit]
