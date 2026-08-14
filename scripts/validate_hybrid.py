#!/usr/bin/env python3
"""
scripts/validate_hybrid.py
──────────────────────────
Validation script for hybrid retrieval (Pass 4).

Tests the 6 benchmark queries against Qdrant + BM25 RRF fusion, checking
that expected evidence ranks in the top retrieved results.

Usage:
    python scripts/validate_hybrid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import settings
from app.infrastructure.embedders import FastEmbedder
from app.infrastructure.vectorstores.qdrant_store import QdrantVectorStore
from app.services.retrieval import RetrievalService
from qdrant_client import QdrantClient

VALIDATION_QUERIES = [
    {
        "id": 1,
        "query": "Does Starter include geofencing?",
        "expected_keywords": ["starter", "business", "commercial-comparison"],
        "description": "Must retrieve Starter plan chunk and Business/comparison chunk showing geofencing is Business-only.",
    },
    {
        "id": 2,
        "query": "We have 500 employees across branches, field sales staff, and need geofencing. Is this suitable?",
        "expected_keywords": ["field", "business", "commercial-comparison", "branch"],
        "description": "Must retrieve Field/distributed Buyer Fit chunk and Business plan chunk.",
    },
    {
        "id": 3,
        "query": "Is it suitable for a 1,200-employee organization?",
        "expected_keywords": ["enterprise", "commercial-comparison"],
        "description": "Must retrieve Enterprise plan chunk (unlimited capacity) and/or comparison chunk.",
    },
    {
        "id": 4,
        "query": "Does Enterprise have a public price?",
        "expected_keywords": ["enterprise", "commercial-comparison"],
        "description": "Must retrieve Enterprise plan chunk (custom pricing) and/or comparison chunk.",
    },
    {
        "id": 5,
        "query": "What support comes with each plan?",
        "expected_keywords": ["commercial-comparison", "starter", "business", "enterprise", "support"],
        "description": "Must retrieve atomic comparison chunk or individual plan support chunks.",
    },
    {
        "id": 6,
        "query": "Does eTimeTracker support rotating shifts?",
        "expected_keywords": ["shift", "scheduling"],
        "description": "Must retrieve shift-scheduling chunk from Product/Buyer Fit.",
    },
]


def run_validation() -> None:
    print("Initializing Hybrid Retrieval Service...")
    embedder = FastEmbedder(model_name=settings.EMBED_MODEL)
    qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION,
        vector_size=settings.QDRANT_VECTOR_SIZE,
    )
    retrieval_service = RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
        chunks_file=settings.CHUNKS_FILE,
    )

    all_passed = True

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL VALIDATION BENCHMARK (6 QUERIES)")
    print("=" * 80)

    for item in VALIDATION_QUERIES:
        q_id = item["id"]
        query = item["query"]
        expected = item["expected_keywords"]

        hits = retrieval_service.search(query=query, limit=5, dense_k=15, bm25_k=15)

        print(f"\nQUERY {q_id}: \"{query}\"")
        print(f"Goal: {item['description']}")
        print("-" * 80)

        found_expected = False
        for rank, hit in enumerate(hits, start=1):
            cid = hit["chunk_id"]
            meta = hit.get("payload", {}) or hit.get("metadata", {})
            sphere = meta.get("intent_sphere", "")
            ctype = meta.get("chunk_type", "")
            tier = meta.get("plan_tier", "none")
            d_rank = hit.get("dense_rank")
            d_score = hit.get("dense_score")
            b_rank = hit.get("bm25_rank")
            b_score = hit.get("bm25_score")
            fused = hit.get("fused_score")
            text_preview = hit.get("text", "")[:120].replace("\n", " ")

            print(
                f"  [{rank}] {cid}\n"
                f"      sphere={sphere} | type={ctype} | tier={tier}\n"
                f"      dense_rank={d_rank} (score={d_score}) | bm25_rank={b_rank} (score={b_score}) | fused_score={fused}\n"
                f"      preview: \"{text_preview}...\""
            )

            # Check if expected keyword matches chunk_id, metadata or text
            chunk_blob = f"{cid} {ctype} {tier} {sphere} {text_preview}".lower()
            if any(k.lower() in chunk_blob for k in expected):
                found_expected = True

        if found_expected:
            print("  Status: ✅ EXPECTED EVIDENCE FOUND")
        else:
            print("  Status: ❌ EXPECTED EVIDENCE MISSING")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("RESULT: ALL 6 VALIDATION QUERIES PASSED SUCCESSFUL RETRIEVAL!")
    else:
        print("RESULT: SOME VALIDATION QUERIES DID NOT MATCH EXPECTED EVIDENCE.")
    print("=" * 80)

    qdrant_client.close()
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
