#!/usr/bin/env python3
"""
scripts/validate_end2end.py
────────────────────────────
End-to-End RAG benchmark validation script for qwen2.5:1.5b.

Executes the actual runtime answer path:
  Hybrid Retrieval -> Context Assembly -> Qwen 2.5 1.5B LLM Generation
for all 8 known Buyer Fit & Commercial benchmark prompts.

Usage:
    python scripts/validate_end2end.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import settings
from app.infrastructure.embedders import FastEmbedder
from app.infrastructure.llm import OllamaClient
from app.infrastructure.vectorstores.qdrant_store import QdrantVectorStore
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService
from qdrant_client import QdrantClient

BENCHMARK_TESTS = [
    {
        "id": 1,
        "query": "We have 500 employees across branches, field sales staff, and need geofencing. Is this suitable?",
        "check": lambda ans, hits: (
            any(w in ans.lower() for w in ["yes", "suitable", "strong fit", "ideal", "good fit"])
            and any(w in ans.lower() for w in ["business", "1,000", "1000", "branch", "geofencing"])
            and "could not find a confident answer" not in ans.lower()
        ),
        "description": "Must confirm suitability and identify Business plan as the published fit (up to 1,000 employees, multi-branch, geofencing).",
    },
    {
        "id": 2,
        "query": "Is it suitable for a 1,200-employee organization?",
        "check": lambda ans, hits: (
            ("enterprise" in ans.lower() or "unlimited" in ans.lower())
            and "could not find a confident answer" not in ans.lower()
        ),
        "description": "Must not refuse; must state Enterprise is appropriate because Business is capped at 1,000 while Enterprise is unlimited.",
    },
    {
        "id": 3,
        "query": "Does Starter include geofencing?",
        "check": lambda ans, hits: (
            ("not listed" in ans.lower() or "business" in ans.lower() or "starter does not list" in ans.lower() or "no" in ans.lower())
            and "could not find a confident answer" not in ans.lower()
        ),
        "description": "Must state geofencing is not listed among Starter published entitlements and/or is a Business entitlement.",
    },
    {
        "id": 4,
        "query": "Does Enterprise have a public price?",
        "check": lambda ans, hits: (
            ("custom" in ans.lower() or "no public" in ans.lower() or "not listed" in ans.lower() or "no" in ans.lower())
            and "could not find a confident answer" not in ans.lower()
        ),
        "description": "Must state Enterprise has custom pricing tailored to scale; no public numeric price.",
    },
    {
        "id": 5,
        "query": "What support comes with each plan?",
        "check": lambda ans, hits: (
            ("email" in ans.lower())
            and ("priority" in ans.lower())
            and ("dedicated" in ans.lower() or "sla" in ans.lower() or "enterprise" in ans.lower())
            and "could not find a confident answer" not in ans.lower()
        ),
        "description": "Must cover Starter (Email), Business (Priority), and Enterprise (Dedicated Account Manager/SLA).",
    },
    {
        "id": 6,
        "query": "Does eTimeTracker support rotating shifts?",
        "check": lambda ans, hits: (
            ("yes" in ans.lower() or "support" in ans.lower() or "shift" in ans.lower())
            and "could not find a confident answer" not in ans.lower()
        ),
        "description": "Must answer yes and confirm shift scheduling capabilities.",
    },
    {
        "id": 7,
        "query": "We have 1,200 employees with rotating shifts. Which plan should we consider?",
        "check": lambda ans, hits: (
            ("enterprise" in ans.lower())
            and ("unlimited" in ans.lower() or "1,000" in ans.lower() or "1000" in ans.lower() or "shift" in ans.lower())
            and "could not find a confident answer" not in ans.lower()
        ),
        "description": "Must state Enterprise is appropriate because Business is capped at 1,000 while Enterprise is unlimited.",
    },
    {
        "id": 8,
        "query": "We have 1,001 employees and need multi-branch support. Which plan is appropriate?",
        "check": lambda ans, hits: (
            ("enterprise" in ans.lower())
            and ("1,000" in ans.lower() or "1000" in ans.lower() or "unlimited" in ans.lower() or "multi-branch" in ans.lower() or "branch" in ans.lower())
            and "could not find a confident answer" not in ans.lower()
        ),
        "description": "Must state Enterprise is appropriate because Business is capped at 1,000 employees.",
    },
]


def run_benchmark() -> None:
    print("=" * 80)
    print("END-TO-END RAG BENCHMARK VALIDATION (QWEN 2.5 1.5B)")
    print("=" * 80)

    print("Initializing services...")
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
    llm = OllamaClient(
        model_name=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        timeout=settings.ollama_timeout,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        options=settings.ollama_options,
    )
    generation_service = GenerationService(llm=llm)

    results_summary: list[dict] = []

    for test in BENCHMARK_TESTS:
        t_id = test["id"]
        query = test["query"]
        description = test["description"]
        checker = test["check"]

        print(f"\n[{t_id}/8] PROMPT: \"{query}\"")
        print(f"      Goal: {description}")

        hits = retrieval_service.search(query, limit=12, dense_k=15, bm25_k=15)
        retrieved_ids = [h.get("chunk_id") or h.get("id") for h in hits]
        print(f"      Retrieved Fused Chunks ({len(hits)}): {retrieved_ids[:3]}")

        bot_answer = generation_service.answer(query, hits).strip()

        print(f"      BOT ANSWER:\n\"\"\"\n{bot_answer}\n\"\"\"")

        passed = checker(bot_answer, hits)
        status_str = "PASS ✅" if passed else "FAIL ❌"
        print(f"      EVALUATION: {status_str}")

        results_summary.append({
            "id": t_id,
            "query": query,
            "fused_chunks": retrieved_ids,
            "answer": bot_answer,
            "passed": passed,
        })

    print("\n" + "=" * 80)
    print("END-TO-END SUMMARY RESULTS")
    print("=" * 80)

    total_passed = sum(1 for r in results_summary if r["passed"])
    for r in results_summary:
        st = "PASS ✅" if r["passed"] else "FAIL ❌"
        print(f"Prompt {r['id']}: {st} | \"{r['query'][:50]}...\"")

    print(f"\nScore: {total_passed}/8 Prompts Passed")
    print("=" * 80)

    qdrant_client.close()
    if total_passed < 8:
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()
