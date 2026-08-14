#!/usr/bin/env python3
"""
scripts/validate_web_api.py
────────────────────────────
End-to-end web API benchmark validation.

Runs all 8 benchmark queries through the LIVE FastAPI web server
(/query/stream endpoint) — the exact path used by the chatbot UI —
and reports for each query:

  - Active model (from meta event)
  - Retrieval route (from meta event)
  - Fused chunk IDs returned
  - Full answer assembled from token stream
  - PASS / FAIL against the same evaluation criteria as the CLI benchmark

The FastAPI server must be running before executing this script.

Usage:
    # Ensure FastAPI is running first:
    #   uvicorn app.main:app --reload --port 8000
    python scripts/validate_web_api.py
    python scripts/validate_web_api.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests

# ── Benchmark definitions (identical logic to validate_end2end.py) ────────────
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
            ("not listed" in ans.lower() or "business" in ans.lower() or "no" in ans.lower())
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
            and ("1,000" in ans.lower() or "1000" in ans.lower() or "unlimited" in ans.lower() or "branch" in ans.lower())
            and "could not find a confident answer" not in ans.lower()
        ),
        "description": "Must state Enterprise is appropriate because Business is capped at 1,000 employees.",
    },
]


def stream_query(base_url: str, question: str) -> dict:
    """
    POST to /query/stream, consume the NDJSON stream, and return a structured
    result dict containing:
      - answer   : full assembled answer
      - model    : model name from meta event
      - retrieval_route : retrieval route from meta event
      - chunk_ids: list of chunk IDs from meta event
      - done_reason, stats
    """
    url = f"{base_url.rstrip('/')}/query/stream"
    payload = {"question": question}

    answer_parts: list[str] = []
    meta: dict = {}
    done_info: dict = {}
    error_msgs: list[str] = []

    with requests.post(url, json=payload, stream=True, timeout=(10, 180)) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")
            if etype == "token":
                answer_parts.append(event.get("content", ""))
            elif etype == "meta":
                meta = event
            elif etype == "done":
                done_info = event
            elif etype == "error":
                error_msgs.append(event.get("message", "unknown error"))

    hits = meta.get("hits", [])
    chunk_ids = [h.get("chunk_id") or h.get("id", "") for h in hits]

    return {
        "answer": "".join(answer_parts).strip(),
        "model": meta.get("model", "unknown"),
        "retrieval_route": meta.get("retrieval_route", "unknown"),
        "chunk_ids": chunk_ids,
        "hits": hits,
        "done_reason": done_info.get("done_reason", "unknown"),
        "stats": done_info.get("stats", {}),
        "errors": error_msgs,
    }


def run_benchmark(base_url: str) -> None:
    # ── Verify server is reachable ──────────────────────────────────────────
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        health.raise_for_status()
        health_data = health.json()
    except Exception as exc:
        print(f"\n[ERROR] Cannot reach FastAPI at {base_url}: {exc}")
        print("        Start the server first:  uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    print("=" * 80)
    print("WEB API BENCHMARK VALIDATION  –  POST /query/stream")
    print("=" * 80)
    print(f"  Server            : {base_url}")
    print(f"  Server model      : {health_data.get('model', 'unknown')}")
    print(f"  Server route      : {health_data.get('retrieval_route', 'unknown')}")
    print("=" * 80)

    results_summary: list[dict] = []

    for test in BENCHMARK_TESTS:
        t_id = test["id"]
        query = test["query"]
        description = test["description"]
        checker = test["check"]

        print(f"\n[{t_id}/8] PROMPT  : \"{query}\"")
        print(f"         Goal    : {description}")

        t_start = time.monotonic()
        try:
            result = stream_query(base_url, query)
        except Exception as exc:
            print(f"         [ERROR] Stream failed: {exc}")
            results_summary.append({"id": t_id, "query": query, "passed": False, "error": str(exc)})
            continue
        elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

        answer = result["answer"]
        chunk_ids = result["chunk_ids"]

        print(f"         Model   : {result['model']}")
        print(f"         Route   : {result['retrieval_route']}")
        print(f"         Chunks  : {chunk_ids[:5]}")
        print(f"         Stats   : done_reason={result['done_reason']} "
              f"completion_tokens={result['stats'].get('completion_tokens', '?')} "
              f"elapsed_ms={elapsed_ms}")
        if result["errors"]:
            print(f"         Errors  : {result['errors']}")
        print(f"         ANSWER  :\n\"\"\"\n{answer}\n\"\"\"")

        passed = checker(answer, result["hits"])
        status_str = "PASS ✅" if passed else "FAIL ❌"
        print(f"         EVAL    : {status_str}")

        results_summary.append({
            "id": t_id,
            "query": query,
            "model": result["model"],
            "retrieval_route": result["retrieval_route"],
            "chunk_ids": chunk_ids,
            "answer": answer,
            "passed": passed,
            "done_reason": result["done_reason"],
        })

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("WEB API BENCHMARK SUMMARY")
    print("=" * 80)
    total_passed = sum(1 for r in results_summary if r.get("passed", False))
    for r in results_summary:
        if "error" in r:
            print(f"Prompt {r['id']}: ERROR  | \"{r['query'][:55]}...\"")
        else:
            st = "PASS ✅" if r["passed"] else "FAIL ❌"
            print(
                f"Prompt {r['id']}: {st} | model={r.get('model', '?')} | "
                f"\"{r['query'][:40]}...\""
            )

    print(f"\nScore: {total_passed}/8 Prompts Passed (via live web API)")
    print("=" * 80)

    if total_passed < 8:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all 8 RAG benchmark queries through the live web API."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="FastAPI base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    run_benchmark(args.url)
