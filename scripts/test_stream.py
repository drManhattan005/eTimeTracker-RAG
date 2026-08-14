#!/usr/bin/env python3
"""
scripts/test_stream.py
───────────────────────
Minimal CLI test for the /query/stream endpoint.

Use this to verify whether truncation is backend-side or frontend-side.
Run it while the FastAPI server is running.

Usage:
    python scripts/test_stream.py
    python scripts/test_stream.py "How does leave approval work?"
    python scripts/test_stream.py "your question here" --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests


def test_stream(question: str, base_url: str) -> None:
    url = f"{base_url.rstrip('/')}/query/stream"
    payload = {"question": question, "limit": 3}

    print(f"\n{'='*60}")
    print(f"  Question : {question}")
    print(f"  Endpoint : {url}")
    print(f"{'='*60}\n")

    t_start = time.monotonic()
    token_count = 0
    full_answer = ""

    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, 180)) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Malformed NDJSON line: {raw_line[:80]} | {exc}")
                    continue

                etype = event.get("type", "?")

                if etype == "token":
                    content = event.get("content", "")
                    full_answer += content
                    token_count += 1
                    # Print tokens inline without newlines for real-time feel
                    sys.stdout.write(content)
                    sys.stdout.flush()

                elif etype == "meta":
                    hits = event.get("hits", [])
                    print(f"\n\n[META] {len(hits)} hits received")
                    for i, h in enumerate(hits, 1):
                        section = h.get("section", {})
                        label = " › ".join(
                            v for v in [section.get("h2"), section.get("h3")] if v
                        )
                        print(f"  [{i}] score={h.get('score', 0):.4f}  {label}")

                elif etype == "error":
                    print(f"\n[ERROR] {event.get('message', 'unknown')}")

                elif etype == "done":
                    elapsed = round((time.monotonic() - t_start) * 1000, 1)
                    stats = event.get("stats", {})
                    reason = event.get("done_reason", "?")
                    print(f"\n\n{'='*60}")
                    print(f"  done_reason      : {reason}")
                    print(f"  prompt_tokens    : {stats.get('prompt_tokens', '?')}")
                    print(f"  completion_tokens: {stats.get('completion_tokens', '?')}")
                    print(f"  token_events     : {token_count}")
                    print(f"  answer_chars     : {len(full_answer)}")
                    print(f"  total_elapsed_ms : {elapsed}")
                    print(f"{'='*60}\n")

                    if reason == "length":
                        print("⚠  WARNING: done_reason=length — answer was cut off by num_predict cap!")
                        print("   Increase OLLAMA_NUM_PREDICT env var (currently 1024).")
                    elif reason == "stop":
                        print("✓  Stream ended cleanly at stop token.")

    except requests.Timeout:
        print("\n[TIMEOUT] The request timed out. "
              "Increase OLLAMA_READ_TIMEOUT or check if Ollama is running.")
        sys.exit(1)
    except requests.ConnectionError:
        print(f"\n[ERROR] Could not connect to {url}. Is the FastAPI server running?")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[CANCELLED] Interrupted by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test /query/stream endpoint")
    parser.add_argument(
        "question",
        nargs="?",
        default="How does eTimeTracker handle field sales attendance?",
        help="Question to send",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="FastAPI base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    test_stream(args.question, args.url)
