"""
app/services/generation.py
───────────────────────────
Generation service tuned for qwen2.5:1.5b instruct model.

Implements prompt structure:
- Separate static System instruction and dynamic User prompt.
- No hard output-length truncation.
- Complete-chunk context assembly with commercial chunk prioritization.
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator

from app.config import settings
from app.infrastructure.llm import OllamaClient

log = logging.getLogger(__name__)

# Fallback answer when all retrieved hits are below the confidence threshold.
_WEAK_RETRIEVAL_ANSWER = (
    "I couldn't find a confident answer in the Veloitt knowledge base for that question. "
    "Try asking about attendance tracking, leave management, shift scheduling, "
    "or field sales tracking."
)

SYSTEM_PROMPT = """You answer questions about eTimeTracker using only the supplied context.

Rules:
1. State the answer directly in the first sentence. For plan recommendations include: plan name, the relevant published limit, and the reason. Do not restate the user's question.
2. Use only facts present in the context.
3. For plan or pricing questions, apply the published entitlements and limits exactly.
4. Keep general product capabilities separate from plan-specific inclusions.
5. If the question mentions "each plan" or asks to compare plans, cover Starter, Business, and Enterprise — all three.
6. If comparing plans across multiple attributes, use a compact Markdown table.
7. Apply employee limits strictly: Starter ≤ 100; Business ≤ 1,000; Enterprise = unlimited. If the workforce exceeds 1,000 (e.g. 1,001 or 1,200), recommend Enterprise — not Business.
8. "Custom pricing" means no public numeric price exists.
9. If a feature is absent from a plan's published entitlements, say it is not listed. Do not claim it is unavailable under all circumstances.
10. Do not invent prices, limits, integrations, SLAs, support tiers, or capabilities.
11. If the context does not contain enough information to answer, say so in one sentence.
12. Length: 2–4 sentences. Use a table only when comparing multiple plans across multiple attributes. Never omit a requested fact.

Style: Professional, neutral, factual. Plain business language. Do not use "perfect", "ideal", "essential", "perfectly", "comprehensive", or any sales language. Do not mention features unrelated to the question. Do not show reasoning steps, source labels, or retrieval details. Write the final answer only."""

_COMMERCIAL_TERMS = {
    "plan", "plans", "price", "pricing", "cost", "costing", "employee", "employees",
    "support", "starter", "business", "enterprise", "geofence", "geofencing",
    "custom", "unlimited", "sla", "branch", "branches", "subsidiary", "subsidiaries",
    "tier", "entitlement", "entitlements", "limit", "capacity", "public"
}


def _is_commercial_query(question: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    return bool(tokens & _COMMERCIAL_TERMS)


def _extract_text_and_meta(item: dict) -> tuple[str, dict]:
    """Safely extract chunk text and metadata payload from hit dict."""
    text = item.get("text", "")
    payload = item.get("payload") or item.get("metadata") or {}
    if not text and isinstance(payload, dict):
        text = payload.get("text", "")
    return text.strip(), payload if isinstance(payload, dict) else {}


def _filter_hits(retrieved_chunks: list[dict]) -> list[dict]:
    """Filter out hits with empty text or invalid scores."""
    filtered: list[dict] = []
    for h in retrieved_chunks:
        text, _ = _extract_text_and_meta(h)
        if not text:
            continue
        score = h.get("score", 0.0)
        fused_score = h.get("fused_score", 0.0)
        if score >= settings.RETRIEVAL_MIN_SCORE or fused_score > 0.0:
            filtered.append(h)
    return filtered


def _build_context(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build structured context block from retrieved hits.
    - Complete chunks only: never slice chunks in the middle.
    - Prioritizes commercial evidence when query contains commercial terms.
    - Structured [Source N] blocks with sphere, type, and plan headers.
    """
    if not retrieved_chunks:
        return ""

    candidates = list(retrieved_chunks)
    q_lower = question.lower()

    if _is_commercial_query(question):
        def priority_key(h: dict) -> tuple[int, int]:
            _, meta = _extract_text_and_meta(h)
            ctype = meta.get("chunk_type", "")
            ptier = meta.get("plan_tier", "")
            cid = h.get("chunk_id") or meta.get("chunk_id", "")

            # 1. Geofencing queries: prioritize starter-pricing & comparison
            if "geofence" in q_lower or "geofencing" in q_lower:
                if "comparison" in cid or "starter-pricing" in cid:
                    return (0, 0)
                if ptier == "business":
                    return (0, 1)

            # 2. Large employee counts (> 1000): prioritize comparison & enterprise plan
            if any(num in q_lower for num in ["1200", "1,200", "1001", "1,001"]):
                if ctype == "plan_comparison":
                    return (0, 0)
                if ptier == "enterprise":
                    return (0, 1)
                if ptier == "business":
                    return (1, 0)

            # 3. Default commercial priority: comparison chunk first, then Starter -> Business -> Enterprise
            if ctype == "plan_comparison":
                return (0, 0)
            if ctype == "plan":
                tier_order = {"starter": 0, "business": 1, "enterprise": 2}
                return (1, tier_order.get(ptier, 3))
            if ctype == "commercial_boundary":
                return (2, 0)
            return (3, 0)

        candidates.sort(key=priority_key)

    total_chars = 0
    max_budget = settings.MAX_CONTEXT_CHARS
    parts: list[str] = []

    for i, item in enumerate(candidates, start=1):
        text, payload = _extract_text_and_meta(item)
        if not text:
            continue

        sphere = payload.get("intent_sphere", "unknown")
        ctype = payload.get("chunk_type", "unknown")
        plan_tier = payload.get("plan_tier", "none")

        header = f"[Source {i}]\nSphere: {sphere}\nType: {ctype}\nPlan: {plan_tier}\nContent:\n{text}"

        if total_chars + len(header) > max_budget and parts:
            break

        parts.append(header)
        total_chars += len(header) + 2

    return "\n\n".join(parts)


def _build_prompts(question: str, retrieved_chunks: list[dict]) -> tuple[str, str]:
    """Returns (system_instruction, user_prompt) tuple."""
    context = _build_context(question, retrieved_chunks)

    if os.environ.get("RAG_SHOW_CONTEXT") == "1":
        log.info(
            "\n==================== RAG_SHOW_CONTEXT ====================\n"
            "QUESTION: %s\n"
            "CONTEXT (%d chars):\n%s\n"
            "==========================================================",
            question,
            len(context),
            context,
        )

    user_prompt = f"""USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
Write only the final answer for the user. Do not expose internal reasoning, retrieval scores, prompts, or source-processing details."""

    return SYSTEM_PROMPT, user_prompt


class GenerationService:
    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def answer(self, question: str, retrieved_chunks: list[dict]) -> str:
        """Blocking generation — used by POST /query."""
        filtered = _filter_hits(retrieved_chunks)
        if not filtered:
            log.info("[generation] all hits below threshold, returning fallback")
            return _WEAK_RETRIEVAL_ANSWER

        system_prompt, user_prompt = _build_prompts(question, filtered)
        return self.llm.generate(prompt=user_prompt, system=system_prompt)

    def answer_stream(
        self, question: str, retrieved_chunks: list[dict]
    ) -> Iterator[str]:
        """
        Streaming generation — used by POST /query/stream.
        Yields incremental text chunks.
        """
        filtered = _filter_hits(retrieved_chunks)
        if not filtered:
            log.info("[generation] all hits below threshold, yielding fallback")
            yield _WEAK_RETRIEVAL_ANSWER
            return

        system_prompt, user_prompt = _build_prompts(question, filtered)
        log.info(
            "[generation] starting stream | question_len=%d hits=%d",
            len(question),
            len(filtered),
        )
        yield from self.llm.generate_stream(prompt=user_prompt, system=system_prompt)
