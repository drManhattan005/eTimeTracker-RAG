"""
app/services/generation.py
───────────────────────────
Generation service for Veloitt RAG with a lightweight classifier-first
abstention layer.

Policy:
- BLOCK obvious junk, greetings, social chatter, and clearly off-domain queries.
- ALLOW anything plausibly product-related or answerable from the Veloitt corpus.
- Retrieval remains responsible for retrieval-oriented rewriting and search.
- If retrieval yields no usable context, abstain as a final safeguard.

This design keeps the classifier narrow and low-risk for a small local model.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.infrastructure.llm import OllamaClient
from app.services.token_budget import estimate_tokens, safe_input_budget, trim_messages_to_budget

log = logging.getLogger(__name__)

ABSTAIN_TEMPLATE = (
    "I’m not confident I can answer that reliably from the available Veloitt knowledge base. "
    "Please contact a human team member for accurate help."
)

SYSTEM_PROMPT = """You answer questions about Veloitt using only the supplied context.

Rules:
1. Answer directly in the first sentence.
2. Use only facts present in the supplied context.
3. Do not invent product capabilities, pricing, limits, integrations, workflows, or policies.
4. If the context is thin, stay concise and only state what is supported by the evidence.
5. Do not mention retrieval, search, chunks, scores, or internal reasoning.
6. Write the final answer only.
"""

CLASSIFIER_SYSTEM_PROMPT = """You are a lightweight gatekeeper for a Veloitt product assistant.

Return JSON only with this exact schema:
{"label":"ALLOW"|"BLOCK","reason":"short reason"}

Use BLOCK only for:
- greetings only,
- social-only chat,
- nonsense, gibberish, keyboard smash, filler,
- clearly off-topic queries unrelated to Veloitt, eTimeTracker, workforce management,
  attendance, leave, shifts, approvals, onboarding, support, pricing, or product usage.

Use ALLOW for:
- any plausible Veloitt or eTimeTracker question,
- short but product-ish queries,
- vague queries that still seem related to attendance, leave, workforce, pricing,
  support, setup, implementation, or platform usage.

Important:
- Be permissive for product-ish queries.
- If unsure, choose ALLOW.
- Return JSON only. No markdown.
"""

REWRITER_SYSTEM_PROMPT = """You rewrite user queries for a Veloitt product assistant.

Return JSON only with this exact schema:
{"rewritten_query":"..."}

Rules:
1. Preserve the user's intent.
2. Make the query clearer and more retrieval-friendly.
3. Expand shorthand only when it is clearly implied.
4. Do not invent product capabilities, policies, or facts.
5. Keep it concise.
6. If the original query is already clear, return it with minimal change.
7. Return JSON only. No markdown.
"""

FOLLOWUP_REWRITER_SYSTEM_PROMPT = """You rewrite a follow-up user query into a standalone Veloitt product question.

Return JSON only with this exact schema:
{"rewritten_query":"..."}

Rules:
1. Use the recent conversation only to resolve references in the latest user query.
2. Preserve the user's intent exactly.
3. Keep it concise and retrieval-friendly.
4. Do not invent product capabilities, policies, or facts.
5. If the latest user query is already standalone, return it with minimal change.
6. Return JSON only. No markdown.
"""


class GateDecision(BaseModel):
    label: Literal["ALLOW", "BLOCK"] = Field(
        ...,
        description="Whether the query should proceed to retrieval/generation.",
    )
    reason: str = Field(
        ...,
        description="Short internal reason for logs only.",
    )


class RewriteDecision(BaseModel):
    rewritten_query: str = Field(
        ...,
        description="A clearer, retrieval-friendly rewrite of the user query.",
    )


@dataclass(frozen=True)
class BudgetedContext:
    text: str
    original_hits: int
    included_hits: int
    context_tokens: int
    retrieval_trimmed: bool


_COMMERCIAL_TERMS = {
    "plan", "plans", "price", "pricing", "cost", "costing", "employee", "employees",
    "support", "starter", "business", "enterprise", "geofence", "geofencing",
    "custom", "unlimited", "sla", "branch", "branches", "subsidiary", "subsidiaries",
    "tier", "entitlement", "entitlements", "limit", "capacity", "public",
}

_OBVIOUS_BLOCKLIST = {
    "hi", "hello", "hey", "yo", "sup", "thanks", "thank you", "ok", "okay", "cool",
    "nice", "hmm", "huh", "lol", "test",
}

_SHORT_FOLLOWUP_PATTERNS = {
    "why", "how", "what about that", "what about it", "and that", "and this",
    "can it", "does it", "is it", "why is that", "how about that",
}


def _is_commercial_query(question: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    return bool(tokens & _COMMERCIAL_TERMS)


def _extract_text_and_meta(item: dict) -> tuple[str, dict]:
    text = item.get("text", "")
    payload = item.get("payload") or item.get("metadata") or {}
    if not text and isinstance(payload, dict):
        text = payload.get("text", "")
    return text.strip(), payload if isinstance(payload, dict) else {}


def _filter_hits(retrieved_chunks: list[dict]) -> list[dict]:
    filtered: list[dict] = []

    for hit in retrieved_chunks:
        text, _ = _extract_text_and_meta(hit)
        if not text:
            continue
        filtered.append(hit)

    return filtered


def _normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def _looks_like_gibberish(q: str) -> bool:
    stripped = re.sub(r"[^a-zA-Z]", "", q or "")
    if not stripped:
        return True
    if len(stripped) <= 2:
        return True
    vowels = sum(1 for ch in stripped.lower() if ch in "aeiou")
    vowel_ratio = vowels / max(len(stripped), 1)
    return len(stripped) >= 6 and vowel_ratio < 0.15


def _cheap_block_check(question: str) -> str | None:
    q = _normalize_query(question)
    if not q:
        return "Empty query."
    if q in _OBVIOUS_BLOCKLIST:
        return "Greeting/social-only query."
    if _looks_like_gibberish(q):
        return "Query appears to be gibberish or too low-signal."
    return None


def _format_context_block(item: dict, idx: int) -> str:
    text, meta = _extract_text_and_meta(item)
    if not text:
        return ""

    cid = item.get("chunk_id") or meta.get("chunk_id") or item.get("id", f"chunk_{idx}")
    section = meta.get("heading_path_text") or meta.get("section", "")
    plan_tier = meta.get("plan_tier", "")

    header = f"[Source: {cid}]"
    if section:
        header += f" {section}"
    if plan_tier:
        header += f" (Tier: {plan_tier})"

    return f"{header}\n{text}"


def _build_context(
    question: str,
    retrieved_chunks: list[dict],
    max_context_tokens: int | None = None,
) -> BudgetedContext:
    if not retrieved_chunks:
        return BudgetedContext(
            text="",
            original_hits=0,
            included_hits=0,
            context_tokens=0,
            retrieval_trimmed=False,
        )

    candidates = list(retrieved_chunks)
    q_lower = question.lower()

    if _is_commercial_query(question):
        def priority_key(hit: dict) -> tuple[int, float]:
            _, meta = _extract_text_and_meta(hit)
            cid = (hit.get("chunk_id") or meta.get("chunk_id", "")).lower()
            plan_tier = str(meta.get("plan_tier", "")).lower()
            score = float(hit.get("fused_score", 0.0))

            if "geofence" in q_lower or "geofencing" in q_lower:
                if "starter-pricing" in cid or "comparison" in cid or plan_tier == "starter":
                    return (1, score)
            return (0, score)

        candidates.sort(key=priority_key, reverse=True)

    context_parts: list[str] = []
    total_chars = 0
    total_tokens = 0
    usable_hits = 0

    for idx, item in enumerate(candidates, start=1):
        block = _format_context_block(item, idx)
        if not block:
            continue
        usable_hits += 1

        if total_chars + len(block) > settings.MAX_CONTEXT_CHARS:
            break

        block_with_separator = block if not context_parts else f"\n\n{block}"
        block_tokens = estimate_tokens(block_with_separator)
        if max_context_tokens is not None and total_tokens + block_tokens > max_context_tokens:
            break

        context_parts.append(block)
        total_chars += len(block)
        total_tokens += block_tokens

    return BudgetedContext(
        text="\n\n".join(context_parts),
        original_hits=usable_hits,
        included_hits=len(context_parts),
        context_tokens=total_tokens,
        retrieval_trimmed=len(context_parts) < usable_hits,
    )


def _history_to_text(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""

    lines: list[str] = []
    for item in history[-8:]:
        role = (item.get("role") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


def _build_answer_prompt(
    user_question: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
    )

    history_text = _history_to_text(history)
    if history_text:
        prompt += f"Recent conversation:\n{history_text}\n\n"

    prompt += (
        f"Question:\n{user_question}\n\n"
        "Answer:"
    )
    return prompt


def _looks_like_followup(question: str, history: list[dict[str, str]] | None) -> bool:
    if not history:
        return False

    q = _normalize_query(question)
    if len(q.split()) <= 4:
        if q in _SHORT_FOLLOWUP_PATTERNS:
            return True
        if any(token in {"it", "that", "this", "they", "those", "these"} for token in q.split()):
            return True
    return False


class GenerationService:
    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def _classify_query(self, question: str) -> GateDecision:
        cheap_block_reason = _cheap_block_check(question)
        if cheap_block_reason:
            return GateDecision(label="BLOCK", reason=cheap_block_reason)

        prompt = (
            "Classify whether this user query should be allowed through for a Veloitt product assistant.\n\n"
            f"User query:\n{question}\n"
        )

        raw = self.llm.generate(prompt=prompt, system=CLASSIFIER_SYSTEM_PROMPT)

        try:
            data = json.loads(raw)
            decision = GateDecision.model_validate(data)
            return decision
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning(
                "[generation] classifier parse failure; defaulting to ALLOW | raw=%r err=%s",
                raw[:300],
                exc,
            )
            return GateDecision(
                label="ALLOW",
                reason="Classifier output was invalid; defaulted to allow.",
            )

    def _rewrite_query(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return q

        prompt = (
            "Rewrite this user query to be clearer and retrieval-friendly for a Veloitt product assistant.\n\n"
            f"User query:\n{q}\n"
        )

        raw = self.llm.generate(prompt=prompt, system=REWRITER_SYSTEM_PROMPT)

        try:
            data = json.loads(raw)
            rewritten = RewriteDecision.model_validate(data).rewritten_query.strip()
            if not rewritten:
                return q
            if len(rewritten) > 300:
                return q
            return rewritten
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning(
                "[generation] rewrite parse failure; using original query | raw=%r err=%s",
                raw[:300],
                exc,
            )
            return q

    def _rewrite_followup(self, question: str, history: list[dict[str, str]] | None) -> str:
        q = (question or "").strip()
        if not q or not history:
            return q

        prompt = (
            "Rewrite the latest user query into a standalone Veloitt question using the recent conversation only when needed.\n\n"
            f"Recent conversation:\n{_history_to_text(history)}\n\n"
            f"Latest user query:\n{q}\n"
        )

        raw = self.llm.generate(prompt=prompt, system=FOLLOWUP_REWRITER_SYSTEM_PROMPT)

        try:
            data = json.loads(raw)
            rewritten = RewriteDecision.model_validate(data).rewritten_query.strip()
            if not rewritten:
                return q
            if len(rewritten) > 300:
                return q
            return rewritten
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning(
                "[generation] follow-up rewrite parse failure; using original query | raw=%r err=%s",
                raw[:300],
                exc,
            )
            return q

    def prepare_effective_query(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if _looks_like_followup(question, history):
            return self._rewrite_followup(question, history)
        return self._rewrite_query(question)

    def _generate_answer(
        self,
        user_question: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        return self.llm.generate(_build_answer_prompt(user_question, context, history))

    def _generate_answer_stream(
        self,
        user_question: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        yield from self.llm.generate_stream(_build_answer_prompt(user_question, context, history))

    def _prepare_budgeted_context(
        self,
        question: str,
        effective_question: str,
        retrieved_chunks: list[dict],
        history: list[dict[str, str]] | None,
    ) -> tuple[list[dict[str, str]], BudgetedContext, dict]:
        input_budget = safe_input_budget()
        history_budget = min(settings.SESSION_SOFT_TURN_BUDGET, input_budget)
        history_for_prompt, history_trimmed, history_tokens = trim_messages_to_budget(
            history,
            history_budget,
        )

        empty_prompt = _build_answer_prompt(question, "", history_for_prompt)
        base_tokens = estimate_tokens(empty_prompt)

        while history_for_prompt and base_tokens >= input_budget:
            history_for_prompt = history_for_prompt[1:]
            history_trimmed += 1
            history_tokens = estimate_tokens(_history_to_text(history_for_prompt))
            empty_prompt = _build_answer_prompt(question, "", history_for_prompt)
            base_tokens = estimate_tokens(empty_prompt)

        context_budget = max(0, input_budget - base_tokens)
        filtered = _filter_hits(retrieved_chunks)
        context = _build_context(
            effective_question,
            filtered,
            max_context_tokens=context_budget,
        )
        final_prompt_tokens = estimate_tokens(
            _build_answer_prompt(question, context.text, history_for_prompt)
        )

        budget = {
            "session_estimate_tokens": history_tokens,
            "history_trimmed_messages": history_trimmed,
            "retrieval_trimmed": context.retrieval_trimmed,
            "retrieval_hits_included": context.included_hits,
            "retrieval_hits_available": context.original_hits,
            "context_estimate_tokens": context.context_tokens,
            "final_prompt_estimate_tokens": final_prompt_tokens,
            "input_budget_tokens": input_budget,
            "output_budget_tokens": settings.SESSION_MAX_OUTPUT_TOKENS,
        }

        log.info(
            "[generation] budget | session_tokens=%d trimmed_history=%d "
            "retrieval_trimmed=%s hits=%d/%d final_prompt_tokens=%d input_budget=%d",
            budget["session_estimate_tokens"],
            budget["history_trimmed_messages"],
            budget["retrieval_trimmed"],
            budget["retrieval_hits_included"],
            budget["retrieval_hits_available"],
            budget["final_prompt_estimate_tokens"],
            budget["input_budget_tokens"],
        )

        return history_for_prompt, context, budget

    def answer(
        self,
        question: str,
        retrieved_chunks: list[dict] | None = None,
        *,
        effective_question: str | None = None,
        hits: list[dict] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        chunks = retrieved_chunks if retrieved_chunks is not None else hits or []
        result = self.answer_or_abstain(
            question,
            chunks,
            effective_question=effective_question,
            history=history,
        )
        return result["answer"]

    def answer_stream(
        self,
        question: str,
        retrieved_chunks: list[dict],
        *,
        effective_question: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        result = self.answer_or_abstain(
            question,
            retrieved_chunks,
            effective_question=effective_question,
            history=history,
        )

        if result["type"] == "ABSTAIN":
            yield result["answer"]
            return

        effective_question = result.get("rewritten_query") or effective_question or question
        history_for_prompt, context, _ = self._prepare_budgeted_context(
            question,
            effective_question,
            retrieved_chunks,
            history,
        )

        if not context.text:
            yield ABSTAIN_TEMPLATE
            return

        yield from self._generate_answer_stream(question, context.text, history=history_for_prompt)

    def answer_or_abstain(
        self,
        question: str,
        retrieved_chunks: list[dict],
        *,
        effective_question: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict:
        gate = self._classify_query(question)

        if gate.label == "BLOCK":
            log.info("[generation] policy=ABSTAIN reasoning=%r", gate.reason)
            return {
                "type": "ABSTAIN",
                "answer": ABSTAIN_TEMPLATE,
                "reasoning": gate.reason,
                "rewritten_query": question,
            }

        rewritten_query = (effective_question or "").strip()
        if not rewritten_query:
            rewritten_query = self.prepare_effective_query(question, history=history)
        history_for_prompt, context, budget = self._prepare_budgeted_context(
            question,
            rewritten_query,
            retrieved_chunks,
            history,
        )

        if not context.text:
            reasoning = "Allowed query, but retrieval produced no usable context."
            log.info("[generation] policy=ABSTAIN reasoning=%r", reasoning)
            return {
                "type": "ABSTAIN",
                "answer": ABSTAIN_TEMPLATE,
                "reasoning": reasoning,
                "rewritten_query": rewritten_query,
                "budget": budget,
            }

        answer_text = self._generate_answer(question, context.text, history=history_for_prompt)
        log.info(
            "[generation] policy=ANSWER reasoning=%r rewritten_query=%r",
            gate.reason,
            rewritten_query,
        )
        return {
            "type": "ANSWER",
            "answer": answer_text,
            "reasoning": gate.reason,
            "rewritten_query": rewritten_query,
            "budget": budget,
        }
