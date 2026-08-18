"""
app/services/token_budget.py
────────────────────────────
Lightweight prompt-size estimation helpers for per-session budgeting.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

from app.config import settings


@lru_cache(maxsize=1)
def _optional_encoder() -> Any | None:
    if settings.session_enforce_by != "tokens":
        return None

    try:
        import tiktoken  # type: ignore[import-not-found]

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """
    Estimate model tokens conservatively.

    When an optional tokenizer is installed we use it. Otherwise, use the higher
    of chars/4 and words*1.3 so short words and punctuation-heavy text are not
    undercounted too aggressively for qwen2.5:1.5b budgeting.
    """
    text = text or ""
    if not text:
        return 0

    encoder = _optional_encoder()
    if encoder is not None:
        return len(encoder.encode(text))

    words = re.findall(r"\S+", text)
    return max(math.ceil(len(text) / 4), math.ceil(len(words) * 1.3))


def estimate_messages_tokens(messages: list[dict[str, str]] | None) -> int:
    if not messages:
        return 0

    total = 0
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        total += estimate_tokens(f"{role}: {content}") + 4
    return total


def trim_messages_to_budget(
    messages: list[dict[str, str]] | None,
    max_tokens: int,
) -> tuple[list[dict[str, str]], int, int]:
    """
    Preserve newest messages while fitting the approximate token budget.

    Returns: (trimmed_messages, removed_count, final_estimate)
    """
    remaining = list(messages or [])
    removed = 0

    while remaining and estimate_messages_tokens(remaining) > max_tokens:
        remaining.pop(0)
        removed += 1

    return remaining, removed, estimate_messages_tokens(remaining)


def safe_input_budget() -> int:
    output_budget = max(0, settings.SESSION_MAX_OUTPUT_TOKENS)
    total_remaining = max(0, settings.SESSION_MAX_TOKENS - output_budget)
    return max(0, min(settings.SESSION_MAX_INPUT_TOKENS, total_remaining))
