"""
app/config.py
─────────────
Centralised runtime config. All values sourced from environment variables
and loaded automatically from .env in the project root.
"""
from __future__ import annotations

import os
from pathlib import Path

# Automatically load environment variables from .env in project root if present
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_file = _PROJECT_ROOT / ".env"

if _env_file.is_file():
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=_env_file, override=False)
    except ImportError:
        # Fallback simple parser if python-dotenv is not yet installed
        with _env_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v


class _Settings:
    # ── Gemini Cloud API connection ─────────────────────────────────────────
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_TEMPERATURE: float = float(os.environ.get("GEMINI_TEMPERATURE", "0.1"))
    GEMINI_MAX_OUTPUT_TOKENS: int = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
    GEMINI_TIMEOUT: int = int(os.environ.get("GEMINI_TIMEOUT", "60"))

    # ── Qdrant ─────────────────────────────────────────────────────────────
    QDRANT_PATH: str = os.environ.get("QDRANT_PATH", "./qdrant_data")
    QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "knowledge_base")
    QDRANT_VECTOR_SIZE: int = int(os.environ.get("QDRANT_VECTOR_SIZE", "384"))

    # Minimum retrieval score threshold.
    RETRIEVAL_MIN_SCORE: float = float(os.environ.get("RETRIEVAL_MIN_SCORE", "0.15"))

    # ── Embedder ────────────────────────────────────────────────────────────
    EMBED_MODEL: str = os.environ.get(
        "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── RAG prompt limits ───────────────────────────────────────────────────
    # Max characters of chunk text included in context.
    MAX_CONTEXT_CHARS: int = int(os.environ.get("MAX_CONTEXT_CHARS", "5000"))

    # ── Per-session prompt budget ──────────────────────────────────────────
    SESSION_MAX_TOKENS: int = int(os.environ.get("SESSION_MAX_TOKENS", "3000"))
    SESSION_MAX_INPUT_TOKENS: int = int(os.environ.get("SESSION_MAX_INPUT_TOKENS", "2400"))
    SESSION_MAX_OUTPUT_TOKENS: int = int(os.environ.get("SESSION_MAX_OUTPUT_TOKENS", "600"))

    SESSION_SOFT_TURN_BUDGET: int = int(os.environ.get("SESSION_SOFT_TURN_BUDGET", "800"))
    SESSION_LIFETIME_MAX_TOKENS: int = int(os.environ.get("SESSION_LIFETIME_MAX_TOKENS", "6000"))
    SESSION_LIMIT_REACHED_MESSAGE: str = os.environ.get(
        "SESSION_LIMIT_REACHED_MESSAGE",
        "Query limit reached. Restart convo.",
    )

    SESSION_ENFORCE_BY: str = os.environ.get("SESSION_ENFORCE_BY", "tokens").lower()

    # Default number of retrieval hits.
    RETRIEVAL_LIMIT_DEFAULT: int = int(os.environ.get("RETRIEVAL_LIMIT_DEFAULT", "5"))

    # ── Chunks file ─────────────────────────────────────────────────────────
    CHUNKS_FILE: str = os.environ.get("CHUNKS_FILE", "output/chunks.jsonl")

    @property
    def session_enforce_by(self) -> str:
        if self.SESSION_ENFORCE_BY in {"tokens", "chars"}:
            return self.SESSION_ENFORCE_BY
        return "tokens"


settings = _Settings()
