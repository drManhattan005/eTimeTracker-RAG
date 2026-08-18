"""
app/config.py
─────────────
Centralised runtime config. All values sourced from environment variables.
"""
from __future__ import annotations

import os


class _Settings:
    # ── Ollama connection ───────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # Connection timeout (seconds): how long to wait for the TCP handshake.
    OLLAMA_CONNECT_TIMEOUT: int = int(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "10"))

    # Read timeout (seconds): how long to wait for each new chunk from Ollama.
    OLLAMA_READ_TIMEOUT: int = int(os.environ.get("OLLAMA_READ_TIMEOUT", "180"))

    # ── Ollama model options ────────────────────────────────────────────────
    # Authoritative default model for 1.5B generation layer
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

    # Keep model loaded between requests. "-1" = forever, "10m" = 10 min.
    OLLAMA_KEEP_ALIVE: str = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")

    # Context window (tokens). 4096 comfortably holds instructions + context + prompt + output.
    OLLAMA_NUM_CTX: int = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))

    # Generous safety ceiling max output tokens (not a target limit).
    OLLAMA_NUM_PREDICT: int = int(os.environ.get("OLLAMA_NUM_PREDICT", "1024"))

    # Temperature. Low for deterministic factual answers.
    OLLAMA_TEMPERATURE: float = float(os.environ.get("OLLAMA_TEMPERATURE", "0.1"))

    # ── Ollama server-level env vars (NOT sent per-request) ─────────────
    OLLAMA_FLASH_ATTENTION: str = os.environ.get("OLLAMA_FLASH_ATTENTION", "0")
    OLLAMA_KV_CACHE_TYPE: str = os.environ.get("OLLAMA_KV_CACHE_TYPE", "f16")

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
    # ~4 chars ≈ 1 token. 5000 chars ≈ 1250 tokens.
    MAX_CONTEXT_CHARS: int = int(os.environ.get("MAX_CONTEXT_CHARS", "5000"))

    # ── Per-session prompt budget ──────────────────────────────────────────
    # qwen2.5:1.5b has a 4096-token context window here, but the app should not
    # spend that whole window on prompt input. These defaults reserve practical
    # headroom for the system prompt, retrieved chunks, formatting, concurrent
    # server load, and a bounded answer from a small local model.
    SESSION_MAX_TOKENS: int = int(os.environ.get("SESSION_MAX_TOKENS", "3000"))
    SESSION_MAX_INPUT_TOKENS: int = int(os.environ.get("SESSION_MAX_INPUT_TOKENS", "2400"))
    SESSION_MAX_OUTPUT_TOKENS: int = int(os.environ.get("SESSION_MAX_OUTPUT_TOKENS", "600"))

    # Soft cap for prior kept conversation turns inside each request. The route
    # still keeps only the last 3 turns first; this is a second guardrail that
    # trims unusually long turns while preserving the newest messages.
    SESSION_SOFT_TURN_BUDGET: int = int(os.environ.get("SESSION_SOFT_TURN_BUDGET", "800"))

    # Hard cumulative lifetime cap for one ephemeral conversation/session. Once
    # reached, the session must be reset before accepting another query.
    SESSION_LIFETIME_MAX_TOKENS: int = int(os.environ.get("SESSION_LIFETIME_MAX_TOKENS", "6000"))
    SESSION_LIMIT_REACHED_MESSAGE: str = os.environ.get(
        "SESSION_LIMIT_REACHED_MESSAGE",
        "Query limit reached. Restart convo.",
    )

    # "tokens" uses an optional tokenizer if installed, otherwise a conservative
    # approximation. "chars" forces the same chars/word fallback estimator.
    SESSION_ENFORCE_BY: str = os.environ.get("SESSION_ENFORCE_BY", "tokens").lower()

    # Default number of retrieval hits.
    RETRIEVAL_LIMIT_DEFAULT: int = int(os.environ.get("RETRIEVAL_LIMIT_DEFAULT", "5"))

    # ── Chunks file ─────────────────────────────────────────────────────────
    CHUNKS_FILE: str = os.environ.get("CHUNKS_FILE", "output/chunks.jsonl")

    @property
    def ollama_options(self) -> dict:
        """Per-request options dict passed to Ollama /api/generate."""
        return {
            "num_ctx": self.OLLAMA_NUM_CTX,
            "num_predict": min(self.OLLAMA_NUM_PREDICT, self.SESSION_MAX_OUTPUT_TOKENS),
            "temperature": self.OLLAMA_TEMPERATURE,
        }

    @property
    def ollama_timeout(self) -> tuple[int, int]:
        """(connect_timeout, read_timeout) for requests."""
        return (self.OLLAMA_CONNECT_TIMEOUT, self.OLLAMA_READ_TIMEOUT)

    @property
    def session_enforce_by(self) -> str:
        if self.SESSION_ENFORCE_BY in {"tokens", "chars"}:
            return self.SESSION_ENFORCE_BY
        return "tokens"


settings = _Settings()
