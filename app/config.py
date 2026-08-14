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

    # Default number of retrieval hits.
    RETRIEVAL_LIMIT_DEFAULT: int = int(os.environ.get("RETRIEVAL_LIMIT_DEFAULT", "5"))

    # ── Chunks file ─────────────────────────────────────────────────────────
    CHUNKS_FILE: str = os.environ.get("CHUNKS_FILE", "output/chunks.jsonl")

    @property
    def ollama_options(self) -> dict:
        """Per-request options dict passed to Ollama /api/generate."""
        return {
            "num_ctx": self.OLLAMA_NUM_CTX,
            "num_predict": self.OLLAMA_NUM_PREDICT,
            "temperature": self.OLLAMA_TEMPERATURE,
        }

    @property
    def ollama_timeout(self) -> tuple[int, int]:
        """(connect_timeout, read_timeout) for requests."""
        return (self.OLLAMA_CONNECT_TIMEOUT, self.OLLAMA_READ_TIMEOUT)


settings = _Settings()
