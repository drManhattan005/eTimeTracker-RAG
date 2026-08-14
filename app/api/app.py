"""
app/api/app.py
───────────────
FastAPI application factory.

Startup banner logs the active model and retrieval route so it is always
visible in the server console:

  ┌──────────────────────────────────────────────────────────────┐
  │  Veloitt RAG API  – startup configuration                    │
  │  LLM model   : qwen2.5:1.5b (Ollama)                        │
  │  Retrieval   : HYBRID  dense(Qdrant/MiniLM) + BM25 → RRF    │
  │  dense pool  : k=15   BM25 pool : k=15   top hits : 12      │
  │  Chunks      : 47 prebuilt  (output/chunks.jsonl)            │
  └──────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

_startup_log = logging.getLogger("startup")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from app.config import settings
from app.domain.models import PrebuiltChunk
from app.infrastructure.embedders import FastEmbedder
from app.infrastructure.llm import OllamaClient
from app.infrastructure.vectorstores.qdrant_store import QdrantVectorStore
from app.services.generation import GenerationService
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService
from app.api.routes import make_router, RETRIEVAL_DENSE_K, RETRIEVAL_BM25_K, RETRIEVAL_LIMIT


def _load_prebuilt_chunks(path: Path) -> list[PrebuiltChunk]:
    chunks: list[PrebuiltChunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            meta = data.get("metadata", {})
            payload = {**meta, **data}
            cid = data.get("chunk_id") or data.get("id") or ""
            chunks.append(
                PrebuiltChunk(
                    chunk_id=cid,
                    text=data["text"],
                    payload=payload,
                )
            )
    return chunks


def create_app() -> FastAPI:
    chunks_file = Path(settings.CHUNKS_FILE)
    if not chunks_file.exists():
        raise RuntimeError(f"chunks.jsonl not found at: {chunks_file}")

    embedder = FastEmbedder(model_name=settings.EMBED_MODEL)

    qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION,
        vector_size=settings.QDRANT_VECTOR_SIZE,
    )

    indexing_service = IndexingService(
        chunker=None,
        embedder=embedder,
        vector_store=vector_store,
    )

    retrieval_service = RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
        chunks_file=chunks_file,
    )

    llm = OllamaClient(
        model_name=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        timeout=settings.ollama_timeout,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        options=settings.ollama_options,
    )
    generation_service = GenerationService(llm=llm)

    chunks = _load_prebuilt_chunks(chunks_file)
    indexing_service.index_prebuilt_chunks(chunks)

    # ── Startup banner ────────────────────────────────────────────────────
    bm25_status = "ACTIVE" if retrieval_service.bm25_retriever is not None else "DISABLED (no chunks file)"
    _startup_log.info(
        "\n"
        "┌──────────────────────────────────────────────────────────────┐\n"
        "│  Veloitt RAG API  – startup configuration                    │\n"
        "│                                                              │\n"
        "│  LLM model   : %-45s│\n"
        "│  Retrieval   : HYBRID  dense(Qdrant/MiniLM) + BM25 → RRF   │\n"
        "│  dense pool  : k=%-3d   BM25 pool : k=%-3d   top hits : %-4d │\n"
        "│  BM25        : %-45s│\n"
        "│  Chunks      : %-3d prebuilt  (%s)   │\n"
        "└──────────────────────────────────────────────────────────────┘",
        f"{settings.OLLAMA_MODEL} (Ollama)",
        RETRIEVAL_DENSE_K,
        RETRIEVAL_BM25_K,
        RETRIEVAL_LIMIT,
        bm25_status,
        len(chunks),
        settings.CHUNKS_FILE,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        vector_store.client.close()

    application = FastAPI(
        title="Veloitt RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    application.include_router(make_router(retrieval_service, generation_service))
    return application


app = create_app()
