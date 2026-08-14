"""
app/main.py
────────────
Entry point for uvicorn:
    uvicorn app.main:app --reload

For the standalone CLI script (index + smoke test), run:
    python -m app.main
"""
from __future__ import annotations

# Re-export the FastAPI app so `uvicorn app.main:app` works.
from app.api.app import app  # noqa: F401

if __name__ == "__main__":
    import json
    from pathlib import Path

    from app.config import settings
    from app.domain.models import PrebuiltChunk
    from app.infrastructure.embedders import FastEmbedder
    from app.infrastructure.vectorstores.qdrant_store import QdrantVectorStore
    from app.services.generation import GenerationService
    from app.services.indexing import IndexingService
    from app.services.retrieval import RetrievalService
    from qdrant_client import QdrantClient

    chunks_file = Path(settings.CHUNKS_FILE)
    if not chunks_file.exists():
        raise SystemExit(
            f"chunks.jsonl not found at: {chunks_file}. Run the chunking script first."
        )

    embedder = FastEmbedder(model_name=settings.EMBED_MODEL)
    qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION,
        vector_size=settings.QDRANT_VECTOR_SIZE,
    )
    indexing_service = IndexingService(
        chunker=None, embedder=embedder, vector_store=vector_store
    )
    retrieval_service = RetrievalService(
        embedder=embedder, vector_store=vector_store, chunks_file=chunks_file
    )

    chunks: list[PrebuiltChunk] = []
    with chunks_file.open(encoding="utf-8") as f:
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

    print(f"Loaded {len(chunks)} prebuilt chunks from {chunks_file}")
    indexed = indexing_service.index_prebuilt_chunks(chunks)
    print(f"Indexed: {indexed} chunks into Qdrant")

    results = retrieval_service.search("field sales attendance tracking", limit=3)
    print("\nSearch results:")
    for item in results:
        print(item)

    vector_store.client.close()
