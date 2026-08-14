from __future__ import annotations

from collections.abc import Iterable

from app.domain.models import Document, EmbeddedChunk, PrebuiltChunk
from app.domain.ports import Chunker, Embedder, VectorStore


class IndexingService:
    def __init__(
        self,
        chunker: Chunker | None,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def index_documents(self, documents: Iterable[Document]) -> int:
        if self.chunker is None:
            raise RuntimeError("No chunker configured. Use index_prebuilt_chunks instead.")

        chunks = []
        for document in documents:
            chunks.extend(
                self.chunker.split(
                    document_text=document.text,
                    metadata={**document.metadata, "document_id": document.id},
                )
            )

        if not chunks:
            return 0

        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        embedded_chunks = [
            EmbeddedChunk(
                id=chunk.id,
                vector=vector,
                payload={**chunk.metadata, "text": chunk.text},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self.vector_store.ensure_collection()
        self.vector_store.upsert(embedded_chunks)
        return len(embedded_chunks)

    def index_prebuilt_chunks(self, chunks: Iterable[PrebuiltChunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        vectors = self.embedder.embed([chunk.text for chunk in chunk_list])
        embedded_chunks = [
            EmbeddedChunk(
                id=chunk.chunk_id,
                vector=vector,
                payload={**chunk.payload, "text": chunk.text},
            )
            for chunk, vector in zip(chunk_list, vectors, strict=True)
        ]

        self.vector_store.ensure_collection()
        self.vector_store.upsert(embedded_chunks)
        return len(embedded_chunks)
