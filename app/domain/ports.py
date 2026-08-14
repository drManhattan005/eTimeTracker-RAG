from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from app.domain.models import Chunk, EmbeddedChunk


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class Chunker(Protocol):
    def split(self, document_text: str, metadata: dict) -> list[Chunk]:
        ...


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self) -> None:
        ...

    @abstractmethod
    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        ...
