from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddedChunk:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(slots=True)
class PrebuiltChunk:
    """A chunk already produced by the chunking script — no further splitting needed."""
    chunk_id: str
    text: str
    payload: dict[str, Any]
