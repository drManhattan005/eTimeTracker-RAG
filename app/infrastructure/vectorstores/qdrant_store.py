import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.domain.models import EmbeddedChunk
from app.domain.ports import VectorStore

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
        )

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=_to_uuid(chunk.id),
                    vector=chunk.vector,
                    payload={**chunk.payload, "chunk_id": chunk.id},
                )
                for chunk in chunks
            ],
        )

    def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )

        return [
            {
                "id": point.payload.get("chunk_id", point.id),
                "score": point.score,
                "payload": point.payload,
            }
            for point in result.points
        ]
