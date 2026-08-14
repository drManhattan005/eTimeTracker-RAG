from fastembed import TextEmbedding

from app.domain.ports import Embedder


class FastEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self.model.embed(texts)]
