from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 32) -> None:
        # Imported here so the chatbot starts fast when RAG is not used
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._prefix = query_prefix
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            self._prefix + text, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder

        self._batch_size = batch_size
        self._model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


class FlashRankReranker:
    """Small, fast CPU reranker (ONNX, no PyTorch needed). Models download once into cache_dir."""

    def __init__(self, model_name: str, cache_dir: Path, max_length: int = 512) -> None:
        from flashrank import Ranker

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._ranker = Ranker(
            model_name=model_name,
            cache_dir=str(cache_dir),
            max_length=max_length,
            log_level="WARNING",
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        from flashrank import RerankRequest

        # FlashRank returns the passages sorted by score, so ids are used to restore the input order
        request = RerankRequest(
            query=query,
            passages=[{"id": i, "text": text} for i, text in enumerate(passages)],
        )
        scores = [0.0] * len(passages)
        for item in self._ranker.rerank(request):
            scores[item["id"]] = float(item["score"])
        return scores
