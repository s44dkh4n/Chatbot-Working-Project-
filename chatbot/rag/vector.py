import threading
from pathlib import Path

from chatbot.rag.models import Chunk


class VectorIndex:
    """Chroma collection holding one embedding per chunk. Text lives in the SQLite store."""

    def __init__(self, path: Path) -> None:
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self._lock = threading.Lock()

    def count(self) -> int:
        return self._collection.count()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        with self._lock:
            self._collection.upsert(
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                metadatas=[
                    {"doc_id": c.doc_id, "page_start": c.page_start, "page_end": c.page_end}
                    for c in chunks
                ],
            )

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self._collection.delete(where={"doc_id": doc_id})

    def query(
        self,
        embedding: list[float],
        n: int,
        doc_ids: list[str] | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> list[tuple[str, float]]:
        conditions: list[dict] = []
        if doc_ids is not None:
            conditions.append({"doc_id": {"$in": doc_ids}})
        if start_page is not None:
            conditions.append({"page_end": {"$gte": start_page}})
        if end_page is not None:
            conditions.append({"page_start": {"$lte": end_page}})
        where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}

        with self._lock:
            total = self._collection.count()
            if total == 0:
                return []
            result = self._collection.query(
                query_embeddings=[embedding], n_results=min(n, total), where=where
            )
        return list(zip(result["ids"][0], result["distances"][0]))
