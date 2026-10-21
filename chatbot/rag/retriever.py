import logging
import time

from chatbot.rag.embedder import Embedder, Reranker
from chatbot.rag.models import Hit, SearchResult
from chatbot.rag.store import RagStore
from chatbot.rag.textutil import words
from chatbot.rag.vector import VectorIndex

logger = logging.getLogger(__name__)

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: merge ranked lists without having to compare their raw scores."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def dedupe(hits: list[Hit], threshold: float = 0.85) -> list[Hit]:
    """Drop hits whose wording is almost identical to a better ranked hit."""
    kept: list[Hit] = []
    kept_sets: list[set[str]] = []
    for hit in hits:
        tokens = set(words(hit.chunk.text))
        if any(len(tokens & other) / max(1, len(tokens | other)) >= threshold for other in kept_sets):
            continue
        kept.append(hit)
        kept_sets.append(tokens)
    return kept


class HybridRetriever:
    def __init__(
        self,
        store: RagStore,
        index: VectorIndex,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 30,
        rerank_pool: int = 20,
        min_rerank_score: float | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_pool = rerank_pool
        self.min_rerank_score = min_rerank_score

    def search(
        self,
        query: str,
        top_k: int = 5,
        document: str | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        mode: str = "hybrid_rerank",
    ) -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}', expected one of {MODES}")

        started = time.perf_counter()
        timings: dict[str, float] = {}

        doc_ids, names = self.store.resolve_documents(document)
        if doc_ids is not None and not doc_ids:
            available = ", ".join(names) or "none"
            return SearchResult([], [], timings, f"Unknown document '{document}'. Available: {available}.")
        filters = {"doc_ids": doc_ids, "start_page": start_page, "end_page": end_page}

        vector_ids: list[str] = []
        bm25_ids: list[str] = []
        if mode != "bm25":
            step = time.perf_counter()
            embedding = self.embedder.embed_query(query)
            vector_ids = [i for i, _ in self.index.query(embedding, self.candidates, **filters)]
            timings["vector"] = (time.perf_counter() - step) * 1000
        if mode != "vector":
            step = time.perf_counter()
            bm25_ids = [i for i, _ in self.store.bm25_search(query, self.candidates, **filters)]
            timings["bm25"] = (time.perf_counter() - step) * 1000

        fused = rrf([ranking for ranking in (vector_ids, bm25_ids) if ranking])
        pool_ids = [chunk_id for chunk_id, _ in fused[: self.rerank_pool]]

        chunks = self.store.get_chunks(pool_ids)
        hits = [
            Hit(chunk=chunks[i][0], document=chunks[i][1], score=score)
            for i, score in fused[: self.rerank_pool]
            if i in chunks
        ]

        message = None
        if mode == "hybrid_rerank":
            if self.reranker is None:
                logger.warning("No reranker configured, using hybrid order")
            elif hits:
                step = time.perf_counter()
                passages = [f"{h.chunk.section}\n{h.chunk.text}" if h.chunk.section else h.chunk.text for h in hits]
                for hit, score in zip(hits, self.reranker.score(query, passages)):
                    hit.score = score
                hits.sort(key=lambda h: h.score, reverse=True)
                timings["rerank"] = (time.perf_counter() - step) * 1000
                if self.min_rerank_score is not None:
                    hits = [h for h in hits if h.score >= self.min_rerank_score]
                    if not hits:
                        message = "No passage was relevant enough to the question."

        hits = dedupe(hits)[:top_k]
        timings["total"] = (time.perf_counter() - started) * 1000
        return SearchResult(hits, pool_ids, timings, message)
