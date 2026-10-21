from chatbot.config import Settings
from chatbot.rag.embedder import (
    CrossEncoderReranker,
    FlashRankReranker,
    Reranker,
    SentenceTransformerEmbedder,
)
from chatbot.rag.retriever import HybridRetriever
from chatbot.rag.store import RagStore
from chatbot.rag.vector import VectorIndex

DEFAULT_RERANKER_MODELS = {
    "flashrank": "ms-marco-MiniLM-L-12-v2",
    "cross_encoder": "BAAI/bge-reranker-base",
}
# FlashRank also lists a 4 GB LLM based ranker that returns no scores, which this pipeline cannot use
UNSUPPORTED_FLASHRANK_MODELS = {"rank_zephyr_7b_v1_full"}


def create_store(settings: Settings) -> RagStore:
    return RagStore(settings.rag_dir / "rag.db")


def create_index(settings: Settings) -> VectorIndex:
    return VectorIndex(settings.rag_dir / "chroma")


def create_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(settings.rag_embedding_model, settings.rag_query_prefix)


def create_reranker(settings: Settings) -> Reranker | None:
    backend = settings.rag_reranker_backend
    if backend == "none":
        return None

    model = settings.rag_reranker_model or DEFAULT_RERANKER_MODELS[backend]
    if backend == "flashrank":
        if model in UNSUPPORTED_FLASHRANK_MODELS:
            raise ValueError(f"FlashRank model '{model}' is not supported, pick a pairwise model.")
        return FlashRankReranker(model, settings.rag_dir / "models")
    return CrossEncoderReranker(model)


def create_retriever(settings: Settings) -> HybridRetriever:
    store = create_store(settings)
    embedder = create_embedder(settings)
    store.ensure_embedding_model(embedder.name)
    return HybridRetriever(
        store=store,
        index=create_index(settings),
        embedder=embedder,
        reranker=create_reranker(settings),
        candidates=settings.rag_candidates,
        rerank_pool=settings.rag_rerank_pool,
        min_rerank_score=settings.rag_min_rerank_score,
    )
