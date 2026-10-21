from chatbot.config import get_settings
from chatbot.rag.factory import create_embedder, create_reranker


def main() -> None:
    """Download and load the models once, so the first chat question is not slow."""
    settings = get_settings()
    create_embedder(settings)
    print(f"Embedding model ready: {settings.rag_embedding_model}")
    reranker = create_reranker(settings)
    if reranker is None:
        print("Reranking is off")
    else:
        print(f"Reranker ready: {settings.rag_reranker_backend}")


if __name__ == "__main__":
    main()
