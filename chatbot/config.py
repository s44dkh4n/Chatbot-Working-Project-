from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration lives here and is read from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    groq_api_key: SecretStr
    llm_model: str = "qwen/qwen3.8-27b"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    llm_max_retries: int = 2
    max_history_tokens: int = 2500

    # Local tools
    alphavantage_api_key: SecretStr | None = None

    # Persistence
    db_path: Path = Path("ChatsHistory/chats.db")

    # MCP: Calculator server (stdio)
    uv_command: str = "uv"
    calculator_server_path: Path | None = None

    # MCP: Expense Tracker server (http)
    expense_tracker_url: str | None = None
    horizon_access_token: SecretStr | None = None

    # MCP: Manim server (stdio)
    manim_python_path: Path | None = None
    manim_server_path: Path | None = None
    manim_executable: Path | None = None

        # RAG: storage and models
    rag_dir: Path = Path("data/rag")
    rag_embedding_model: str = "BAAI/bge-small-en-v1.5"
    rag_query_prefix: str = "Represent this sentence for searching relevant passages: "
    rag_reranker_backend: Literal["flashrank", "cross_encoder", "none"] = "flashrank"
    rag_reranker_model: str | None = None

    # RAG: retrieval
    rag_candidates: int = 30
    rag_rerank_pool: int = 20
    rag_top_k: int = 5
    rag_min_rerank_score: float | None = None
    rag_max_tool_tokens: int = 1200

    # RAG: chunking (token counts are estimates, about 4 characters per token)
    rag_chunk_tokens: int = 350
    rag_chunk_max_tokens: int = 500
    rag_chunk_overlap_tokens: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
