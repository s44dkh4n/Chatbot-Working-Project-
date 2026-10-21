from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    chunk_index: int
    page_start: int
    page_end: int
    section: str
    text: str
    n_tokens: int


@dataclass
class Hit:
    chunk: Chunk
    document: str
    score: float


@dataclass
class SearchResult:
    hits: list[Hit]
    # Candidate ids before reranking, used by the evaluation script
    pool_ids: list[str]
    timings_ms: dict[str, float]
    message: str | None = None
