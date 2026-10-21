import asyncio
import json
import logging
import re
import threading
from typing import Annotated

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from chatbot.config import Settings
from chatbot.rag.models import SearchResult
from chatbot.rag.store import RagStore
from chatbot.rag.textutil import STOPWORDS, words

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    query: str = Field(description="Self-contained search query with the key terms, names and numbers")
    document: str | None = Field(default=None, description="Optional document name to search in")
    start_page: Annotated[int, Field(ge=1)] | None = Field(default=None, description="Optional first page")
    end_page: Annotated[int, Field(ge=1)] | None = Field(default=None, description="Optional last page")
    top_k: Annotated[int, Field(ge=1, le=8)] = Field(default=5, description="Number of passages")


class PageInput(BaseModel):
    document: str = Field(description="Document name")
    page: Annotated[int, Field(ge=1)] = Field(description="Page number, starting at 1")


_SENTENCES = re.compile(r"(?<=[.!?])\s+")
MIN_TEXT_CHARS = 200


def _cut(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " ...", True


def focus_text(text: str, query: str, max_chars: int) -> tuple[str, bool]:
    """Shrink a passage by keeping the sentences that share the most words with the query."""
    if len(text) <= max_chars:
        return text, False

    sentences = _SENTENCES.split(text)
    wanted = {w for w in words(query) if w not in STOPWORDS}
    by_relevance = sorted(
        range(len(sentences)), key=lambda i: (-len(wanted & set(words(sentences[i]))), i)
    )

    chosen: list[int] = []
    used = 0
    for i in by_relevance:
        if used + len(sentences[i]) + 5 <= max_chars:
            chosen.append(i)
            used += len(sentences[i]) + 5
    if not chosen:
        return _cut(text, max_chars)
    return " ... ".join(sentences[i] for i in sorted(chosen)), True


def _pages(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def format_search(result: SearchResult, max_tokens: int, query: str = "") -> dict:
    """Build the tool output and keep its total size inside the token budget."""
    if not result.hits:
        return {
            "results": [],
            "message": result.message or "No relevant passages were found in the indexed documents.",
        }

    def meta(rank: int, hit) -> dict:
        return {
            "rank": rank,
            "document": hit.document,
            "pages": _pages(hit.chunk.page_start, hit.chunk.page_end),
            "section": hit.chunk.section,
        }

    budget = max_tokens * 4
    hits = list(result.hits)
    while True:
        # The metadata of each hit also costs tokens, so it is counted before text is shared out
        overhead = sum(len(json.dumps(meta(r, h))) + 40 for r, h in enumerate(hits, start=1))
        per_hit = (budget - overhead - 120) // len(hits)
        if per_hit >= MIN_TEXT_CHARS or len(hits) == 1:
            break
        # Drop the weakest hit instead of showing uselessly short fragments
        hits.pop()
    per_hit = max(per_hit, 80)

    items = []
    for rank, hit in enumerate(hits, start=1):
        text, truncated = focus_text(hit.chunk.text, query, per_hit)
        items.append({**meta(rank, hit), "text": text, "truncated": truncated})

    output: dict = {"results": items}
    if any(item["truncated"] for item in items):
        output["hint"] = "Use get_page to read the full page when a passage is cut off."
    return output


def format_page(document: str, page: int, text: str, max_tokens: int) -> dict:
    body, truncated = _cut(text, max_tokens * 4)
    return {"document": document, "page": page, "text": body, "truncated": truncated}


def build_rag_tools(settings: Settings) -> list[BaseTool]:
    """Return the document tools, or nothing when no document has been ingested yet."""
    db_path = settings.rag_dir / "rag.db"
    if not db_path.exists():
        return []

    store = RagStore(db_path)
    documents = store.list_documents(ready_only=True)
    if not documents:
        return []

    names = ", ".join(f"'{row['name']}'" for row in documents)
    state: dict = {"retriever": None}
    lock = threading.Lock()

    def get_retriever():
        # Models are loaded on first use so the chatbot starts quickly
        with lock:
            if state["retriever"] is None:
                from chatbot.rag.factory import create_retriever

                state["retriever"] = create_retriever(settings)
            return state["retriever"]

    def run_search(args: dict) -> dict:
        result = get_retriever().search(**args)
        logger.info("search_documents timings_ms=%s", {k: round(v) for k, v in result.timings_ms.items()})
        return format_search(result, settings.rag_max_tool_tokens, args["query"])

    def run_page(document: str, page: int) -> dict:
        doc_ids, available = store.resolve_documents(document)
        if not doc_ids:
            return {"error": f"Unknown document '{document}'. Available: {', '.join(available)}."}
        row = store.get_document_by_id(doc_ids[0])
        text = store.get_page(doc_ids[0], page)
        if text is None:
            return {"error": f"Page {page} is out of range. '{row['name']}' has {row['n_pages']} pages."}
        return format_page(row["name"], page, text, settings.rag_max_tool_tokens)

    async def search_documents(
        query: str,
        document: str | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        top_k: int = settings.rag_top_k,
    ) -> dict:
        args = {
            "query": query,
            "document": document,
            "start_page": start_page,
            "end_page": end_page,
            "top_k": top_k,
        }
        return await asyncio.to_thread(run_search, args)

    async def get_page(document: str, page: int) -> dict:
        return await asyncio.to_thread(run_page, document, page)

    return [
        StructuredTool.from_function(
            coroutine=search_documents,
            name="search_documents",
            description=(
                f"Search the indexed documents ({names}) and return the most relevant passages "
                "with page numbers. Use for any question about their content."
            ),
            args_schema=SearchInput,
        ),
        StructuredTool.from_function(
            coroutine=get_page,
            name="get_page",
            description="Read the full text of one page of an indexed document.",
            args_schema=PageInput,
        ),
    ]
