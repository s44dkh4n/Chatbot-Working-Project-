import asyncio
import json
import math

import pymupdf
import pytest

from chatbot.config import Settings
from chatbot.rag.chunking import build_chunks
from chatbot.rag.errors import EmbeddingModelMismatch, ScannedPdfError
from chatbot.rag.evaluate import evaluate, format_report
from chatbot.rag.ingest import ingest_pdf
from chatbot.rag.make_golden import parse_reply, sample_chunks
from chatbot.rag.models import Chunk, Hit, SearchResult
from chatbot.rag.parsing import parse_pdf
from chatbot.rag.retriever import MODES, HybridRetriever, dedupe, rrf
from chatbot.rag.store import RagStore, fts_query
from chatbot.rag.vector import VectorIndex
from chatbot.tools.rag_tools import build_rag_tools, format_page, format_search
from tests.rag_helpers import FACTS, QUESTIONS, FakeEmbedder, FakeReranker, build_pdf

NAME = "ACME Handbook"


def make_env(root, embedder=None):
    store = RagStore(root / "rag.db")
    index = VectorIndex(root / "chroma")
    embedder = embedder or FakeEmbedder()
    retriever = HybridRetriever(store, index, embedder, FakeReranker(), candidates=30, rerank_pool=20)
    return store, index, embedder, retriever


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory):
    return build_pdf(tmp_path_factory.mktemp("pdf") / "handbook.pdf", 40, FACTS)


@pytest.fixture(scope="module")
def env(tmp_path_factory, pdf_path):
    root = tmp_path_factory.mktemp("rag")
    store, index, embedder, retriever = make_env(root)
    report = ingest_pdf(pdf_path, NAME, store, index, embedder, progress=lambda _: None)
    return store, index, embedder, retriever, report, root


def test_parse_removes_boilerplate_and_finds_headings(pdf_path):
    parsed = parse_pdf(pdf_path)
    assert len(parsed.pages) == 40 and not parsed.empty_pages
    all_text = " ".join(p.text for p in parsed.pages)
    assert "ACME Corp Handbook" not in all_text
    assert "Page 3 of" not in all_text
    levels = {b.text: b.heading_level for p in parsed.pages for b in p.blocks if b.heading_level}
    assert levels["Chapter 1 Overview"] == 1
    assert levels["Section 1.5 Details"] == 2


def test_dehyphenation(tmp_path):
    document = pymupdf.open()
    for _ in range(6):
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(50, 100, 545, 200), "The inter-\nnational standard applies to all sites.", fontsize=11)
    document.save(tmp_path / "hyphen.pdf")
    text = parse_pdf(tmp_path / "hyphen.pdf").pages[0].text
    assert "international standard" in text


def test_chunks_have_sections_pages_and_sane_sizes(pdf_path):
    parsed = parse_pdf(pdf_path)
    chunks = build_chunks("doc1", parsed.pages)
    assert chunks and all(c.n_tokens > 30 for c in chunks)
    assert max(c.n_tokens for c in chunks) <= 560
    assert all(c.page_start <= c.page_end for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    on_page_7 = [c for c in chunks if c.page_start <= 7 <= c.page_end and "2555" in c.text]
    assert on_page_7 and on_page_7[0].section.startswith("Chapter 1")


def test_ingest_report_and_idempotence(env, pdf_path):
    store, index, embedder, _, report, _ = env
    assert report.pages == 40 and report.chunks > 40 and report.status == "ready"
    assert index.count() == report.chunks
    assert [r["status"] for r in store.list_documents()] == ["ready"]
    calls = embedder.document_calls
    again = ingest_pdf(pdf_path, NAME, store, index, embedder, progress=lambda _: None)
    assert again.status == "already ingested" and embedder.document_calls == calls


def test_resume_after_crash_only_embeds_the_rest(tmp_path, pdf_path):
    embedder = FakeEmbedder()
    embedder.fail_on_call = 2
    store, index, _, retriever = make_env(tmp_path, embedder)
    with pytest.raises(RuntimeError):
        ingest_pdf(pdf_path, NAME, store, index, embedder, batch_size=8, progress=lambda _: None)

    doc = store.get_document(NAME)
    pending = store.count_pending(doc["doc_id"])
    assert doc["status"] == "ingesting" and 0 < pending < doc["n_chunks"]
    assert retriever.search("audit logs").hits == []

    embedder.fail_on_call = None
    embedder.document_calls = 0
    ingest_pdf(pdf_path, NAME, store, index, embedder, batch_size=8, progress=lambda _: None)
    assert embedder.document_calls == math.ceil(pending / 8)
    assert index.count() == doc["n_chunks"]
    assert store.get_document(NAME)["status"] == "ready"
    assert retriever.search("audit logs").hits


def test_embedding_model_mismatch_is_blocked(env):
    store = env[0]
    with pytest.raises(EmbeddingModelMismatch):
        store.ensure_embedding_model("some-other-model")


@pytest.mark.parametrize("mode", MODES)
def test_planted_facts_are_found_in_every_mode(env, mode):
    retriever = env[3]
    for question, page, snippet in QUESTIONS:
        result = retriever.search(question, top_k=5, mode=mode)
        if mode == "vector":
            # The hash based fake embedder has no word weighting, so only the plumbing is checked here
            assert result.hits
            continue
        assert any(h.chunk.page_start <= page <= h.chunk.page_end and snippet in h.chunk.text for h in result.hits), (mode, question)


def test_hybrid_rerank_puts_the_answer_first(env):
    retriever = env[3]
    for question, page, snippet in QUESTIONS:
        top = retriever.search(question, top_k=3).hits[0]
        assert snippet in top.chunk.text and top.document == NAME


def test_filters_and_unknown_document(env):
    retriever = env[3]
    result = retriever.search("audit logs retention", document="acme")
    assert result.hits and result.hits[0].document == NAME
    unknown = retriever.search("audit logs", document="Missing Manual")
    assert not unknown.hits and NAME in unknown.message
    ranged = retriever.search("audit logs retention", start_page=30, end_page=40)
    assert all(h.chunk.page_end >= 30 and h.chunk.page_start <= 40 for h in ranged.hits)
    assert all("2555" not in h.chunk.text for h in ranged.hits)


def test_relevance_threshold_can_abstain(env):
    store, index, embedder, _, _, _ = env
    strict = HybridRetriever(store, index, embedder, FakeReranker(), min_rerank_score=0.99)
    result = strict.search("quantum chromodynamics lagrangian")
    assert not result.hits and result.message


def test_rrf_dedupe_and_fts_query():
    fused = [i for i, _ in rrf([["a", "b", "c"], ["b", "c", "d"]])]
    assert fused[0] == "b" and set(fused) == {"a", "b", "c", "d"}
    base = Chunk("1", "d", 0, 1, 1, "", "the quick brown fox jumps over the lazy dog today", 12)
    twin = Chunk("2", "d", 1, 2, 2, "", "the quick brown fox jumps over the lazy dog today", 12)
    other = Chunk("3", "d", 2, 3, 3, "", "completely different words about invoices and budgets", 12)
    kept = dedupe([Hit(base, "x", 1.0), Hit(twin, "x", 0.9), Hit(other, "x", 0.8)])
    assert [h.chunk.id for h in kept] == ["1", "3"]
    # Search operators typed by a user are quoted into plain terms, "or" is a stopword
    assert fts_query('What is "NEAR" (the) OR* retention?') == '"near" OR "retention"'


def test_scanned_pdf_is_rejected(tmp_path):
    document = pymupdf.open()
    for _ in range(10):
        document.new_page()
    document.save(tmp_path / "scan.pdf")
    store, index, embedder, _ = make_env(tmp_path / "r")
    with pytest.raises(ScannedPdfError):
        ingest_pdf(tmp_path / "scan.pdf", "scan", store, index, embedder, progress=lambda _: None)


def test_tool_output_stays_inside_the_token_budget(env):
    retriever = env[3]
    query = "audit logs retention period"
    result = retriever.search(query, top_k=8)
    output = format_search(result, max_tokens=300, query=query)
    assert len(json.dumps(output)) / 4 <= 300 * 1.15
    assert len(output["results"]) < 8 and "hint" in output
    # Even when shortened, the sentence that answers the question is the one that is kept
    assert "2555" in output["results"][0]["text"] and output["results"][0]["truncated"]
    roomy = format_search(result, max_tokens=4000, query=query)
    assert len(roomy["results"]) == len(result.hits) and not roomy["results"][0]["truncated"]
    assert format_search(SearchResult([], [], {}, None), 300)["results"] == []
    page = format_page(NAME, 7, "word " * 5000, max_tokens=100)
    assert page["truncated"] and len(page["text"]) <= 410


def test_chatbot_tools_end_to_end(env, monkeypatch, tmp_path):
    _, _, _, retriever, _, root = env
    settings = Settings(groq_api_key="x", rag_dir=root, _env_file=None)
    assert build_rag_tools(Settings(groq_api_key="x", rag_dir=tmp_path / "none", _env_file=None)) == []

    monkeypatch.setattr("chatbot.rag.factory.create_retriever", lambda s: retriever)
    search_tool, page_tool = build_rag_tools(settings)
    assert (search_tool.name, page_tool.name) == ("search_documents", "get_page")
    assert NAME in search_tool.description

    found = asyncio.run(search_tool.ainvoke({"query": "How long are audit logs retained?", "top_k": 3}))
    assert "2555" in found["results"][0]["text"] and found["results"][0]["pages"] == "7"
    page = asyncio.run(page_tool.ainvoke({"document": NAME, "page": 7}))
    assert "2555 days" in page["text"]
    assert "out of range" in asyncio.run(page_tool.ainvoke({"document": NAME, "page": 999}))["error"]
    assert "Unknown document" in asyncio.run(page_tool.ainvoke({"document": "nope", "page": 1}))["error"]


def test_evaluation_report_and_golden_helpers(env):
    retriever = env[3]
    items = [{"question": q, "document": NAME, "snippets": [s], "pages": [p]} for q, p, s in QUESTIONS]
    report = evaluate(retriever, items, tuple(MODES), 5)
    assert report["hybrid_rerank"]["hit@1"] == 1.0 and report["hybrid_rerank"]["pool_recall"] == 1.0
    assert "hybrid_rerank" in format_report(report, 5)

    chunk = Chunk("c", "d", 0, 4, 5, "", "Refunds are paid within thirty business days of approval by finance.", 100)
    good = parse_reply('```json\n{"question": "How fast are refunds paid?", "snippet": "within thirty business days"}\n```', chunk)
    assert good["pages"] == [4, 5] and good["snippets"] == ["within thirty business days"]
    assert parse_reply('{"question": "q", "snippet": "text not in chunk"}', chunk) is None
    assert parse_reply("no json here", chunk) is None
    many = [Chunk(str(i), "d", i, i, i, "", "x", 100) for i in range(100)]
    assert len(sample_chunks(many, 10)) == 10


def test_blank_env_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("CALCULATOR_SERVER_PATH", "")
    monkeypatch.setenv("EXPENSE_TRACKER_URL", "")
    settings = Settings(_env_file=None)
    assert settings.calculator_server_path is None and settings.expense_tracker_url is None


def test_flashrank_adapter_restores_input_order(monkeypatch, tmp_path):
    import sys
    import types

    class FakeRanker:
        def __init__(self, model_name, cache_dir, max_length, log_level):
            self.args = (model_name, cache_dir, max_length, log_level)

        def rerank(self, request):
            # Like the real library: scores added, list sorted best first
            for passage in request.passages:
                passage["score"] = len(passage["text"])
            return sorted(request.passages, key=lambda p: p["score"], reverse=True)

    class FakeRequest:
        def __init__(self, query, passages):
            self.query, self.passages = query, passages

    monkeypatch.setitem(sys.modules, "flashrank", types.SimpleNamespace(Ranker=FakeRanker, RerankRequest=FakeRequest))
    from chatbot.rag.embedder import FlashRankReranker

    reranker = FlashRankReranker("some-model", tmp_path / "models")
    assert reranker.score("q", ["aa", "bbbbbb", "c"]) == [2.0, 6.0, 1.0]
    assert reranker.score("q", []) == []
    assert reranker._ranker.args[2:] == (512, "WARNING") and (tmp_path / "models").is_dir()


def test_flashrank_real_api_contract():
    flashrank = pytest.importorskip("flashrank")
    import inspect

    inspect.signature(flashrank.Ranker.__init__).bind(
        None, model_name="m", cache_dir="c", max_length=512, log_level="WARNING"
    )
    request = flashrank.RerankRequest(query="q", passages=[{"id": 0, "text": "t"}])
    assert request.passages[0]["id"] == 0


def test_reranker_factory_defaults_and_guards(monkeypatch, tmp_path):
    from chatbot.rag import factory

    made = {}
    monkeypatch.setattr(factory, "FlashRankReranker", lambda model, cache: made.update(flash=(model, cache)) or "flash")
    monkeypatch.setattr(factory, "CrossEncoderReranker", lambda model: made.update(cross=model) or "cross")

    base = Settings(groq_api_key="x", rag_dir=tmp_path, _env_file=None)
    assert base.rag_reranker_backend == "flashrank" and base.rag_reranker_model is None
    assert factory.create_reranker(base) == "flash"
    assert made["flash"] == ("ms-marco-MiniLM-L-12-v2", tmp_path / "models")

    other = Settings(groq_api_key="x", rag_dir=tmp_path, rag_reranker_backend="cross_encoder", _env_file=None)
    assert factory.create_reranker(other) == "cross" and made["cross"] == "BAAI/bge-reranker-base"

    off = Settings(groq_api_key="x", rag_dir=tmp_path, rag_reranker_backend="none", _env_file=None)
    assert factory.create_reranker(off) is None

    llm_ranker = Settings(groq_api_key="x", rag_dir=tmp_path, rag_reranker_model="rank_zephyr_7b_v1_full", _env_file=None)
    with pytest.raises(ValueError):
        factory.create_reranker(llm_ranker)
