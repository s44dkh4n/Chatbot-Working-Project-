"""Synthetic scale test for the storage and search path.

Run: python -m tests.benchmark_scale --pages 1200
It uses a fake embedder, so it measures parsing, chunking, SQLite, Chroma, and fusion speed.
Embedding and reranking model speed has to be measured with the real models (see README).
"""

import argparse
import random
import tempfile
import time
from pathlib import Path

from chatbot.rag.evaluate import evaluate, format_report
from chatbot.rag.ingest import ingest_pdf
from chatbot.rag.retriever import MODES, HybridRetriever
from chatbot.rag.store import RagStore
from chatbot.rag.vector import VectorIndex
from tests.rag_helpers import FakeEmbedder, FakeReranker, build_pdf

TOPICS = (
    "zeppelin quasar mongoose tamarind obsidian falconer pavilion sextant lacquer trident "
    "gazebo kestrel monsoon bellows catalpa dirigible ember fjord gargoyle hyacinth iguana "
    "jackfruit kaleidoscope lighthouse"
).split()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1200)
    args = parser.parse_args()

    rng = random.Random(3)
    pages = rng.sample(range(5, args.pages), len(TOPICS))
    facts, golden = {}, []
    for topic, page in zip(TOPICS, pages):
        weeks = rng.randint(3, 90)
        facts[page] = f"The maintenance interval for the {topic} unit is {weeks} weeks according to procedure."
        golden.append(
            {"question": f"How often must the {topic} unit be serviced?", "snippets": [f"{weeks} weeks"], "pages": [page]}
        )

    root = Path(tempfile.mkdtemp())
    started = time.perf_counter()
    pdf = build_pdf(root / "big.pdf", args.pages, facts, paragraphs_per_page=5, words_per_paragraph=80)
    print(f"Built a {args.pages} page PDF ({pdf.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - started:.0f}s")

    store, index, embedder = RagStore(root / "rag.db"), VectorIndex(root / "chroma"), FakeEmbedder()
    report = ingest_pdf(pdf, "Big Manual", store, index, embedder, progress=lambda _: None)
    db_mb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and f.name != "big.pdf") / 1e6
    print(
        f"Ingest: pages={report.pages} chunks={report.chunks} avg_tokens={report.avg_chunk_tokens:.0f} "
        f"parse={report.parse_seconds:.1f}s embed+store={report.embed_seconds:.1f}s index_size={db_mb:.0f} MB"
    )

    retriever = HybridRetriever(store, index, embedder, FakeReranker())
    print(format_report(evaluate(retriever, golden, tuple(MODES), 5), 5))


if __name__ == "__main__":
    main()
