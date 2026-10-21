import argparse
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from chatbot.rag.chunking import build_chunks, embedding_text
from chatbot.rag.embedder import Embedder
from chatbot.rag.errors import RagError, ScannedPdfError
from chatbot.rag.parsing import parse_pdf
from chatbot.rag.store import RagStore
from chatbot.rag.vector import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    name: str
    status: str
    pages: int = 0
    chunks: int = 0
    empty_pages: int = 0
    avg_chunk_tokens: float = 0.0
    parse_seconds: float = 0.0
    embed_seconds: float = 0.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdf(
    path: Path,
    name: str,
    store: RagStore,
    index: VectorIndex,
    embedder: Embedder,
    chunk_target: int = 350,
    chunk_max: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 64,
    max_empty_share: float = 0.2,
    progress=print,
) -> IngestReport:
    path = Path(path)
    store.ensure_embedding_model(embedder.name)
    sha = file_sha256(path)
    doc_id = sha[:12]

    existing = store.get_document(name)
    by_id = store.get_document_by_id(doc_id)
    if by_id is not None and by_id["name"].lower() != name.lower():
        raise RagError(f"This file is already ingested as '{by_id['name']}'.")

    report = IngestReport(name=name, status="ready")
    if existing is not None and existing["sha256"] == sha:
        if existing["status"] == "ready":
            report.status = "already ingested"
            return report
        progress(f"Resuming '{name}': {store.count_pending(doc_id)} chunks left to embed")
    else:
        if existing is not None:
            progress(f"'{name}' changed on disk, replacing the old version")
            index.delete_document(existing["doc_id"])
            store.delete_document(existing["doc_id"])

        started = time.perf_counter()
        parsed = parse_pdf(path)
        if len(parsed.empty_pages) > max_empty_share * len(parsed.pages):
            raise ScannedPdfError(
                f"{len(parsed.empty_pages)} of {len(parsed.pages)} pages have no text layer. "
                "Run OCR on the PDF first (for example with ocrmypdf) and ingest the result."
            )
        chunks = build_chunks(doc_id, parsed.pages, chunk_target, chunk_max, chunk_overlap)
        store.create_document(
            doc_id, name, str(path), sha, [(p.number, p.text) for p in parsed.pages], chunks
        )
        report.parse_seconds = time.perf_counter() - started
        report.pages = len(parsed.pages)
        report.chunks = len(chunks)
        report.empty_pages = len(parsed.empty_pages)
        report.avg_chunk_tokens = sum(c.n_tokens for c in chunks) / max(1, len(chunks))
        progress(
            f"Parsed {report.pages} pages into {report.chunks} chunks "
            f"(avg {report.avg_chunk_tokens:.0f} tokens) in {report.parse_seconds:.1f}s"
        )

    started = time.perf_counter()
    total = store.count_pending(doc_id)
    done = 0
    while True:
        batch = store.pending_chunks(doc_id, batch_size)
        if not batch:
            break
        vectors = embedder.embed_documents([embedding_text(name, c) for c in batch])
        index.upsert(batch, vectors)
        # Marked only after the vectors are saved, so a crash re-embeds at most one batch
        store.mark_embedded([c.id for c in batch])
        done += len(batch)
        progress(f"Embedded {done}/{total} chunks")

    store.set_status(doc_id, "ready")
    report.embed_seconds = time.perf_counter() - started
    return report


def main(argv: list[str] | None = None) -> None:
    from chatbot.config import get_settings
    from chatbot.rag.factory import create_embedder, create_index, create_store

    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG index")
    parser.add_argument("pdf", nargs="?", type=Path, help="Path to the PDF file")
    parser.add_argument("--name", help="Document name used in answers (default: file name)")
    parser.add_argument("--list", action="store_true", help="List ingested documents")
    parser.add_argument("--delete", metavar="NAME", help="Remove a document from the index")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    store = create_store(settings)

    if args.list:
        for row in store.list_documents(ready_only=False):
            print(f"{row['name']}: {row['n_pages']} pages, {row['n_chunks']} chunks, {row['status']}")
        return

    index = create_index(settings)
    if args.delete:
        row = store.get_document(args.delete)
        if row is None:
            raise SystemExit(f"No document named '{args.delete}'")
        index.delete_document(row["doc_id"])
        store.delete_document(row["doc_id"])
        print(f"Deleted '{args.delete}'")
        return

    if args.pdf is None:
        parser.error("a PDF path is required")

    report = ingest_pdf(
        args.pdf,
        args.name or args.pdf.stem,
        store,
        index,
        create_embedder(settings),
        chunk_target=settings.rag_chunk_tokens,
        chunk_max=settings.rag_chunk_max_tokens,
        chunk_overlap=settings.rag_chunk_overlap_tokens,
    )
    print(
        f"{report.name}: {report.status}. "
        f"pages={report.pages} chunks={report.chunks} empty_pages={report.empty_pages} "
        f"parse={report.parse_seconds:.1f}s embed={report.embed_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
