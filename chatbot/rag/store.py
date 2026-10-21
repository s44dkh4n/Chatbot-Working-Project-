import sqlite3
from contextlib import contextmanager
from pathlib import Path

from chatbot.rag.errors import EmbeddingModelMismatch
from chatbot.rag.models import Chunk
from chatbot.rag.textutil import STOPWORDS, words

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    n_pages INTEGER NOT NULL,
    n_chunks INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (doc_id, page)
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    n_tokens INTEGER NOT NULL,
    embedded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_pending ON chunks (doc_id, embedded);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED, doc_id UNINDEXED, section, text, tokenize = 'porter unicode61'
);
"""


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        chunk_index=row["chunk_index"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        section=row["section"],
        text=row["text"],
        n_tokens=row["n_tokens"],
    )


def fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 query: quoted terms joined with OR."""
    terms = [w for w in words(query) if w not in STOPWORDS and len(w) > 1][:32]
    return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))


class RagStore:
    """SQLite store for documents, pages, chunks, keyword search (FTS5), and ingest progress."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            # The inner "with" commits on success and rolls back on error
            with conn:
                yield conn
        finally:
            conn.close()

    def ensure_embedding_model(self, name: str) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
            has_documents = conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone() is not None
            if row is not None and row["value"] != name and has_documents:
                raise EmbeddingModelMismatch(
                    f"The index was built with '{row['value']}' but '{name}' is configured. "
                    "Use the same model, or delete data/rag and ingest again."
                )
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_model', ?)", (name,))

    def get_document(self, name: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM documents WHERE name = ?", (name,)).fetchone()

    def get_document_by_id(self, doc_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()

    def list_documents(self, ready_only: bool = True) -> list[sqlite3.Row]:
        sql = "SELECT * FROM documents"
        if ready_only:
            sql += " WHERE status = 'ready'"
        with self._connect() as conn:
            return conn.execute(sql + " ORDER BY name").fetchall()

    def create_document(
        self,
        doc_id: str,
        name: str,
        path: str,
        sha256: str,
        pages: list[tuple[int, str]],
        chunks: list[Chunk],
    ) -> None:
        # One transaction, so a crash never leaves a document with half of its chunks
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, name, path, sha256, n_pages, n_chunks, status) "
                "VALUES (?, ?, ?, ?, ?, ?, 'ingesting')",
                (doc_id, name, path, sha256, len(pages), len(chunks)),
            )
            conn.executemany(
                "INSERT INTO pages (doc_id, page, text) VALUES (?, ?, ?)",
                [(doc_id, number, text) for number, text in pages],
            )
            conn.executemany(
                "INSERT INTO chunks (id, doc_id, chunk_index, page_start, page_end, section, text, n_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, c.doc_id, c.chunk_index, c.page_start, c.page_end, c.section, c.text, c.n_tokens)
                    for c in chunks
                ],
            )
            conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, doc_id, section, text) VALUES (?, ?, ?, ?)",
                [(c.id, c.doc_id, c.section, c.text) for c in chunks],
            )

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def set_status(self, doc_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE documents SET status = ? WHERE doc_id = ?", (status, doc_id))

    def count_pending(self, doc_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ? AND embedded = 0", (doc_id,)
            ).fetchone()
            return row["n"]

    def pending_chunks(self, doc_id: str, limit: int) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND embedded = 0 ORDER BY chunk_index LIMIT ?",
                (doc_id, limit),
            ).fetchall()
            return [_row_to_chunk(row) for row in rows]

    def mark_embedded(self, chunk_ids: list[str]) -> None:
        with self._connect() as conn:
            conn.executemany("UPDATE chunks SET embedded = 1 WHERE id = ?", [(i,) for i in chunk_ids])

    def get_chunks(self, ids: list[str], ready_only: bool = True) -> dict[str, tuple[Chunk, str]]:
        """Return {chunk_id: (chunk, document name)} for the ids that exist."""
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        sql = (
            "SELECT c.*, d.name AS doc_name FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            f"WHERE c.id IN ({marks})"
        )
        if ready_only:
            sql += " AND d.status = 'ready'"
        with self._connect() as conn:
            rows = conn.execute(sql, ids).fetchall()
            return {row["id"]: (_row_to_chunk(row), row["doc_name"]) for row in rows}

    def get_page(self, doc_id: str, page: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT text FROM pages WHERE doc_id = ? AND page = ?", (doc_id, page)
            ).fetchone()
            return row["text"] if row else None

    def resolve_documents(self, name: str | None) -> tuple[list[str] | None, list[str]]:
        """Map a document name to ids. None means no filter, an empty list means no match."""
        ready = self.list_documents(ready_only=True)
        names = [row["name"] for row in ready]
        if not name:
            return None, names
        wanted = name.strip().lower()
        exact = [row["doc_id"] for row in ready if row["name"].lower() == wanted]
        if exact:
            return exact, names
        return [row["doc_id"] for row in ready if wanted in row["name"].lower()], names

    def bm25_search(
        self,
        query: str,
        limit: int,
        doc_ids: list[str] | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> list[tuple[str, float]]:
        match = fts_query(query)
        if not match:
            return []

        sql = [
            "SELECT chunks_fts.chunk_id AS id, bm25(chunks_fts, 0.0, 0.0, 2.0, 1.0) AS score",
            "FROM chunks_fts",
            "JOIN chunks c ON c.id = chunks_fts.chunk_id",
            "JOIN documents d ON d.doc_id = c.doc_id",
            "WHERE chunks_fts MATCH ? AND d.status = 'ready'",
        ]
        params: list = [match]
        if doc_ids is not None:
            sql.append(f"AND c.doc_id IN ({','.join('?' * len(doc_ids))})")
            params.extend(doc_ids)
        if start_page is not None:
            sql.append("AND c.page_end >= ?")
            params.append(start_page)
        if end_page is not None:
            sql.append("AND c.page_start <= ?")
            params.append(end_page)
        # In FTS5 a lower bm25 value means a better match
        sql.append("ORDER BY score LIMIT ?")
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
            return [(row["id"], row["score"]) for row in rows]
