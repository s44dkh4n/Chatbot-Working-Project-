import re

from chatbot.rag.models import Chunk
from chatbot.rag.parsing import ParsedPage
from chatbot.rag.textutil import estimate_tokens

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_long(text: str, max_tokens: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries, then by characters as a last resort."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        if current and estimate_tokens(f"{current} {sentence}") > max_tokens:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)

    final: list[str] = []
    for piece in pieces:
        while estimate_tokens(piece) > max_tokens:
            final.append(piece[: max_tokens * 4])
            piece = piece[max_tokens * 4 :]
        final.append(piece)
    return final


def embedding_text(doc_name: str, chunk: Chunk) -> str:
    """Text that gets embedded: the section path is prepended so a chunk keeps its context."""
    header = f"{doc_name} > {chunk.section}" if chunk.section else doc_name
    return f"{header}\n{chunk.text}"


class _Builder:
    def __init__(self, doc_id: str, target: int, max_tokens: int, overlap: int) -> None:
        self.doc_id = doc_id
        self.target = target
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.stack: dict[int, str] = {}
        self._reset("")

    def _reset(self, carry: str) -> None:
        self.parts = [carry] if carry else []
        self.carry = carry
        self.pages: list[int] = []
        self.tokens = estimate_tokens(carry) if carry else 0
        self.has_body = False
        self.section = ""

    def _section_path(self) -> str:
        return " > ".join(self.stack[level] for level in sorted(self.stack))

    def _tail(self, text: str) -> str:
        if self.overlap <= 0:
            return ""
        tail = text[-self.overlap * 4 :]
        cut = tail.find(" ")
        return tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail

    def flush(self, overlap: bool) -> None:
        # Content made only of headings waits for the body text that follows it
        if not self.has_body:
            return
        text = "\n\n".join(self.parts)
        index = len(self.chunks)
        self.chunks.append(
            Chunk(
                id=f"{self.doc_id}:{index:05d}",
                doc_id=self.doc_id,
                chunk_index=index,
                page_start=min(self.pages),
                page_end=max(self.pages),
                section=self.section,
                text=text,
                n_tokens=estimate_tokens(text),
            )
        )
        self._reset(self._tail(text) if overlap else "")

    def add_heading(self, text: str, level: int, page: int) -> None:
        if not self.has_body and self.carry:
            # An overlap tail must not leak into a new section
            self.parts, self.tokens, self.carry = [], 0, ""
        if self.has_body and (level <= 2 or self.tokens >= self.target // 2):
            self.flush(overlap=False)
        self.stack = {lvl: title for lvl, title in self.stack.items() if lvl < level}
        self.stack[level] = text
        self.parts.append(text)
        self.pages.append(page)

    def add_paragraph(self, text: str, page: int) -> None:
        pieces = split_long(text, self.max_tokens) if estimate_tokens(text) > self.max_tokens else [text]
        for piece in pieces:
            tokens = estimate_tokens(piece)
            if self.has_body and self.tokens + tokens > self.max_tokens:
                self.flush(overlap=True)
            if not self.has_body:
                self.section = self._section_path()
            self.parts.append(piece)
            self.pages.append(page)
            self.tokens += tokens
            self.has_body = True
            if self.tokens >= self.target:
                self.flush(overlap=True)


def build_chunks(
    doc_id: str,
    pages: list[ParsedPage],
    target_tokens: int = 350,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    builder = _Builder(doc_id, target_tokens, max_tokens, overlap_tokens)
    for page in pages:
        for block in page.blocks:
            if block.heading_level is not None:
                builder.add_heading(block.text, block.heading_level, page.number)
            else:
                builder.add_paragraph(block.text, page.number)
    builder.flush(overlap=False)
    return builder.chunks
