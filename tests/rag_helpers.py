import hashlib
import math
import random
from pathlib import Path

import pymupdf

from chatbot.rag.textutil import STOPWORDS, words

VOCAB = (
    "table window garden river market letter ticket bridge number planet signal pattern "
    "forest history engine mirror silver yellow coffee island paper stone summer winter "
    "orange carpet melody harbor canvas lantern ribbon meadow thunder pencil basket "
    "compass velvet anchor feather marble cabin desert valley rocket button saddle "
    "candle glacier pepper blanket whistle cotton dragon puzzle shadow cellar tunnel "
    "harvest morning evening village castle journey library station theory visitor "
    "weather yellowish crystal diamond emerald copper bronze circle square triangle "
    "monday tuesday friday sunday color texture sample recipe season picture frame"
).split()


class FakeEmbedder:
    """Deterministic bag-of-words embedder, so tests need no model download."""

    name = "fake-hash-embedder"
    dim = 256

    def __init__(self) -> None:
        self.document_calls = 0
        self.fail_on_call: int | None = None

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in words(text):
            digest = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vector[digest % self.dim] += 1.0 if (digest >> 8) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        if self.fail_on_call == self.document_calls:
            raise RuntimeError("simulated crash")
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeReranker:
    """Scores a passage by the share of query words it contains."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        wanted = {w for w in words(query) if w not in STOPWORDS}
        scores = []
        for passage in passages:
            present = set(words(passage))
            scores.append(len(wanted & present) / max(1, len(wanted)))
        return scores


def build_pdf(
    path: Path,
    n_pages: int,
    facts: dict[int, str] | None = None,
    seed: int = 7,
    paragraphs_per_page: int = 5,
    words_per_paragraph: int = 60,
) -> Path:
    """Create a PDF with a header, footer, chapter headings, filler text, and planted facts."""
    rng = random.Random(seed)
    facts = facts or {}
    document = pymupdf.open()
    for number in range(1, n_pages + 1):
        page = document.new_page()
        page.insert_text((50, 30), "ACME Corp Handbook", fontsize=9)
        page.insert_text((50, 825), f"Page {number} of {n_pages}", fontsize=9)

        y = 110
        if number % 10 == 1:
            page.insert_text((50, y), f"Chapter {number // 10 + 1} Overview", fontsize=20, fontname="hebo")
            y += 40
        elif number % 10 == 5:
            page.insert_text((50, y), f"Section {number // 10 + 1}.5 Details", fontsize=14, fontname="hebo")
            y += 30

        paragraphs = [
            " ".join(rng.choice(VOCAB) for _ in range(words_per_paragraph)).capitalize() + "."
            for _ in range(paragraphs_per_page)
        ]
        if number in facts:
            paragraphs[2] = facts[number]
        for text in paragraphs:
            page.insert_textbox(pymupdf.Rect(50, y, 545, y + 95), text, fontsize=11)
            y += 100

    document.save(path)
    document.close()
    return path


FACTS = {
    7: "The retention period for audit logs is 2555 days under the compliance policy.",
    19: "Employees must submit expense reports within 14 calendar days after travel.",
    33: "The emergency shutdown code for the cooling system is ORCHID-77.",
}

QUESTIONS = [
    ("How long are audit logs retained?", 7, "2555 days"),
    ("When must employees file their expense reports after a trip?", 19, "14 calendar days"),
    ("What is the shutdown code of the cooling system in an emergency?", 33, "ORCHID-77"),
]
