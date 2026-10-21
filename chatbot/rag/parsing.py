import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Lines whose top edge sits in the outer 8 percent of the page are header or footer candidates
MARGIN = 0.08
HEADING_SIZE_RATIO = 1.12
MAX_HEADING_SHARE = 0.15


@dataclass
class Block:
    text: str
    heading_level: int | None = None


@dataclass
class ParsedPage:
    number: int
    blocks: list[Block]

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)


@dataclass
class ParseResult:
    pages: list[ParsedPage]
    empty_pages: list[int]
    body_size: float
    heading_sizes: list[float]


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    y: float
    block_no: int
    block_lines: int


def _key(text: str) -> str:
    # Digits are masked so "Page 12" and "Page 13" count as the same repeated line
    return re.sub(r"\d+", "#", text.lower()).strip()


def _in_margin(line: _Line) -> bool:
    return line.y < MARGIN or line.y > 1 - MARGIN


def _extract_lines(page: pymupdf.Page) -> list[_Line]:
    height = page.rect.height or 1.0
    lines: list[_Line] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        block_lines = [
            line for line in block["lines"] if any(s["text"].strip() for s in line["spans"])
        ]
        for line in block_lines:
            spans = [s for s in line["spans"] if s["text"].strip()]
            text = unicodedata.normalize("NFKC", "".join(s["text"] for s in line["spans"])).strip()
            lines.append(
                _Line(
                    text=text,
                    size=round(max(s["size"] for s in spans) * 2) / 2,
                    bold=all(bool(s["flags"] & 16) or "bold" in s["font"].lower() for s in spans),
                    y=line["bbox"][1] / height,
                    block_no=block["number"],
                    block_lines=len(block_lines),
                )
            )
    return lines


def _find_boilerplate(pages_lines: list[list[_Line]]) -> set[str]:
    if len(pages_lines) < 5:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        counts.update({_key(line.text) for line in lines if _in_margin(line)})
    threshold = max(3, int(0.3 * len(pages_lines)))
    return {key for key, count in counts.items() if count >= threshold}


def _body_size(pages_lines: list[list[_Line]], boilerplate: set[str]) -> float:
    weights: Counter[float] = Counter()
    for lines in pages_lines:
        for line in lines:
            if not (_in_margin(line) and _key(line.text) in boilerplate):
                weights[line.size] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def _is_candidate(line: _Line, body_size: float) -> bool:
    text = line.text
    if not 2 <= len(text) <= 120 or not any(ch.isalpha() for ch in text):
        return False
    if line.size >= body_size * HEADING_SIZE_RATIO:
        return True
    # Short bold lines at body size are treated as low level headings
    return (
        line.bold
        and line.block_lines <= 2
        and len(text) <= 80
        and not text.endswith((".", ",", ";", ":"))
    )


def _join_lines(lines: list[str]) -> str:
    out = lines[0]
    for nxt in lines[1:]:
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and nxt[:1].islower():
            out = out[:-1] + nxt
        else:
            out = f"{out} {nxt}"
    return " ".join(out.split())


def parse_pdf(path: Path, min_chars: int = 20) -> ParseResult:
    with pymupdf.open(path) as document:
        pages_lines = [_extract_lines(page) for page in document]

    boilerplate = _find_boilerplate(pages_lines)
    body_size = _body_size(pages_lines, boilerplate)

    candidates = [
        line
        for lines in pages_lines
        for line in lines
        if not (_in_margin(line) and _key(line.text) in boilerplate)
        and _is_candidate(line, body_size)
    ]
    total_lines = sum(len(lines) for lines in pages_lines) or 1
    heading_sizes: list[float] = []
    if len(candidates) / total_lines <= MAX_HEADING_SHARE:
        heading_sizes = sorted(
            {c.size for c in candidates if c.size >= body_size * HEADING_SIZE_RATIO},
            reverse=True,
        )

    def level_of(line: _Line) -> int | None:
        # Too many heading candidates means the layout is unusual, so headings are switched off
        if not heading_sizes or not _is_candidate(line, body_size):
            return None
        if line.size in heading_sizes:
            return min(heading_sizes.index(line.size) + 1, 3)
        return 4

    pages: list[ParsedPage] = []
    for number, lines in enumerate(pages_lines, start=1):
        pages.append(ParsedPage(number, _build_blocks(lines, boilerplate, level_of)))

    empty = [p.number for p in pages if len(p.text) < min_chars]
    return ParseResult(pages, empty, body_size, heading_sizes)


def _build_blocks(lines: list[_Line], boilerplate: set[str], level_of) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []
    paragraph_block: int | None = None
    heading_block: int | None = None

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(Block(_join_lines(paragraph)))
            paragraph = []

    for line in lines:
        if _in_margin(line) and _key(line.text) in boilerplate:
            continue

        level = level_of(line)
        if level is not None:
            flush()
            if blocks and blocks[-1].heading_level == level and heading_block == line.block_no:
                # A heading that wraps onto a second line stays one heading
                blocks[-1] = Block(f"{blocks[-1].text} {line.text}", level)
            else:
                blocks.append(Block(line.text, level))
            heading_block = line.block_no
            paragraph_block = None
            continue

        heading_block = None
        if paragraph and line.block_no != paragraph_block:
            flush()
        paragraph_block = line.block_no
        paragraph.append(line.text)

    flush()
    return blocks
