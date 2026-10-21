import argparse
import asyncio
import json
import re
from pathlib import Path

from chatbot.rag.models import Chunk
from chatbot.rag.store import RagStore
from chatbot.rag.textutil import normalize_space

PROMPT = """Read the passage and write ONE question that it answers.
Rules:
- Use your own wording. Do not copy phrases from the passage into the question.
- The question must make sense without seeing the passage.
- Also copy an exact phrase of 3 to 10 words from the passage that contains the answer.
Reply with JSON only: {{"question": "...", "snippet": "..."}}

Passage:
{passage}"""


def sample_chunks(chunks: list[Chunk], count: int, min_tokens: int = 80) -> list[Chunk]:
    """Pick chunks evenly across the document so every part of it is tested."""
    usable = [c for c in chunks if c.n_tokens >= min_tokens]
    if len(usable) <= count:
        return usable
    step = len(usable) / count
    return [usable[int(i * step)] for i in range(count)]


def parse_reply(reply: str, chunk: Chunk) -> dict | None:
    """Parse the LLM reply and keep it only if the snippet really occurs in the chunk."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        question, snippet = str(data["question"]).strip(), str(data["snippet"]).strip()
    except (ValueError, KeyError):
        return None
    if not question or normalize_space(snippet) not in normalize_space(chunk.text):
        return None
    return {"question": question, "pages": [chunk.page_start, chunk.page_end], "snippets": [snippet]}


async def generate(llm, chunks: list[Chunk], document: str, delay: float) -> list[dict]:
    items: list[dict] = []
    for chunk in chunks:
        reply = await llm.ainvoke(PROMPT.format(passage=chunk.text))
        item = parse_reply(str(reply.content), chunk)
        if item:
            item["document"] = document
            items.append(item)
        # A pause keeps a free tier within its per minute limits
        await asyncio.sleep(delay)
    return items


def main(argv: list[str] | None = None) -> None:
    from chatbot.config import get_settings
    from chatbot.llm import create_llm
    from chatbot.rag.factory import create_store

    parser = argparse.ArgumentParser(description="Draft a golden question set from your documents")
    parser.add_argument("--document", required=True, help="Document name")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between LLM calls")
    parser.add_argument("--out", type=Path, default=Path("golden.json"))
    args = parser.parse_args(argv)

    settings = get_settings()
    store: RagStore = create_store(settings)
    row = store.get_document(args.document)
    if row is None:
        raise SystemExit(f"No document named '{args.document}'")

    with store._connect() as conn:
        rows = conn.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (row["doc_id"],)).fetchall()
    chunks = [
        Chunk(r["id"], r["doc_id"], r["chunk_index"], r["page_start"], r["page_end"], r["section"], r["text"], r["n_tokens"])
        for r in rows
    ]

    items = asyncio.run(generate(create_llm(settings), sample_chunks(chunks, args.count), row["name"], args.delay))
    args.out.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} questions to {args.out}. Review them and delete weak ones before trusting the numbers.")


if __name__ == "__main__":
    main()
