import argparse
import json
import statistics
from pathlib import Path

from chatbot.rag.models import Chunk
from chatbot.rag.retriever import MODES, HybridRetriever
from chatbot.rag.textutil import normalize_space


def is_relevant(chunk: Chunk, item: dict) -> bool:
    """A chunk is relevant if it holds an answer snippet, or else if it overlaps an expected page."""
    snippets = item.get("snippets") or []
    if snippets:
        text = normalize_space(chunk.text)
        return any(normalize_space(s) in text for s in snippets)
    pages = item.get("pages") or []
    return any(chunk.page_start <= p <= chunk.page_end for p in pages)


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]


def evaluate(
    retriever: HybridRetriever, items: list[dict], modes: tuple[str, ...], k: int = 5
) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for mode in modes:
        ranks: list[int | None] = []
        pool_hits = 0
        latencies: list[float] = []
        failures: list[dict] = []

        for item in items:
            result = retriever.search(
                item["question"], top_k=k, document=item.get("document"), mode=mode
            )
            latencies.append(result.timings_ms.get("total", 0.0))

            rank = next(
                (i for i, hit in enumerate(result.hits, start=1) if is_relevant(hit.chunk, item)), None
            )
            ranks.append(rank)

            pool = retriever.store.get_chunks(result.pool_ids)
            if any(is_relevant(chunk, item) for chunk, _ in pool.values()):
                pool_hits += 1
            if rank is None:
                failures.append(
                    {
                        "question": item["question"],
                        "expected": item.get("snippets") or item.get("pages"),
                        "got_pages": [f"{h.chunk.page_start}-{h.chunk.page_end}" for h in result.hits],
                    }
                )

        n = len(items)
        report[mode] = {
            "hit@1": sum(1 for r in ranks if r == 1) / n,
            "hit@3": sum(1 for r in ranks if r is not None and r <= 3) / n,
            f"hit@{k}": sum(1 for r in ranks if r is not None) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "pool_recall": pool_hits / n,
            "latency_ms_avg": statistics.mean(latencies),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "failures": failures,
        }
    return report


def format_report(report: dict[str, dict], k: int) -> str:
    columns = ["hit@1", "hit@3", f"hit@{k}", "mrr", "pool_recall", "latency_ms_avg", "latency_ms_p95"]
    lines = ["mode".ljust(15) + "".join(c.rjust(16) for c in columns)]
    for mode, stats in report.items():
        cells = "".join(
            (f"{stats[c]:.0f}" if c.startswith("latency") else f"{stats[c]:.3f}").rjust(16) for c in columns
        )
        lines.append(mode.ljust(15) + cells)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from chatbot.config import get_settings
    from chatbot.rag.factory import create_retriever

    parser = argparse.ArgumentParser(description="Measure retrieval quality on a golden question set")
    parser.add_argument("golden", type=Path, help="JSON file made by make_golden or written by hand")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--details", action="store_true", help="Print the questions that failed")
    args = parser.parse_args(argv)

    items = json.loads(args.golden.read_text(encoding="utf-8"))
    report = evaluate(create_retriever(get_settings()), items, tuple(args.modes), args.k)
    print(format_report(report, args.k))
    if args.details:
        for mode, stats in report.items():
            for failure in stats["failures"]:
                print(f"\n[{mode}] missed: {failure['question']}\n  expected {failure['expected']}, got pages {failure['got_pages']}")


if __name__ == "__main__":
    main()
