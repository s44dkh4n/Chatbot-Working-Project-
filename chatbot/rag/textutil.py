import re

_WORD = re.compile(r"\w+", re.UNICODE)

# Short list on purpose: BM25 already down-weights common words through IDF
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the "
    "this to was were what when where which who why with do does did can could should "
    "would i you we they about into than then there their".split()
)


def estimate_tokens(text: str) -> int:
    # Roughly 4 characters per token for English, good enough for budgeting
    return max(1, len(text) // 4)


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def normalize_space(text: str) -> str:
    return " ".join(text.split()).lower()
