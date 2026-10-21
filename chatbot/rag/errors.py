class RagError(Exception):
    """Base class for RAG pipeline errors."""


class ScannedPdfError(RagError):
    """The PDF has too many pages without a text layer and needs OCR first."""


class EmbeddingModelMismatch(RagError):
    """The index was built with a different embedding model than the one configured."""
