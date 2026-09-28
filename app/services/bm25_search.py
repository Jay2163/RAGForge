import re
from typing import Any
from rank_bm25 import BM25Okapi


class BM25SearchService:
    """
    Production-grade Lexical BM25 (Best Matching 25) Search Service.

    Why BM25 is superior to simple full-text search:
    1. Term Saturation (k1=1.5): As a keyword appears multiple times, its score increases asymptotically, preventing keyword stuffing from breaking search.
    2. Document Length Normalization (b=0.75): Penalizes overly long documents and rewards concise code snippets containing exact keywords.
    3. Inverse Document Frequency (IDF): Assigns higher importance to rare, highly specific technical terms (e.g. 'CORSMiddleware', 'OAuth2PasswordBearer').
    """

    def __init__(self, chunks: list[Any] | None = None):
        self.chunks: list[Any] = []
        self.corpus_tokens: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

        if chunks:
            self.index_chunks(chunks)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Tokenizer for technical documentation:
        - Extracts alphanumeric tokens and identifiers (e.g. 'api_key', 'status_code').
        - Lowercases for case-insensitive matching.
        """
        if not text:
            return []
        # Match words and underscores for code identifiers
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def index_chunks(self, chunks: list[Any]) -> None:
        """
        Build the BM25 inverted index from a list of DocumentChunk models or dicts.
        """
        self.chunks = list(chunks)
        self.corpus_tokens = [
            self.tokenize(getattr(c, "content", "") or str(c))
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Any, float]]:
        """
        Query the BM25 index and return the top-K highest scoring chunks.
        Returns: list of (chunk_obj, bm25_score) tuples.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Get raw BM25 scores across all chunks
        scores = self.bm25.get_scores(query_tokens)

        # Pair each chunk with its score and sort descending
        scored_pairs = [
            (chunk, float(score))
            for chunk, score in zip(self.chunks, scores)
            if score > 0.0  # Only return chunks with at least 1 matching term
        ]

        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]
