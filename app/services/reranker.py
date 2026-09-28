from typing import Any
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Two-Stage Retrieval Re-ranker using a Transformer Cross-Encoder.

    Why Re-ranking?
    1. Bi-encoders (e.g. all-MiniLM-L6-v2 in vector search) independently compress the query and document into fixed vectors.
       Because they don't see each other during embedding, bi-encoders miss nuanced word-level interactions.
    2. Cross-Encoders pass the (Query, Document) pair simultaneously through full transformer attention.
       Every word in the query directly attends to every word in the document chunk.
    3. Architecture Pattern (Two-Stage Retrieval):
       - Stage 1 (Candidate Generation): Hybrid Search (Dense + BM25) retrieves top 20 candidates in <50ms.
       - Stage 2 (Precision Re-ranking): Cross-Encoder re-scores the top 20 candidates and picks the true top 5.
       - This gives the speed of vector search + the precision of deep transformer cross-attention.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[Any],
        top_k: int = 5,
    ) -> list[tuple[Any, float]]:
        """
        Re-ranks a list of candidate chunks against the user query.
        
        Args:
            query: The user search query.
            candidates: List of chunk objects OR list of (chunk, score) tuples.
            top_k: Number of highest-relevance chunks to return.
            
        Returns:
            List of (chunk, cross_encoder_score) sorted descending by relevance.
        """
        if not candidates:
            return []

        # Normalize candidates if passed as (chunk, score) tuples
        chunks = [
            item[0] if isinstance(item, tuple) else item
            for item in candidates
        ]

        # Form (query, document_text) pairs for the cross-encoder
        pairs = [
            (query, getattr(c, "content", "") or str(c))
            for c in chunks
        ]

        # Compute cross-attention scores
        scores = self.model.predict(pairs)

        # Pair each chunk with its new re-ranked score
        scored_results = [
            (chunk, float(score))
            for chunk, score in zip(chunks, scores)
        ]

        # Sort descending by cross-encoder score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
