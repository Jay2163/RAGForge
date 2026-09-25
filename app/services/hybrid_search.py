from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.services.vector_search import VectorSearchService
from app.services.lexical_search import LexicalSearchService


@dataclass
class HybridSearchResult:
    chunk: object
    rrf_score: float
    vector_rank: int | None
    lexical_rank: int | None


class HybridSearchService:
    RRF_K = 60

    def __init__(self, db: Session):
        self.db = db

        self.vector_search = VectorSearchService(db)
        self.lexical_search = LexicalSearchService(db)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[HybridSearchResult]:

        # Retrieve independently from both systems
        vector_results = self.vector_search.search(
            query=query,
            top_k=top_k,
        )

        lexical_results = self.lexical_search.search(
            query=query,
            top_k=top_k,
        )

        # chunk_id -> information used for RRF
        combined = {}

        # Vector ranking
        for rank, (chunk, similarity) in enumerate(
            vector_results,
            start=1,
        ):
            combined[chunk.id] = {
                "chunk": chunk,
                "rrf_score": 1 / (self.RRF_K + rank),
                "vector_rank": rank,
                "lexical_rank": None,
            }

        # Lexical ranking
        for rank, (chunk, lexical_score) in enumerate(
            lexical_results,
            start=1,
        ):
            if chunk.id not in combined:
                combined[chunk.id] = {
                    "chunk": chunk,
                    "rrf_score": 0,
                    "vector_rank": None,
                    "lexical_rank": rank,
                }

            combined[chunk.id]["rrf_score"] += (
                1 / (self.RRF_K + rank)
            )

            combined[chunk.id]["lexical_rank"] = rank

        # Sort by combined RRF score
        ranked_results = sorted(
            combined.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return [
            HybridSearchResult(
                chunk=item["chunk"],
                rrf_score=item["rrf_score"],
                vector_rank=item["vector_rank"],
                lexical_rank=item["lexical_rank"],
            )
            for item in ranked_results[:top_k]
        ]