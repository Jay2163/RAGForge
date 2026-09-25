from sqlalchemy.orm import Session

from app.evaluation.dataset import EVALUATION_DATASET
from app.services.vector_search import VectorSearchService


class RetrievalEvaluator:

    def __init__(self, db: Session):
        self.db = db
        self.search_service = VectorSearchService(db)

    def inspect_question(
        self,
        question: str,
        top_k: int = 5,
    ):
        results = self.search_service.search(
            query=question,
            top_k=top_k,
        )

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        for rank, (chunk, similarity) in enumerate(
            results,
            start=1,
        ):
            print(f"\nRank: {rank}")
            print(f"Chunk ID: {chunk.id}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Page: {chunk.page_number}")
            print(f"Section: {chunk.section}")
            print("-" * 60)
            print(chunk.content[:1000])

        return results