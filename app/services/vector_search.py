from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk
from app.services.embedding import EmbeddingService


class VectorSearchService:
    def __init__(
        self,
        db: Session,
        embedder: EmbeddingService | None = None,
    ):
        self.db = db
        self.embedder = embedder or EmbeddingService()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:

        # Convert the user's query into an embedding
        query_embedding = self.embedder.embed_text(query)

        # pgvector cosine distance
        distance = DocumentChunk.embedding.cosine_distance(
            query_embedding
        )

        statement = (
            select(
                DocumentChunk,
                distance.label("distance"),
            )
            .where(
                DocumentChunk.embedding.is_not(None)
            )
            .order_by(distance)
            .limit(top_k)
        )

        result = self.db.execute(statement)

        results = []

        for chunk, distance_value in result.all():
            # Cosine similarity = 1 - cosine distance
            similarity = 1 - distance_value

            results.append(
                (chunk, similarity)
            )

        return results