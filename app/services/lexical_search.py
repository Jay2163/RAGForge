from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


class LexicalSearchService:
    def __init__(self, db: Session):
        self.db = db

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:

        search_query = func.plainto_tsquery(
            "english",
            query,
        )

        rank = func.ts_rank_cd(
            func.to_tsvector(
                "english",
                DocumentChunk.content,
            ),
            search_query,
        )

        statement = (
            select(
                DocumentChunk,
                rank.label("rank"),
            )
            .where(
                DocumentChunk.content.is_not(None),
                func.to_tsvector(
                    "english",
                    DocumentChunk.content,
                ).op("@@")(search_query),
            )
            .order_by(rank.desc())
            .limit(top_k)
        )

        result = self.db.execute(statement)

        return list(result.all())