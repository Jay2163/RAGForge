from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.lexical_search import LexicalSearchService


router = APIRouter(
    prefix="/lexical-search",
    tags=["Lexical Search"],
)


class LexicalSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Keyword-based search query",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )


@router.post("")
def lexical_search(
    request: LexicalSearchRequest,
    db: Session = Depends(get_db),
):
    service = LexicalSearchService(db)

    results = service.search(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "results": [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "rank": round(rank, 4),
                "content": chunk.content,
                "page_number": chunk.page_number,
                "section": chunk.section,
            }
            for chunk, rank in results
        ],
    }