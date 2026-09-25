from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.vector_search import VectorSearchService


router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Natural language search query",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )


@router.post("")
def search_documents(
    request: SearchRequest,
    db: Session = Depends(get_db),
):
    service = VectorSearchService(db)

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
                "similarity": round(similarity, 4),
                "content": chunk.content,
                "page_number": chunk.page_number,
                "section": chunk.section,
            }
            for chunk, similarity in results
        ],
    }