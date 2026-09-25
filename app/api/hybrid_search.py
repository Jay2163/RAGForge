from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.hybrid_search import HybridSearchService


router = APIRouter(
    prefix="/hybrid-search",
    tags=["Hybrid Search"],
)


class HybridSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Search query",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )


@router.post("")
def hybrid_search(
    request: HybridSearchRequest,
    db: Session = Depends(get_db),
):
    service = HybridSearchService(db)

    results = service.search(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "results": [
            {
                "rank": index,
                "chunk_id": result.chunk.id,
                "document_id": result.chunk.document_id,
                "chunk_index": result.chunk.chunk_index,
                "rrf_score": round(
                    result.rrf_score,
                    6,
                ),
                "vector_rank": result.vector_rank,
                "lexical_rank": result.lexical_rank,
                "content": result.chunk.content,
                "page_number": result.chunk.page_number,
                "section": result.chunk.section,
            }
            for index, result in enumerate(
                results,
                start=1,
            )
        ],
    }