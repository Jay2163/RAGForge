from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.document_chunk import DocumentChunk
from app.services.bm25_search import BM25SearchService

router = APIRouter(
    prefix="/bm25",
    tags=["BM25 Search"],
)


class BM25SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Keyword or code search query",
        examples=["OAuth2PasswordRequestForm status_code"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of highest-ranking lexical chunks to retrieve",
    )


@router.post("/search")
def search_bm25(
    request: BM25SearchRequest,
    db: Session = Depends(get_db),
):
    """
    Perform probabilistic lexical search using Rank-BM25 with term saturation and document length normalization.
    """
    chunks = db.query(DocumentChunk).filter(DocumentChunk.content.is_not(None)).all()
    service = BM25SearchService(chunks)

    results = service.search(query=request.query, top_k=request.top_k)

    return {
        "engine": "Rank-BM25 (Probabilistic Lexical)",
        "query": request.query,
        "total_results": len(results),
        "results": [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "bm25_score": round(score, 4),
                "page_number": chunk.page_number,
                "section": chunk.section,
                "content": chunk.content,
            }
            for chunk, score in results
        ],
    }
