from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.chroma_search import ChromaSearchService

router = APIRouter(
    prefix="/chroma",
    tags=["ChromaDB Search"],
)


class ChromaSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Natural language search query",
        examples=["How do you define a path parameter in FastAPI?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of nearest neighbor chunks to retrieve",
    )


@router.post("/search")
def search_chroma(request: ChromaSearchRequest):
    """
    Perform dense vector similarity search using persistent ChromaDB (HNSW index).
    """
    service = ChromaSearchService()
    results = service.search(query=request.query, top_k=request.top_k)

    return {
        "engine": "ChromaDB (HNSW Cosine)",
        "query": request.query,
        "total_results": len(results),
        "results": [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "similarity": round(similarity, 4),
                "page_number": chunk.page_number,
                "section": chunk.section,
                "content": chunk.content,
            }
            for chunk, similarity in results
        ],
    }
