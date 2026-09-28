from typing import Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.document_chunk import DocumentChunk
from app.services.chroma_search import ChromaSearchService
from app.services.bm25_search import BM25SearchService
from app.services.reranker import CrossEncoderReranker

router = APIRouter(
    prefix="/advanced",
    tags=["Advanced Search (Hybrid & Re-ranking)"],
)


class HybridSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Search query",
        examples=["How do you configure CORS middleware in FastAPI?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of final fused results to return",
    )
    candidate_k: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of candidates to retrieve from each retriever before fusion",
    )
    rrf_k: int = Field(
        default=60,
        description="Smoothing constant for Reciprocal Rank Fusion",
    )


class RerankSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Search query to be evaluated by cross-attention",
        examples=["How do you handle authentication in FastAPI?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of re-ranked chunks to return",
    )
    candidate_k: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of hybrid candidates to generate before re-ranking",
    )


def execute_chroma_bm25_rrf(
    query: str,
    top_k: int,
    candidate_k: int,
    rrf_k: int,
    db: Session,
) -> list[dict[str, Any]]:
    # 1. Retrieve candidates from ChromaDB
    chroma_svc = ChromaSearchService()
    chroma_results = chroma_svc.search(query=query, top_k=candidate_k)

    # 2. Retrieve candidates from BM25
    chunks = db.query(DocumentChunk).filter(DocumentChunk.content.is_not(None)).all()
    bm25_svc = BM25SearchService(chunks)
    bm25_results = bm25_svc.search(query=query, top_k=candidate_k)

    # 3. Reciprocal Rank Fusion (RRF)
    combined: dict[int, dict[str, Any]] = {}

    for rank, (chunk, score) in enumerate(chroma_results, start=1):
        combined[chunk.id] = {
            "chunk": chunk,
            "rrf_score": 1.0 / (rrf_k + rank),
            "dense_rank": rank,
            "lexical_rank": None,
        }

    for rank, (chunk, score) in enumerate(bm25_results, start=1):
        if chunk.id not in combined:
            combined[chunk.id] = {
                "chunk": chunk,
                "rrf_score": 0.0,
                "dense_rank": None,
                "lexical_rank": rank,
            }
        combined[chunk.id]["rrf_score"] += 1.0 / (rrf_k + rank)
        combined[chunk.id]["lexical_rank"] = rank

    ranked = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:top_k]


@router.post("/hybrid")
def search_chroma_bm25_hybrid(
    request: HybridSearchRequest,
    db: Session = Depends(get_db),
):
    """
    Two-way Hybrid Search combining ChromaDB Dense Embeddings + Rank-BM25 Lexical Search via RRF.
    """
    fused_results = execute_chroma_bm25_rrf(
        query=request.query,
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        rrf_k=request.rrf_k,
        db=db,
    )

    return {
        "engine": "ChromaDB + Rank-BM25 Hybrid (RRF)",
        "query": request.query,
        "total_results": len(fused_results),
        "results": [
            {
                "chunk_id": item["chunk"].id,
                "document_id": item["chunk"].document_id,
                "chunk_index": item["chunk"].chunk_index,
                "rrf_score": round(item["rrf_score"], 6),
                "dense_rank": item["dense_rank"],
                "lexical_rank": item["lexical_rank"],
                "page_number": item["chunk"].page_number,
                "section": item["chunk"].section,
                "content": item["chunk"].content,
            }
            for item in fused_results
        ],
    }


@router.post("/rerank")
def search_and_rerank(
    request: RerankSearchRequest,
    db: Session = Depends(get_db),
):
    """
    Two-Stage Precision Search:
    Stage 1: Generates top candidate chunks via ChromaDB + BM25 Hybrid RRF.
    Stage 2: Re-ranks all candidates using a Cross-Encoder Transformer (ms-marco-MiniLM-L-6-v2).
    """
    # Stage 1: Candidate Generation
    candidates_info = execute_chroma_bm25_rrf(
        query=request.query,
        top_k=request.candidate_k,
        candidate_k=request.candidate_k,
        rrf_k=60,
        db=db,
    )
    candidates = [item["chunk"] for item in candidates_info]

    # Stage 2: Cross-Encoder Re-ranking
    reranker = CrossEncoderReranker()
    reranked_results = reranker.rerank(
        query=request.query,
        candidates=candidates,
        top_k=request.top_k,
    )

    return {
        "engine": "Two-Stage Retrieval (ChromaDB + BM25 Hybrid -> Cross-Encoder Re-ranker)",
        "query": request.query,
        "total_results": len(reranked_results),
        "results": [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "cross_encoder_score": round(score, 4),
                "page_number": chunk.page_number,
                "section": chunk.section,
                "content": chunk.content,
            }
            for chunk, score in reranked_results
        ],
    }
