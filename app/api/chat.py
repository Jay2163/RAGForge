from enum import Enum
from typing import Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.document_chunk import DocumentChunk
from app.services.vector_search import VectorSearchService
from app.services.hybrid_search import HybridSearchService
from app.services.chroma_search import ChromaSearchService
from app.services.bm25_search import BM25SearchService
from app.services.reranker import CrossEncoderReranker
from app.services.llm import LLMService

router = APIRouter(
    prefix="/chat",
    tags=["Chat & RAG Generation"],
)


class RetrievalPipeline(str, Enum):
    PGVECTOR = "pgvector"                  # 1. PostgreSQL pgvector Dense
    POSTGRES_HYBRID = "postgres_hybrid"    # 2. PostgreSQL pgvector + tsvector RRF
    CHROMA = "chroma"                      # 3. ChromaDB HNSW Dense
    BM25 = "bm25"                          # 4. Rank-BM25 Lexical
    HYBRID_RRF = "hybrid_rrf"              # 5. ChromaDB + BM25 Hybrid (RRF)
    RERANKED = "reranked"                  # 6. ChromaDB + BM25 + Cross-Encoder Re-ranking


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to ask the engineering knowledge base",
        examples=["How do you define a path parameter in FastAPI?"],
    )
    pipeline: RetrievalPipeline = Field(
        default=RetrievalPipeline.RERANKED,
        description=(
            "Retrieval pipeline architecture: "
            "'pgvector' (PostgreSQL Dense), "
            "'postgres_hybrid' (Postgres pgvector + tsvector), "
            "'chroma' (ChromaDB HNSW Dense), "
            "'bm25' (Rank-BM25 Lexical), "
            "'hybrid_rrf' (ChromaDB + BM25 Hybrid RRF), "
            "'reranked' (ChromaDB + BM25 + Cross-Encoder Transformer Re-ranker)"
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of evidence context chunks to provide to the LLM",
    )


def retrieve_context(
    query: str,
    pipeline: RetrievalPipeline,
    top_k: int,
    db: Session,
) -> list[tuple[Any, float]]:
    # 1. PGVector Dense
    if pipeline == RetrievalPipeline.PGVECTOR:
        svc = VectorSearchService(db)
        return svc.search(query=query, top_k=top_k)

    # 2. PostgreSQL Hybrid (pgvector + tsvector RRF)
    elif pipeline == RetrievalPipeline.POSTGRES_HYBRID:
        hyb_svc = HybridSearchService(db)
        results = hyb_svc.search(query=query, top_k=top_k)
        return [(r.chunk, r.rrf_score) for r in results]

    # 3. ChromaDB HNSW Dense
    elif pipeline == RetrievalPipeline.CHROMA:
        svc = ChromaSearchService()
        return svc.search(query=query, top_k=top_k)

    # 4. Rank-BM25 Lexical
    elif pipeline == RetrievalPipeline.BM25:
        chunks = db.query(DocumentChunk).filter(DocumentChunk.content.is_not(None)).all()
        svc = BM25SearchService(chunks)
        return svc.search(query=query, top_k=top_k)

    # 5. ChromaDB + BM25 Hybrid (RRF)
    elif pipeline == RetrievalPipeline.HYBRID_RRF:
        chroma_svc = ChromaSearchService()
        chunks = db.query(DocumentChunk).filter(DocumentChunk.content.is_not(None)).all()
        bm25_svc = BM25SearchService(chunks)

        chroma_res = chroma_svc.search(query=query, top_k=20)
        bm25_res = bm25_svc.search(query=query, top_k=20)

        combined: dict[int, dict[str, Any]] = {}
        for rank, (chunk, _) in enumerate(chroma_res, start=1):
            combined[chunk.id] = {"chunk": chunk, "rrf": 1.0 / (60 + rank)}
        for rank, (chunk, _) in enumerate(bm25_res, start=1):
            if chunk.id not in combined:
                combined[chunk.id] = {"chunk": chunk, "rrf": 0.0}
            combined[chunk.id]["rrf"] += 1.0 / (60 + rank)

        ranked = sorted(combined.values(), key=lambda x: x["rrf"], reverse=True)
        return [(item["chunk"], item["rrf"]) for item in ranked[:top_k]]

    # 6. Two-Stage Retrieval: ChromaDB + BM25 Hybrid -> Cross-Encoder Re-ranker
    elif pipeline == RetrievalPipeline.RERANKED:
        chroma_svc = ChromaSearchService()
        chunks = db.query(DocumentChunk).filter(DocumentChunk.content.is_not(None)).all()
        bm25_svc = BM25SearchService(chunks)

        chroma_res = chroma_svc.search(query=query, top_k=20)
        bm25_res = bm25_svc.search(query=query, top_k=20)

        combined: dict[int, dict[str, Any]] = {}
        for rank, (chunk, _) in enumerate(chroma_res, start=1):
            combined[chunk.id] = {"chunk": chunk, "rrf": 1.0 / (60 + rank)}
        for rank, (chunk, _) in enumerate(bm25_res, start=1):
            if chunk.id not in combined:
                combined[chunk.id] = {"chunk": chunk, "rrf": 0.0}
            combined[chunk.id]["rrf"] += 1.0 / (60 + rank)

        ranked = sorted(combined.values(), key=lambda x: x["rrf"], reverse=True)
        candidates = [item["chunk"] for item in ranked[:20]]

        reranker = CrossEncoderReranker()
        return reranker.rerank(query=query, candidates=candidates, top_k=top_k)

    return []


@router.post("/ask")
def ask_question(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Grounded RAG Question Answering Endpoint.
    Retrieves evidence from the selected pipeline and generates an accurate, hallucination-free answer with Gemini.
    """
    # 1. Retrieve relevant evidence chunks
    scored_chunks = retrieve_context(
        query=request.question,
        pipeline=request.pipeline,
        top_k=request.top_k,
        db=db,
    )

    # 2. Build structured evidence context
    context_parts = []
    for index, (chunk, score) in enumerate(scored_chunks, start=1):
        source = (
            f"[Source {index}]\n"
            f"Document ID: {chunk.document_id}\n"
            f"Chunk ID: {chunk.id}\n"
            f"Page: {chunk.page_number}\n"
            f"Section: {chunk.section}\n"
            f"Relevance Score: {score:.4f}\n"
            f"Content:\n{chunk.content}"
        )
        context_parts.append(source)

    context = "\n\n".join(context_parts)

    # 3. Generate grounded answer using Gemini LLM
    llm = LLMService()
    answer = llm.generate(
        question=request.question,
        context=context,
    )

    # 4. Return answer + citation sources
    return {
        "question": request.question,
        "pipeline_used": request.pipeline.value,
        "answer": answer,
        "sources": [
            {
                "source": index,
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "relevance_score": round(score, 4),
                "page_number": chunk.page_number,
                "section": chunk.section,
            }
            for index, (chunk, score) in enumerate(scored_chunks, start=1)
        ],
    }