from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.vector_search import VectorSearchService
from app.services.llm import LLMService


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to ask the knowledge base",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )


@router.post("/ask")
def ask_question(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    # 1. Retrieve relevant chunks
    vector_search = VectorSearchService(db)

    chunks = vector_search.search(
        query=request.question,
        top_k=request.top_k,
    )

    # 2. Build context for the LLM
    context_parts = []

    for index, (chunk, similarity) in enumerate(
        chunks,
        start=1,
    ):
        source = (
            f"[Source {index}]\n"
            f"Document ID: {chunk.document_id}\n"
            f"Chunk ID: {chunk.id}\n"
            f"Page: {chunk.page_number}\n"
            f"Section: {chunk.section}\n"
            f"Similarity: {similarity:.4f}\n"
            f"Content:\n{chunk.content}"
        )

        context_parts.append(source)

    context = "\n\n".join(context_parts)

    # 3. Generate answer using Gemini
    llm = LLMService()

    answer = llm.generate(
        question=request.question,
        context=context,
    )

    # 4. Return answer + source metadata
    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            {
                "source": index,
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "similarity": round(similarity, 4),
                "page_number": chunk.page_number,
                "section": chunk.section,
            }
            for index, (chunk, similarity) in enumerate(
                chunks,
                start=1,
            )
        ],
    }