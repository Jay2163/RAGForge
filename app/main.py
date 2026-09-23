from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.health import router as health_router


app = FastAPI(
    title="RAGForge",
    description=(
        "Engineering Knowledge Intelligence Platform "
        "using RAG, hybrid search, reranking, and LLMs."
    ),
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(documents_router)