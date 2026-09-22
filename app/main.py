from fastapi import FastAPI

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