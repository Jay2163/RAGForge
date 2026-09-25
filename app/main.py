from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.api.chat import router as chat_router
from app.api.lexical_search import router as lexical_search_router
from app.api.hybrid_search import router as hybrid_search_router


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
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(lexical_search_router)
app.include_router(hybrid_search_router)