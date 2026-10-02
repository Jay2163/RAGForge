from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.api.lexical_search import router as lexical_search_router
from app.api.hybrid_search import router as hybrid_search_router
from app.api.chroma import router as chroma_router
from app.api.bm25 import router as bm25_router
from app.api.advanced_search import router as advanced_search_router
from app.api.chat import router as chat_router


app = FastAPI(
    title="RAGForge",
    description=(
        "Production-Grade Engineering Knowledge Intelligence Platform "
        "featuring PGVector, ChromaDB, Rank-BM25, Hybrid RRF, Cross-Encoder Re-ranking, and Gemini LLM."
    ),
    version="0.2.0",
)

# Core Routers
app.include_router(documents_router)
app.include_router(chat_router)

# Retrieval Search Endpoints
app.include_router(search_router)             # POST /search (PGVector Dense)
app.include_router(lexical_search_router)     # POST /lexical-search (Postgres tsvector)
app.include_router(chroma_router)             # POST /chroma/search (ChromaDB HNSW)
app.include_router(bm25_router)               # POST /bm25/search (Rank-BM25)
app.include_router(hybrid_search_router)       # POST /hybrid-search (Postgres RRF)
app.include_router(advanced_search_router)     # POST /advanced/hybrid & /advanced/rerank