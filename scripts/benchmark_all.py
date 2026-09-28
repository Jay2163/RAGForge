import time
from statistics import mean
from typing import Any

from app.db.database import SessionLocal
from app.models.document_chunk import DocumentChunk
from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import RetrievalEvaluator

from app.services.vector_search import VectorSearchService
from app.services.lexical_search import LexicalSearchService
from app.services.chroma_search import ChromaSearchService
from app.services.bm25_search import BM25SearchService
from app.services.reranker import CrossEncoderReranker


# -------------------------------------------------------------------------
# Search Pipeline Adapters
# -------------------------------------------------------------------------

class ChromaAdapter:
    """Adapts ChromaSearchService to the evaluator interface."""
    def __init__(self, chroma_service: ChromaSearchService):
        self.chroma = chroma_service

    def search(self, query: str, top_k: int = 5):
        return self.chroma.search(query=query, top_k=top_k)


class BM25Adapter:
    """Adapts BM25SearchService to the evaluator interface."""
    def __init__(self, bm25_service: BM25SearchService):
        self.bm25 = bm25_service

    def search(self, query: str, top_k: int = 5):
        return self.bm25.search(query=query, top_k=top_k)


class ChromaBM25HybridAdapter:
    """
    Hybrid Search combining ChromaDB (Dense) + Rank-BM25 (Lexical)
    using Reciprocal Rank Fusion (RRF).
    """
    RRF_K = 60

    def __init__(self, chroma: ChromaSearchService, bm25: BM25SearchService):
        self.chroma = chroma
        self.bm25 = bm25

    def search(self, query: str, top_k: int = 5):
        # 1. Retrieve candidates from both systems
        chroma_res = self.chroma.search(query=query, top_k=20)
        bm25_res = self.bm25.search(query=query, top_k=20)

        # 2. Combine ranks using RRF
        combined: dict[int, dict[str, Any]] = {}

        # Dense ranking
        for rank, (chunk, _) in enumerate(chroma_res, start=1):
            combined[chunk.id] = {
                "chunk": chunk,
                "rrf_score": 1.0 / (self.RRF_K + rank),
            }

        # Lexical ranking
        for rank, (chunk, _) in enumerate(bm25_res, start=1):
            if chunk.id not in combined:
                combined[chunk.id] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                }
            combined[chunk.id]["rrf_score"] += 1.0 / (self.RRF_K + rank)

        # Sort by RRF score
        ranked = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
        return [(item["chunk"], item["rrf_score"]) for item in ranked[:top_k]]


class ChromaBM25RerankAdapter:
    """
    Two-Stage Hybrid Search Pipeline:
    Stage 1: ChromaDB + BM25 Hybrid retrieves top 20 candidates.
    Stage 2: Cross-Encoder Transformer re-ranks to top 5.
    """
    def __init__(self, hybrid_adapter: ChromaBM25HybridAdapter, reranker: CrossEncoderReranker):
        self.hybrid = hybrid_adapter
        self.reranker = reranker

    def search(self, query: str, top_k: int = 5):
        # Stage 1: Candidate generation (retrieve top 20)
        candidates = self.hybrid.search(query=query, top_k=20)
        
        # Stage 2: Cross-Encoder Re-ranking
        reranked = self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        return reranked


# -------------------------------------------------------------------------
# Benchmark Runner
# -------------------------------------------------------------------------

def compute_summary(results: list[dict]) -> dict:
    return {
        "hit_at_1": mean(r["hit_at_1"] for r in results),
        "hit_at_3": mean(r["hit_at_3"] for r in results),
        "hit_at_5": mean(r["hit_at_5"] for r in results),
        "precision_at_5": mean(r["precision_at_5"] for r in results),
        "recall_at_5": mean(r["recall_at_5"] for r in results),
        "mrr": mean(r["mrr"] for r in results),
        "ndcg_at_5": mean(r["ndcg_at_5"] for r in results),
        "latency_ms": mean(r["search_latency_ms"] for r in results),
    }


def main():
    db = SessionLocal()
    try:
        evaluator = RetrievalEvaluator(db)

        print("=" * 105)
        print("RAGFORGE — COMPREHENSIVE RETRIEVAL BENCHMARK ACROSS 6 ARCHITECTURES")
        print(f"Evaluating {len(EVALUATION_DATASET)} Ground-Truth Queries")
        print("=" * 105)

        # 1. Load All Chunks for BM25
        all_chunks = db.query(DocumentChunk).filter(DocumentChunk.embedding.is_not(None)).all()
        bm25_service = BM25SearchService(all_chunks)
        chroma_service = ChromaSearchService()
        reranker_service = CrossEncoderReranker()

        # Define all 6 strategies
        strategies = [
            ("1. PGVector (Dense)", VectorSearchService(db)),
            ("2. Postgres tsvector (Lexical)", LexicalSearchService(db)),
            ("3. ChromaDB (Dense HNSW)", ChromaAdapter(chroma_service)),
            ("4. Rank-BM25 (Lexical)", BM25Adapter(bm25_service)),
            (
                "5. Chroma + BM25 (Hybrid RRF)",
                ChromaBM25HybridAdapter(chroma_service, bm25_service),
            ),
            (
                "6. Hybrid + Cross-Encoder Reranker",
                ChromaBM25RerankAdapter(
                    ChromaBM25HybridAdapter(chroma_service, bm25_service),
                    reranker_service,
                ),
            ),
        ]

        benchmark_matrix = []

        for name, service in strategies:
            print(f"\nEvaluating: {name}...")
            res = evaluator.evaluate_search_service(service, EVALUATION_DATASET, top_k=5)
            summary = compute_summary(res)
            summary["name"] = name
            benchmark_matrix.append(summary)

        # Render Comparison Matrix
        print("\n" + "=" * 105)
        print("FINAL RETRIEVAL BENCHMARK COMPARISON MATRIX")
        print("=" * 105)
        headers = [
            "Retrieval Strategy",
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "Precision@5",
            "Recall@5",
            "MRR",
            "NDCG@5",
            "Latency (ms)",
        ]
        col_widths = [36, 8, 8, 8, 12, 10, 8, 8, 14]

        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        print(header_line)
        print(sep_line)

        for s in benchmark_matrix:
            row = [
                s["name"],
                f"{s['hit_at_1']:.3f}",
                f"{s['hit_at_3']:.3f}",
                f"{s['hit_at_5']:.3f}",
                f"{s['precision_at_5']:.3f}",
                f"{s['recall_at_5']:.3f}",
                f"{s['mrr']:.3f}",
                f"{s['ndcg_at_5']:.3f}",
                f"{s['latency_ms']:.1f}",
            ]
            print(" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)))
        print("=" * 105)

    finally:
        db.close()


if __name__ == "__main__":
    main()
