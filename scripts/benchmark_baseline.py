from statistics import mean

from app.db.database import SessionLocal
from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import RetrievalEvaluator
from app.services.vector_search import VectorSearchService
from app.services.lexical_search import LexicalSearchService
from app.services.hybrid_search import HybridSearchService


class HybridSearchAdapter:
    """Adapter to make HybridSearchService conform to the evaluator interface."""
    def __init__(self, service: HybridSearchService):
        self.service = service

    def search(self, query: str, top_k: int = 5):
        results = self.service.search(query=query, top_k=top_k)
        return [(r.chunk, r.rrf_score) for r in results]


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

        print("=" * 90)
        print("RUNNING RETRIEVAL BENCHMARK ACROSS 3 BASELINE STRATEGIES")
        print(f"Dataset Size: {len(EVALUATION_DATASET)} Ground-Truth Queries")
        print("=" * 90)

        # 1. PGVector Dense
        print("\n1. Evaluating PGVector (Dense Vector Search)...")
        v_svc = VectorSearchService(db)
        v_res = evaluator.evaluate_search_service(v_svc, EVALUATION_DATASET, top_k=5)
        v_sum = compute_summary(v_res)

        # 2. Postgres Lexical (tsvector)
        print("2. Evaluating PostgreSQL tsvector (Lexical Search)...")
        l_svc = LexicalSearchService(db)
        l_res = evaluator.evaluate_search_service(l_svc, EVALUATION_DATASET, top_k=5)
        l_sum = compute_summary(l_res)

        # 3. Postgres Hybrid (RRF)
        print("3. Evaluating PostgreSQL Hybrid (RRF k=60)...")
        h_svc = HybridSearchAdapter(HybridSearchService(db))
        h_res = evaluator.evaluate_search_service(h_svc, EVALUATION_DATASET, top_k=5)
        h_sum = compute_summary(h_res)

        # Print Comparison Table
        print("\n" + "=" * 90)
        print("BENCHMARK COMPARISON MATRIX")
        print("=" * 90)
        headers = ["Retrieval Strategy", "Hit@1", "Hit@3", "Hit@5", "Precision@5", "Recall@5", "MRR", "NDCG@5", "Latency (ms)"]
        rows = [
            ["PGVector (Dense)", f"{v_sum['hit_at_1']:.3f}", f"{v_sum['hit_at_3']:.3f}", f"{v_sum['hit_at_5']:.3f}", f"{v_sum['precision_at_5']:.3f}", f"{v_sum['recall_at_5']:.3f}", f"{v_sum['mrr']:.3f}", f"{v_sum['ndcg_at_5']:.3f}", f"{v_sum['latency_ms']:.1f}"],
            ["Postgres tsvector (Lexical)", f"{l_sum['hit_at_1']:.3f}", f"{l_sum['hit_at_3']:.3f}", f"{l_sum['hit_at_5']:.3f}", f"{l_sum['precision_at_5']:.3f}", f"{l_sum['recall_at_5']:.3f}", f"{l_sum['mrr']:.3f}", f"{l_sum['ndcg_at_5']:.3f}", f"{l_sum['latency_ms']:.1f}"],
            ["Postgres Hybrid (RRF)", f"{h_sum['hit_at_1']:.3f}", f"{h_sum['hit_at_3']:.3f}", f"{h_sum['hit_at_5']:.3f}", f"{h_sum['precision_at_5']:.3f}", f"{h_sum['recall_at_5']:.3f}", f"{h_sum['mrr']:.3f}", f"{h_sum['ndcg_at_5']:.3f}", f"{h_sum['latency_ms']:.1f}"],
        ]

        # Format as table
        col_widths = [28, 8, 8, 8, 12, 10, 8, 8, 14]
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        print(header_line)
        print(sep_line)
        for row in rows:
            print(" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)))
        print("=" * 90)

    finally:
        db.close()


if __name__ == "__main__":
    main()
