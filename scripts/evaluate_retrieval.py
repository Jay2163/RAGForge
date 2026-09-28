from statistics import mean

from app.db.database import SessionLocal
from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import RetrievalEvaluator
from app.services.vector_search import VectorSearchService


def print_result(result):
    print("\n" + "-" * 80)

    print(f"{result['id']}")
    print(f"Question: {result['question']}")

    print(
        f"Retrieved: {result['retrieved_ids']}"
    )

    print(
        f"Hit@1:     {result['hit_at_1']:.3f}"
    )

    print(
        f"Hit@3:     {result['hit_at_3']:.3f}"
    )

    print(
        f"Hit@5:     {result['hit_at_5']:.3f}"
    )

    print(
        f"Precision@5: {result['precision_at_5']:.3f}"
    )

    print(
        f"Recall@5:  {result['recall_at_5']:.3f}"
    )

    print(
        f"MRR:       {result['mrr']:.3f}"
    )

    print(
        f"NDCG@5:    {result['ndcg_at_5']:.3f}"
    )

    print(
        f"Latency:   {result['search_latency_ms']:.2f} ms"
    )


def print_average(results):
    print("\n" + "=" * 80)
    print("AVERAGE METRICS")
    print("=" * 80)

    print(
        f"Hit@1:       "
        f"{mean(r['hit_at_1'] for r in results):.3f}"
    )

    print(
        f"Hit@3:       "
        f"{mean(r['hit_at_3'] for r in results):.3f}"
    )

    print(
        f"Hit@5:       "
        f"{mean(r['hit_at_5'] for r in results):.3f}"
    )

    print(
        f"Precision@5: "
        f"{mean(r['precision_at_5'] for r in results):.3f}"
    )

    print(
        f"Recall@5:    "
        f"{mean(r['recall_at_5'] for r in results):.3f}"
    )

    print(
        f"MRR:         "
        f"{mean(r['mrr'] for r in results):.3f}"
    )

    print(
        f"NDCG@5:      "
        f"{mean(r['ndcg_at_5'] for r in results):.3f}"
    )

    print(
        f"Latency:     "
        f"{mean(r['search_latency_ms'] for r in results):.2f} ms"
    )


def main():
    db = SessionLocal()

    try:
        evaluator = RetrievalEvaluator(db)

        vector_search = VectorSearchService(db)

        results = evaluator.evaluate_search_service(
            search_service=vector_search,
            dataset=EVALUATION_DATASET,
            top_k=10,
        )

        print("\n" + "=" * 80)
        print("RAGFORGE — VECTOR SEARCH BASELINE")
        print("=" * 80)

        for result in results:
            print_result(result)

        print_average(results)

    finally:
        db.close()


if __name__ == "__main__":
    main()