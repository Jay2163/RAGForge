import time

from sqlalchemy.orm import Session

from app.evaluation.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)


class RetrievalEvaluator:

    def __init__(self, db: Session):
        self.db = db

    def evaluate_results(
        self,
        question,
        retrieved_results,
        relevant_chunk_ids,
    ):
        retrieved_ids = [
            chunk.id
            for chunk, _score in retrieved_results
        ]

        start = time.perf_counter()

        # Metrics are calculated locally,
        # so this timing only represents evaluation processing.
        hit1 = hit_at_k(
            retrieved_ids,
            relevant_chunk_ids,
            1,
        )

        hit3 = hit_at_k(
            retrieved_ids,
            relevant_chunk_ids,
            3,
        )

        hit5 = hit_at_k(
            retrieved_ids,
            relevant_chunk_ids,
            5,
        )

        precision5 = precision_at_k(
            retrieved_ids,
            relevant_chunk_ids,
            5,
        )

        recall5 = recall_at_k(
            retrieved_ids,
            relevant_chunk_ids,
            5,
        )

        mrr = reciprocal_rank(
            retrieved_ids,
            relevant_chunk_ids,
        )

        ndcg5 = ndcg_at_k(
            retrieved_ids,
            relevant_chunk_ids,
            5,
        )

        metric_time_ms = (
            time.perf_counter() - start
        ) * 1000

        return {
            "question": question,
            "retrieved_ids": retrieved_ids,
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "hit_at_5": hit5,
            "precision_at_5": precision5,
            "recall_at_5": recall5,
            "mrr": mrr,
            "ndcg_at_5": ndcg5,
            "metric_time_ms": metric_time_ms,
        }

    def evaluate_search_service(
        self,
        search_service,
        dataset,
        top_k=10,
    ):
        results = []

        for item in dataset:
            start = time.perf_counter()

            retrieved_results = search_service.search(
                query=item["question"],
                top_k=top_k,
            )

            search_latency_ms = (
                time.perf_counter() - start
            ) * 1000

            result = self.evaluate_results(
                question=item["question"],
                retrieved_results=retrieved_results,
                relevant_chunk_ids=set(
                    item["relevant_chunk_ids"]
                ),
            )

            result["id"] = item["id"]
            result["search_latency_ms"] = (
                search_latency_ms
            )

            results.append(result)

        return results