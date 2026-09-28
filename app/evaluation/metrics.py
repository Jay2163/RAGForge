import math


def hit_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]

    return float(
        any(chunk_id in relevant_ids for chunk_id in top_k)
    )


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        chunk_id in relevant_ids
        for chunk_id in top_k
    )

    return relevant_count / len(top_k)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])

    retrieved_relevant = top_k.intersection(
        relevant_ids
    )

    return len(retrieved_relevant) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    dcg = 0.0

    for rank, chunk_id in enumerate(
        top_k,
        start=1,
    ):
        relevance = (
            1 if chunk_id in relevant_ids else 0
        )

        dcg += relevance / math.log2(rank + 1)

    ideal_count = min(
        len(relevant_ids),
        k,
    )

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg