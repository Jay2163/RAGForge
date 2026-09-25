import math


def hit_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """
    Returns 1 if at least one relevant chunk
    appears in the top-k results.
    """

    top_k = retrieved_ids[:k]

    return float(
        any(chunk_id in relevant_ids for chunk_id in top_k)
    )


def recall_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """
    Measures how many relevant chunks were retrieved
    within the top-k results.
    """

    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])

    retrieved_relevant = top_k.intersection(
        relevant_ids
    )

    return len(retrieved_relevant) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[int],
    relevant_ids: set[int],
) -> float:
    """
    Reciprocal rank of the first relevant result.
    """

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """
    Binary relevance NDCG@K.
    """

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