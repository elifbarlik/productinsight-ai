"""Standard binary-relevance ranking metrics."""
import math
from typing import Sequence


def precision_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    top = list(ranked)[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def recall_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in list(ranked)[:k] if d in relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[int], relevant: set) -> float:
    for rank, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(list(ranked)[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0
