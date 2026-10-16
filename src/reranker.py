"""Cross-encoder re-ranking for the mortgage exception retrieval pipeline.

Pipeline
--------
1. Bi-encoder (MiniLM) in find_similar_cases produces a short candidate list
   from the ChromaDB vector index (fast approximate recall).
2. rerank() takes those candidates, scores every (query, chunk) pair with the
   cross-encoder, and returns them sorted by the cross-encoder score.
3. Feedback bias from suggestion_feedback (accept/reject votes) is applied on
   top of the cross-encoder score so confirmed good matches rise further and
   confirmed bad matches are suppressed.

The cross-encoder runs on CPU without any GPU requirement and adds < 200 ms
per query on the synthetic v1 dataset size.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

# ms-marco MiniLM L-6 is 22 MB, CPU-friendly, and well-calibrated for passage
# relevance tasks which is semantically close to exception-text matching.
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank(
    query: str,
    candidates: pd.DataFrame,
    feedback_bias: Optional[dict[str, float]] = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Re-rank retrieved candidates using the cross-encoder.

    Args:
        query: The incoming exception text (already PII-sanitised by the caller).
        candidates: DataFrame from find_similar_cases; must have a
            ``matched_chunk`` column used as the passage to score.
        feedback_bias: Optional mapping of case_id → float delta computed from
            stored accept/reject votes. Positive values push a case up;
            negative values push it down.
        top_k: Maximum rows to return after re-ranking.

    Returns:
        A new DataFrame sorted by ``rerank_score`` descending, with the column
        added alongside the original ``similarity`` (bi-encoder score).
    """
    if candidates.empty:
        return candidates

    encoder = _get_cross_encoder()
    passages = candidates["matched_chunk"].fillna(candidates["exception_text"]).tolist()
    pairs = [(query, passage) for passage in passages]

    # CrossEncoder.predict returns a raw logit; higher = more relevant.
    logits: np.ndarray = encoder.predict(pairs)

    # Normalise logits to [0, 1] with sigmoid so the scale is comparable to
    # bi-encoder cosine similarity.
    scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))

    result = candidates.copy()
    result["rerank_score"] = scores

    # Apply feedback bias if provided.
    if feedback_bias:
        result["rerank_score"] = result.apply(
            lambda row: float(row["rerank_score"])
            + feedback_bias.get(str(row["case_id"]), 0.0),
            axis=1,
        )
        # Clamp to [0, 1] so downstream confidence display stays sensible.
        result["rerank_score"] = result["rerank_score"].clip(0.0, 1.0)

    result = result.sort_values("rerank_score", ascending=False)
    return result.head(top_k).reset_index(drop=True)
