import pandas as pd
import bm25s
from typing import Any, Dict

STATE: Dict[str, Any] = {}

def search_tool(
    question,
    top_k=5,
    candidate_k=30,
    label_filter=None
):
    """
    Hybrid search tool:
    BM25S + BGE semantic search + RRF.
    """

    chunks = STATE["full_chunks_df"]
    embeddings = STATE["full_embeddings"]
    bm25 = STATE["full_bm25s"]
    embedding_model = STATE["embedding_model"]

    # -----------------------------------------------
    # Optional document-type filtering
    # -----------------------------------------------

    if label_filter:
        filtered = chunks[
            chunks["label"] == label_filter
        ].copy()
    else:
        filtered = chunks.copy()

    if filtered.empty:
        return {
            "tool": "search",
            "status": "empty",
            "results": []
        }

    # -----------------------------------------------
    # BM25S
    # -----------------------------------------------

    query_tokens = bm25s.tokenize(
        [question],
        show_progress=False
    )

    bm25_result = bm25.retrieve(
        query_tokens,
        k=candidate_k,
        show_progress=False
    )

    bm25_indices = bm25_result[1][0]

    # Keep only indices belonging to our filtered set
    allowed_indices = set(filtered.index.to_numpy())

    bm25_candidates = [
        int(i)
        for i in bm25_indices
        if int(i) in allowed_indices
    ]

    # -----------------------------------------------
    # BGE semantic search
    # -----------------------------------------------

    query_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    semantic_scores = embeddings[
        filtered.index.to_numpy()
    ] @ query_embedding

    semantic_order = semantic_scores.argsort()[::-1]

    semantic_candidates = [
        int(filtered.index.to_numpy()[i])
        for i in semantic_order[:candidate_k]
    ]

    # -----------------------------------------------
    # RRF
    # -----------------------------------------------

    rrf_k = 60
    scores = {}

    for rank, idx in enumerate(
        bm25_candidates,
        start=1
    ):
        scores[idx] = scores.get(idx, 0) + 1 / (
            rrf_k + rank
        )

    for rank, idx in enumerate(
        semantic_candidates,
        start=1
    ):
        scores[idx] = scores.get(idx, 0) + 1 / (
            rrf_k + rank
        )

    ranked_indices = sorted(
        scores,
        key=lambda x: scores.get(x, 0.0),
        reverse=True
    )[:top_k]

    # -----------------------------------------------
    # Build results
    # -----------------------------------------------

    output = []

    for idx in ranked_indices:

        row = chunks.loc[idx]

        output.append({
            "document_id": row["document_id"],
            "chunk_id": row["chunk_id"],
            "label": row["label"],
            "score": float(scores[idx]),
            "text": str(row["text"])
        })

    return {
        "tool": "search",
        "status": "success",
        "results": output
    }