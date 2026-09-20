# ============================================================
# DocuMind Evaluation: Hybrid Search & Retrieval Benchmark
# ============================================================

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import bm25s
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Setup paths
EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from common.metrics import compute_mrr, compute_ndcg_at_k, compute_recall_at_k
from common.utils import PROJECT_ROOT, ML_DIR, timer, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.retrieval")


def run_retrieval_evaluation() -> Dict[str, Any]:
    """
    Evaluates BM25-only, BGE-only, and Hybrid (RRF) retrieval across the 15-query
    retrieval benchmark on the full 223,234 enterprise document chunks.
    Computes Recall@1, Recall@3, Recall@5, Recall@10, MRR, and nDCG@10.
    """
    logger.info("Starting Hybrid Search / Retrieval Evaluation...")

    benchmark_path = EVAL_DIR / "retrieval" / "retrieval_benchmark.json"
    chunks_path = ML_DIR / "processed" / "full_text_chunks.parquet"
    embeddings_path = ML_DIR / "processed" / "full_text_embeddings.npy"
    bm25_path = ML_DIR / "processed" / "full_text_bm25s"

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file missing: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    logger.info("Loaded %d benchmark queries from %s", len(queries), benchmark_path)

    # 1. Load Corpus Chunks & BM25 Index
    logger.info("Loading chunk corpus and BM25 index...")
    chunks_df = pd.read_parquet(chunks_path)
    embeddings = np.load(embeddings_path)
    bm25 = bm25s.BM25.load(bm25_path, mmap=True)

    # 2. Load BGE Embedding Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading BGE embedding model on device: %s...", device)
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    systems = ["bm25_only", "bge_semantic_only", "hybrid_rrf"]
    results_by_system: Dict[str, Dict[str, Any]] = {
        s: {
            "recall_1": [],
            "recall_3": [],
            "recall_5": [],
            "recall_10": [],
            "mrr": [],
            "ndcg_10": [],
            "latency_ms": [],
        }
        for s in systems
    }

    per_query_details = []

    for item in queries:
        q_id = item["query_id"]
        q_text = item["query"]
        target_doc_ids = set(item["target_doc_ids"])

        # --- A. BM25 Retrieval ---
        with timer() as t_bm25:
            q_tokens = bm25s.tokenize([q_text], show_progress=False)
            bm25_results = bm25.retrieve(q_tokens, k=30, show_progress=False)
            bm25_indices = bm25_results.documents[0]
            bm25_doc_ids = [str(chunks_df.iloc[int(idx)]["document_id"]) for idx in bm25_indices]

        # --- B. BGE Semantic Retrieval ---
        with timer() as t_bge:
            q_emb = embedding_model.encode(
                q_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            sim_scores = embeddings @ q_emb
            bge_indices = sim_scores.argsort()[::-1][:30]
            bge_doc_ids = [str(chunks_df.iloc[int(idx)]["document_id"]) for idx in bge_indices]

        # --- C. Hybrid RRF Retrieval ---
        with timer() as t_hybrid:
            rrf_k = 60
            scores = {}
            for rank, idx in enumerate(bm25_indices, start=1):
                idx = int(idx)
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
            for rank, idx in enumerate(bge_indices, start=1):
                idx = int(idx)
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)

            ranked_indices = sorted(scores, key=scores.get, reverse=True)[:30]
            hybrid_doc_ids = [str(chunks_df.iloc[idx]["document_id"]) for idx in ranked_indices]

        candidate_maps = {
            "bm25_only": (bm25_doc_ids, t_bm25["elapsed_ms"]),
            "bge_semantic_only": (bge_doc_ids, t_bge["elapsed_ms"]),
            "hybrid_rrf": (hybrid_doc_ids, t_hybrid["elapsed_ms"]),
        }

        query_record = {
            "query_id": q_id,
            "query": q_text,
            "category": item["category"],
            "target_doc_ids": list(target_doc_ids),
            "system_metrics": {},
        }

        for sys_name in systems:
            retrieved_docs, lat_ms = candidate_maps[sys_name]
            # Deduplicate document IDs in ranked order while preserving first seen rank
            deduped_retrieved = list(dict.fromkeys(retrieved_docs))

            r1 = compute_recall_at_k(deduped_retrieved, target_doc_ids, k=1)
            r3 = compute_recall_at_k(deduped_retrieved, target_doc_ids, k=3)
            r5 = compute_recall_at_k(deduped_retrieved, target_doc_ids, k=5)
            r10 = compute_recall_at_k(deduped_retrieved, target_doc_ids, k=10)
            mrr = compute_mrr(deduped_retrieved, target_doc_ids)
            ndcg = compute_ndcg_at_k(deduped_retrieved, target_doc_ids, k=10)

            results_by_system[sys_name]["recall_1"].append(r1)
            results_by_system[sys_name]["recall_3"].append(r3)
            results_by_system[sys_name]["recall_5"].append(r5)
            results_by_system[sys_name]["recall_10"].append(r10)
            results_by_system[sys_name]["mrr"].append(mrr)
            results_by_system[sys_name]["ndcg_10"].append(ndcg)
            results_by_system[sys_name]["latency_ms"].append(lat_ms)

            query_record["system_metrics"][sys_name] = {
                "recall_1": r1,
                "recall_5": r5,
                "recall_10": r10,
                "mrr": mrr,
                "ndcg_10": ndcg,
                "top_3_retrieved": deduped_retrieved[:3],
                "hit": r10 > 0,
            }

        per_query_details.append(query_record)

    # 3. Aggregate Macro Averages
    summary_metrics = {}
    for sys_name in systems:
        summary_metrics[sys_name] = {
            "recall_at_1": round(float(np.mean(results_by_system[sys_name]["recall_1"])), 4),
            "recall_at_3": round(float(np.mean(results_by_system[sys_name]["recall_3"])), 4),
            "recall_at_5": round(float(np.mean(results_by_system[sys_name]["recall_5"])), 4),
            "recall_at_10": round(float(np.mean(results_by_system[sys_name]["recall_10"])), 4),
            "mrr": round(float(np.mean(results_by_system[sys_name]["mrr"])), 4),
            "ndcg_at_10": round(float(np.mean(results_by_system[sys_name]["ndcg_10"])), 4),
            "mean_latency_ms": round(float(np.mean(results_by_system[sys_name]["latency_ms"])), 2),
        }

    results = {
        "benchmark": "retrieval_hybrid_search",
        "corpus_chunks_evaluated": len(chunks_df),
        "total_benchmark_queries": len(queries),
        "systems_evaluated": systems,
        "summary_metrics": summary_metrics,
        "per_query_results": per_query_details,
    }

    save_results(results, "retrieval_results.json")
    logger.info("Hybrid Search / Retrieval Evaluation complete!")
    logger.info("BM25   -> Recall@5: %.4f | MRR: %.4f | nDCG@10: %.4f | Latency: %.2f ms",
                summary_metrics["bm25_only"]["recall_at_5"], summary_metrics["bm25_only"]["mrr"],
                summary_metrics["bm25_only"]["ndcg_at_10"], summary_metrics["bm25_only"]["mean_latency_ms"])
    logger.info("BGE    -> Recall@5: %.4f | MRR: %.4f | nDCG@10: %.4f | Latency: %.2f ms",
                summary_metrics["bge_semantic_only"]["recall_at_5"], summary_metrics["bge_semantic_only"]["mrr"],
                summary_metrics["bge_semantic_only"]["ndcg_at_10"], summary_metrics["bge_semantic_only"]["mean_latency_ms"])
    logger.info("Hybrid -> Recall@5: %.4f | MRR: %.4f | nDCG@10: %.4f | Latency: %.2f ms",
                summary_metrics["hybrid_rrf"]["recall_at_5"], summary_metrics["hybrid_rrf"]["mrr"],
                summary_metrics["hybrid_rrf"]["ndcg_at_10"], summary_metrics["hybrid_rrf"]["mean_latency_ms"])
    return results


if __name__ == "__main__":
    run_retrieval_evaluation()
