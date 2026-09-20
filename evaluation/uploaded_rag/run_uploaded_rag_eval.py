# ============================================================
# DocuMind Evaluation: Uploaded Document RAG Benchmark
# ============================================================

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Setup paths
EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from common.metrics import compute_token_f1
from common.utils import PROJECT_ROOT, ML_DIR, setup_ml_paths, timer, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.uploaded_rag")


def run_uploaded_rag_evaluation() -> Dict[str, Any]:
    """
    Evaluates ChromaDB-backed uploaded document RAG workflow across
    PDF, DOCX, TXT, and EML documents in single-doc and multi-doc query modes.
    Computes:
    - Retrieval Recall@K
    - Source Document Correctness
    - Answer Correctness (Token F1 & Keyword Grounding)
    - Latency breakdown
    """
    logger.info("Starting Uploaded Document RAG Evaluation...")

    setup_ml_paths()
    benchmark_path = EVAL_DIR / "uploaded_rag" / "uploaded_rag_benchmark.json"

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file missing: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)
    logger.info("Loaded %d Uploaded RAG benchmark items", len(benchmarks))

    # Add backend/packages for chromadb
    sys.path.insert(0, str(PROJECT_ROOT / "backend" / "packages"))
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

    import chromadb
    from sentence_transformers import SentenceTransformer
    from documind_agent import qwen_model, qwen_tokenizer, device

    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "backend" / "storage" / "chroma"))
    collection = chroma_client.get_collection("documind_documents")
    logger.info("Connected to ChromaDB collection with %d items", collection.count())

    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    per_query_results = []
    retrieval_recalls = []
    source_correctnesses = []
    token_f1s = []
    keyword_recalls = []
    latencies_ms = []

    for item in benchmarks:
        q_id = item["id"]
        question = item["question"]
        target_doc_ids = set(item["target_doc_ids"])
        expected_ans = item["expected_answer"]
        expected_keywords = [kw.lower() for kw in item["expected_answer_keywords"]]

        logger.info("Evaluating [%s]: '%s'...", q_id, question)

        # 1. Retrieval Phase
        t0 = time.perf_counter()
        q_emb = bge_model.encode(question, normalize_embeddings=True).tolist()
        
        query_results = collection.query(
            query_embeddings=[q_emb],
            n_results=5,
            where={"user_id": "dev_user_001"},
            include=["documents", "metadatas", "distances"]
        )
        t_ret = (time.perf_counter() - t0) * 1000.0

        retrieved_docs = []
        retrieved_doc_ids = []
        if query_results and query_results.get("documents") and len(query_results["documents"][0]) > 0:
            docs = query_results["documents"][0]
            metas = query_results["metadatas"][0]
            for i in range(len(docs)):
                doc_id = metas[i].get("document_id", "")
                fname = metas[i].get("filename", "")
                retrieved_doc_ids.append(doc_id)
                retrieved_docs.append(f"--- SOURCE (File: {fname}, DocID: {doc_id}) ---\n{docs[i][:1500]}")

        evidence_text = "\n\n".join(retrieved_docs)

        # 2. Generation Phase with Qwen2.5-1.5B
        t1 = time.perf_counter()
        prompt = f"""You are DocuMind, an enterprise document intelligence assistant.
Answer the following question using ONLY the provided evidence. Be concise, precise, and factual.

Evidence:
{evidence_text}

Question:
{question}

Answer:
"""
        inputs = qwen_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_tokens = qwen_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=qwen_tokenizer.eos_token_id,
            )
        generated_answer = qwen_tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        t_gen = (time.perf_counter() - t1) * 1000.0
        total_time_ms = t_ret + t_gen

        # 3. Metrics
        retrieval_hit = any(td in retrieved_doc_ids for td in target_doc_ids)
        retrieval_recalls.append(1.0 if retrieval_hit else 0.0)

        source_correct = any(td in evidence_text.lower() or td in generated_answer.lower() for td in target_doc_ids)
        source_correctnesses.append(1.0 if source_correct else 0.0)

        tf1 = compute_token_f1(generated_answer, expected_ans)
        token_f1s.append(tf1)

        # Keyword recall
        ans_lower = generated_answer.lower()
        matched_kw = sum(1 for kw in expected_keywords if kw in ans_lower)
        kw_recall = round(matched_kw / len(expected_keywords), 4) if expected_keywords else 1.0
        keyword_recalls.append(kw_recall)

        latencies_ms.append(total_time_ms)

        per_query_results.append({
            "id": q_id,
            "question": question,
            "file_type": item["file_type"],
            "query_scope": item["query_scope"],
            "target_doc_ids": list(target_doc_ids),
            "generated_answer": generated_answer,
            "expected_answer": expected_ans,
            "retrieval_hit": retrieval_hit,
            "source_correct": source_correct,
            "token_f1": tf1,
            "keyword_recall": kw_recall,
            "retrieval_latency_ms": round(t_ret, 2),
            "generation_latency_ms": round(t_gen, 2),
            "total_latency_ms": round(total_time_ms, 2),
        })

    results = {
        "benchmark": "uploaded_document_rag",
        "vector_store": "ChromaDB (cosine similarity index)",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "total_queries_evaluated": len(benchmarks),
        "supported_file_types_tested": ["PDF", "DOCX", "TXT", "EML"],
        "metrics": {
            "retrieval_recall_at_5": round(float(np.mean(retrieval_recalls)), 4),
            "source_correctness_accuracy": round(float(np.mean(source_correctnesses)), 4),
            "mean_token_f1": round(float(np.mean(token_f1s)), 4),
            "mean_keyword_grounding_recall": round(float(np.mean(keyword_recalls)), 4),
            "mean_total_latency_ms": round(float(np.mean(latencies_ms)), 2),
        },
        "query_results": per_query_results,
    }

    save_results(results, "uploaded_rag_results.json")
    logger.info("Uploaded Document RAG Evaluation complete!")
    logger.info("Recall@5: %.4f | Source Correctness: %.4f | Token F1: %.4f | Keyword Grounding: %.4f",
                results["metrics"]["retrieval_recall_at_5"],
                results["metrics"]["source_correctness_accuracy"],
                results["metrics"]["mean_token_f1"],
                results["metrics"]["mean_keyword_grounding_recall"])
    logger.info("Mean Total Latency: %.2f ms", results["metrics"]["mean_total_latency_ms"])
    return results


if __name__ == "__main__":
    run_uploaded_rag_evaluation()
