# ============================================================
# DocuMind Master Evaluation Runner & Consolidator
# ============================================================

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from classification.run_classification_eval import run_classification_evaluation
from metadata.run_metadata_eval import run_metadata_evaluation
from retrieval.run_retrieval_eval import run_retrieval_evaluation
from rag.run_rag_eval import run_rag_evaluation
from agent.run_agent_eval import run_agent_evaluation
from uploaded_rag.run_uploaded_rag_eval import run_uploaded_rag_evaluation
from common.utils import save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("documind.eval.runner")


def run_all():
    logger.info("=" * 70)
    logger.info("DOCUMIND ENTERPRISE AI EVALUATION MATRIX SUITE")
    logger.info("Executing comprehensive quantitative evaluation across all 6 components...")
    logger.info("=" * 70)

    start_time = datetime.now(timezone.utc).isoformat()
    summary = {
        "title": "DocuMind Enterprise Document Intelligence Evaluation Matrix",
        "timestamp": start_time,
        "components": {},
    }

    # 1. Document Classification
    try:
        logger.info("\n>>> [1/6] Running Document Classification Evaluation...")
        res1 = run_classification_evaluation()
        summary["components"]["classification"] = {
            "model": "DistilBERT",
            "dataset": "ml/processed/test.csv (996 samples)",
            "accuracy": res1["models"]["distilbert"]["metrics"]["accuracy"],
            "macro_f1": res1["models"]["distilbert"]["metrics"]["macro_f1"],
            "weighted_f1": res1["models"]["distilbert"]["metrics"]["weighted_f1"],
            "baseline_accuracy": res1["models"]["baseline_tfidf_lr"]["metrics"]["accuracy"],
            "baseline_macro_f1": res1["models"]["baseline_tfidf_lr"]["metrics"]["macro_f1"],
            "accuracy_delta": res1["comparison"]["accuracy_delta"],
        }
    except Exception as e:
        logger.error("Classification evaluation failed: %s", e)
        summary["components"]["classification"] = {"error": str(e)}

    # 2. Metadata Extraction
    try:
        logger.info("\n>>> [2/6] Running Metadata Extraction Evaluation...")
        res2 = run_metadata_evaluation()
        summary["components"]["metadata_extraction"] = {
            "model": "LayoutLM Multi-Label + Heuristics",
            "dataset": "DocILE Annotated Invoices",
            "total_instances_evaluated": res2["total_field_instances_evaluated"],
            "token_level_validation_macro_f1": res2["evaluation_scope"]["token_level_validation_macro_f1"],
            "field_exact_match_rate": res2["evaluation_scope"]["field_level_evaluation"]["exact_match_rate"],
            "field_mean_similarity": res2["evaluation_scope"]["field_level_evaluation"]["mean_character_similarity"],
            "field_success_rate_80": res2["evaluation_scope"]["field_level_evaluation"]["success_rate_at_80_threshold"],
        }
    except Exception as e:
        logger.error("Metadata evaluation failed: %s", e)
        summary["components"]["metadata_extraction"] = {"error": str(e)}

    # 3. Hybrid Search / Retrieval
    try:
        logger.info("\n>>> [3/6] Running Retrieval Benchmark Evaluation...")
        res3 = run_retrieval_evaluation()
        summary["components"]["retrieval"] = {
            "corpus_chunks": res3["corpus_chunks_evaluated"],
            "total_queries": res3["total_benchmark_queries"],
            "bm25_only": res3["summary_metrics"]["bm25_only"],
            "bge_semantic_only": res3["summary_metrics"]["bge_semantic_only"],
            "hybrid_rrf": res3["summary_metrics"]["hybrid_rrf"],
        }
    except Exception as e:
        logger.error("Retrieval evaluation failed: %s", e)
        summary["components"]["retrieval"] = {"error": str(e)}

    # 4. RAG / Question Answering
    try:
        logger.info("\n>>> [4/6] Running Grounded RAG QA Evaluation...")
        res4 = run_rag_evaluation()
        summary["components"]["rag_qa"] = {
            "model": "Qwen2.5-1.5B-Instruct",
            "total_questions": res4["total_benchmark_questions"],
            "retrieval_hit_rate": res4["metrics"]["retrieval_hit_rate"],
            "citation_grounding_accuracy": res4["metrics"]["citation_grounding_accuracy"],
            "mean_token_f1": res4["metrics"]["mean_token_f1"],
            "exact_match_rate": res4["metrics"]["exact_match_rate"],
            "hallucination_rate": res4["metrics"]["hallucination_rate"],
            "mean_latency_ms": res4["metrics"]["latency"]["mean_total_latency_ms"],
        }
    except Exception as e:
        logger.error("RAG evaluation failed: %s", e)
        summary["components"]["rag_qa"] = {"error": str(e)}

    # 5. Agent Tool Routing
    try:
        logger.info("\n>>> [5/6] Running Agent Tool Routing Evaluation...")
        res5 = run_agent_evaluation()
        summary["components"]["agent_tool_routing"] = {
            "planner": "Rule-Based Intent Classifier & Multi-Tool Orchestrator",
            "total_queries": res5["total_queries_evaluated"],
            "tool_selection_accuracy": res5["metrics"]["tool_selection_accuracy"],
            "single_tool_accuracy": res5["metrics"]["single_tool_accuracy"],
            "multi_tool_accuracy": res5["metrics"]["multi_tool_accuracy"],
            "unnecessary_tool_call_rate": res5["metrics"]["unnecessary_tool_call_rate"],
            "mean_latency_ms": res5["metrics"]["mean_planning_latency_ms"],
        }
    except Exception as e:
        logger.error("Agent routing evaluation failed: %s", e)
        summary["components"]["agent_tool_routing"] = {"error": str(e)}

    # 6. Uploaded Document RAG
    try:
        logger.info("\n>>> [6/6] Running Uploaded Document RAG Evaluation...")
        res6 = run_uploaded_rag_evaluation()
        summary["components"]["uploaded_rag"] = {
            "vector_store": "ChromaDB (Local)",
            "supported_types_tested": res6["supported_file_types_tested"],
            "retrieval_recall_at_5": res6["metrics"]["retrieval_recall_at_5"],
            "source_correctness_accuracy": res6["metrics"]["source_correctness_accuracy"],
            "mean_token_f1": res6["metrics"]["mean_token_f1"],
            "mean_keyword_grounding_recall": res6["metrics"]["mean_keyword_grounding_recall"],
            "mean_latency_ms": res6["metrics"]["mean_total_latency_ms"],
        }
    except Exception as e:
        logger.error("Uploaded RAG evaluation failed: %s", e)
        summary["components"]["uploaded_rag"] = {"error": str(e)}

    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_results(summary, "evaluation_summary.json")

    logger.info("\n" + "=" * 70)
    logger.info("ALL EVALUATIONS SUCCESSFULLY EXECUTED AND CONSOLIDATED!")
    logger.info("Master summary written to evaluation/results/evaluation_summary.json")
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    run_all()
