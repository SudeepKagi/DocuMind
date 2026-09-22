"""
DocuMind W: Generalized Agent Stress Test Suite Evaluator
Evaluates the 5 Core Capabilities (CLASSIFICATION, RETRIEVAL, EXTRACTION, CALCULATION, REASONING)
across compound queries, conversational masks, relative item targeting, deterministic math,
strict negative restraint, and temporal reasoning.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List

# Setup paths
EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))
PROJECT_ROOT = EVAL_DIR.parent

BACKEND_DIR = PROJECT_ROOT / "backend"
PACKAGES_DIR = BACKEND_DIR / "packages"
ML_SRC = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(PACKAGES_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ML_SRC))

from common.utils import timer, save_results
from services.agent import agent, Evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.stress")


def run_stress_evaluation() -> Dict[str, Any]:
    """
    Executes the generalized stress test suite covering compound queries,
    conversational masks, relative item targeting, deterministic math,
    strict negative restraint, and temporal reasoning across the 5 Core Primitives.
    """
    logger.info("Starting Simplified 5-Primitive Agent Stress Test Evaluation...")

    benchmark_path = EVAL_DIR / "stress_tests" / "stress_benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file missing: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)
    logger.info("Loaded %d Generalized Stress Benchmark items", len(benchmarks))

    # Read invoice text from uploaded storage for document context
    import chromadb
    chroma_client = chromadb.PersistentClient(path=str(BACKEND_DIR / "storage" / "chroma"))
    col = chroma_client.get_collection("documind_documents")
    chunk_data = col.get(ids=["doc_43d7e12f_0", "doc_43d7e12f_doc_43d7e12f_0"])
    doc_text = ""
    if chunk_data and chunk_data.get("documents") and len(chunk_data["documents"]) > 0:
        doc_text = chunk_data["documents"][0]
    logger.info("Loaded document context for stress evaluation (length: %d chars)", len(doc_text))

    eval_items = []
    dag_accuracies = []
    calculation_matches = []
    negative_restraint_passes = []
    keyword_recalls = []
    latencies_ms = []

    for item in benchmarks:
        q_id = item["id"]
        prompt_id = item.get("prompt_id", "")
        query = item["query"]
        category = item["category"]
        expected_caps = item.get("expected_capabilities", [])
        expected_kw = item.get("expected_answer_keywords", [])

        t0 = time.time()

        # Step 1: Query Understanding & Task Graph
        analysis = agent.query_engine.analyze(query)
        graph = agent.decomposer.decompose(analysis)
        actual_caps = [node.capability for node in graph.nodes.values()]

        # Check DAG capability coverage across the 5 core primitives
        dag_match = all(cap in actual_caps for cap in expected_caps)
        dag_accuracies.append(1.0 if dag_match else 0.0)

        # Step 2: Run Agent with Initial Evidence
        ev_obj = Evidence(
            document_id="doc_43d7e12f",
            filename="Commercial Tax Invoice.pdf",
            page=1,
            chunk_id="0",
            source_text=doc_text,
            score=0.98,
        )

        executed_state = agent.run(
            query=query,
            user_id="test_user",
            initial_evidence=[ev_obj],
        )

        elapsed_ms = (time.time() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        final_ans = executed_state.final_answer

        # Verification Checks:
        # Check Calculation / Numeric ground truth
        calc_ok = True
        if "expected_effective_tax_rate" in item:
            exp_rate = item["expected_effective_tax_rate"]
            rate_str = f"{exp_rate:.4f}%"
            if rate_str not in final_ans:
                calc_ok = False

        if "expected_reconciliation_status" in item:
            if item["expected_reconciliation_status"] not in final_ans:
                calc_ok = False

        if "expected_unit_rate" in item:
            rate_s = f"${item['expected_unit_rate']:,.2f}"
            if rate_s not in final_ans:
                calc_ok = False

        if "expected_tier" in item:
            if item["expected_tier"] not in final_ans:
                calc_ok = False

        if "expected_credit_amount" in item:
            cred_s = f"{item['expected_credit_amount']:,.2f}"
            if cred_s not in final_ans:
                calc_ok = False

        if "expected_net_payable" in item:
            net_s = f"{item['expected_net_payable']:,.2f}"
            if net_s not in final_ans:
                calc_ok = False

        calculation_matches.append(1.0 if calc_ok else 0.0)

        # Check Negative Restraint
        neg_ok = True
        if item.get("expected_answer_state") == "NOT_PROVIDED" or item.get("expected_iban_state") == "NOT_PROVIDED":
            if "Not Provided" not in final_ans:
                neg_ok = False

        if "prohibited_hallucinations" in item:
            for proh in item["prohibited_hallucinations"]:
                if proh.lower() in final_ans.lower():
                    neg_ok = False
                    break

        negative_restraint_passes.append(1.0 if neg_ok else 0.0)

        # Check Keyword Grounding
        kw_hits = sum(1 for kw in expected_kw if kw.lower() in final_ans.lower())
        kw_recall = (kw_hits / len(expected_kw)) if expected_kw else 1.0
        keyword_recalls.append(kw_recall)

        is_success = dag_match and calc_ok and neg_ok and (kw_recall >= 0.70)

        logger.info(
            "[%s] Prompt %s (%s): DAG=%s, Calc=%s, NegRestraint=%s, KW=%.2f, Latency=%.1fms",
            "PASS" if is_success else "FAIL",
            prompt_id,
            q_id,
            "OK" if dag_match else "FAIL",
            "OK" if calc_ok else "FAIL",
            "OK" if neg_ok else "FAIL",
            kw_recall,
            elapsed_ms,
        )

        eval_items.append({
            "id": q_id,
            "prompt_id": prompt_id,
            "category": category,
            "query": query,
            "clean_query": analysis.clean_query,
            "dag_capabilities": actual_caps,
            "dag_match": dag_match,
            "calculation_match": calc_ok,
            "negative_restraint_pass": neg_ok,
            "keyword_recall": round(kw_recall, 4),
            "latency_ms": round(elapsed_ms, 2),
            "status": "PASS" if is_success else "FAIL",
            "final_answer_snippet": final_ans[:200],
        })

    total = len(eval_items)
    passes = sum(1 for it in eval_items if it["status"] == "PASS")
    overall_pass_rate = (passes / total) if total else 0.0

    summary = {
        "total_stress_tests": total,
        "passed_stress_tests": passes,
        "overall_pass_rate": round(overall_pass_rate, 4),
        "dag_capability_accuracy": round(sum(dag_accuracies) / total, 4) if total else 0.0,
        "calculation_exact_match_rate": round(sum(calculation_matches) / total, 4) if total else 0.0,
        "negative_restraint_compliance_rate": round(sum(negative_restraint_passes) / total, 4) if total else 0.0,
        "mean_keyword_grounding_recall": round(sum(keyword_recalls) / total, 4) if total else 0.0,
        "mean_latency_ms": round(sum(latencies_ms) / total, 2) if total else 0.0,
        "items": eval_items,
    }

    save_results(summary, "stress_test_results.json")

    logger.info("=" * 50)
    logger.info("SIMPLIFIED 5-PRIMITIVE STRESS TEST SUITE SUMMARY")
    logger.info("=" * 50)
    logger.info("Overall Stress Pass Rate: %.2f%%", overall_pass_rate * 100)
    logger.info("DAG Capability Accuracy:  %.2f%%", summary["dag_capability_accuracy"] * 100)
    logger.info("Calculation Exact Match:  %.2f%%", summary["calculation_exact_match_rate"] * 100)
    logger.info("Negative Restraint Rate:  %.2f%%", summary["negative_restraint_compliance_rate"] * 100)
    logger.info("Mean Keyword Grounding:   %.2f%%", summary["mean_keyword_grounding_recall"] * 100)
    logger.info("Mean Latency:             %.2f ms", summary["mean_latency_ms"])
    logger.info("=" * 50)

    return summary


if __name__ == "__main__":
    run_stress_evaluation()
