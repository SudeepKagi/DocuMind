# ============================================================
# DocuMind Evaluation: Agent Tool Routing Benchmark
# ============================================================

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Setup paths
EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from common.metrics import compute_routing_metrics
from common.utils import PROJECT_ROOT, ML_DIR, setup_ml_paths, timer, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.agent")


def run_agent_evaluation() -> Dict[str, Any]:
    """
    Evaluates DocuMind Agent rule-based planner tool selection across
    20 realistic user queries covering single-tool and multi-tool scenarios.
    Computes:
    - Tool-selection accuracy
    - Single-tool accuracy
    - Multi-tool routing accuracy
    - Unnecessary/invalid tool-call rate
    """
    logger.info("Starting Agent Tool Routing Evaluation...")

    setup_ml_paths()
    benchmark_path = EVAL_DIR / "agent" / "agent_benchmark.json"

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file missing: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)
    logger.info("Loaded %d Agent benchmark queries", len(benchmarks))

    from documind_agent import plan_tools

    eval_items = []
    latencies_ms = []

    for item in benchmarks:
        q_id = item["id"]
        query = item["query"]
        expected = item["expected_tools"]

        with timer() as plan_timer:
            predicted = plan_tools(query)

        latencies_ms.append(plan_timer["elapsed_ms"])

        eval_items.append({
            "id": q_id,
            "query": query,
            "category": item["category"],
            "expected_tools": expected,
            "predicted_tools": predicted,
            "is_correct": expected == predicted,
            "planning_latency_ms": round(plan_timer["elapsed_ms"], 3),
        })

    metrics = compute_routing_metrics(eval_items)
    metrics["mean_planning_latency_ms"] = round(float(np.mean(latencies_ms)), 3)

    results = {
        "benchmark": "agent_tool_routing",
        "planner": "Rule-Based Intent Classifier & Multi-Tool Orchestrator",
        "total_queries_evaluated": len(benchmarks),
        "metrics": metrics,
        "query_results": eval_items,
    }

    save_results(results, "agent_results.json")
    logger.info("Agent Routing Evaluation complete!")
    logger.info("Tool Selection Accuracy: %.4f | Single-Tool: %.4f | Multi-Tool: %.4f | Unnecessary: %.4f",
                metrics["tool_selection_accuracy"], metrics["single_tool_accuracy"],
                metrics["multi_tool_accuracy"], metrics["unnecessary_tool_call_rate"])
    logger.info("Average Planning Latency: %.3f ms", metrics["mean_planning_latency_ms"])
    return results


if __name__ == "__main__":
    run_agent_evaluation()
