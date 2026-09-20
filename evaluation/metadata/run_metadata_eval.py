# ============================================================
# DocuMind Evaluation: Metadata Extraction Benchmark
# ============================================================

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Setup paths
EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from common.metrics import compute_char_similarity, compute_exact_match, compute_token_f1
from common.utils import PROJECT_ROOT, ML_DIR, timer, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.metadata")


def run_metadata_evaluation() -> Dict[str, Any]:
    """
    Evaluates LayoutLM multi-label metadata extraction on invoice documents.
    Computes both token-level and field-level metrics across the 7 core invoice fields:
    - vendor_name
    - vendor_address
    - customer_billing_name
    - customer_billing_address
    - date_issue
    - amount_total_gross
    - amount_due
    """
    logger.info("Starting Metadata Extraction Evaluation...")

    quality_csv_path = ML_DIR / "processed" / "invoice_field_quality.csv"
    thresholds_path = ML_DIR / "models" / "layoutlm_multilabel" / "final" / "field_thresholds.json"

    if not quality_csv_path.exists():
        raise FileNotFoundError(f"Field quality data not found at {quality_csv_path}")

    # Load field thresholds configuration
    thresholds = {}
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    # 1. Load Field Quality Test Set
    df = pd.read_csv(quality_csv_path)
    logger.info("Loaded %d field instances across %d documents", len(df), df["document_id"].nunique())

    target_fields = [
        "vendor_name",
        "vendor_address",
        "customer_billing_name",
        "customer_billing_address",
        "date_issue",
        "amount_total_gross",
        "amount_due",
    ]

    # Filter to target invoice fields
    filtered_df = df[df["field"].isin(target_fields)].copy()

    per_field_stats = {}
    all_similarities = []
    all_exact_matches = []
    all_success_80 = []
    all_token_f1s = []

    with timer() as eval_timer:
        for field in target_fields:
            field_df = filtered_df[filtered_df["field"] == field]
            if field_df.empty:
                continue

            field_exact_matches = []
            field_similarities = []
            field_success_80 = []
            field_token_f1s = []

            for _, row in field_df.iterrows():
                gt_val = str(row["value"]) if pd.notna(row["value"]) else ""
                pred_val = str(row["matched_text"]) if pd.notna(row["matched_text"]) else ""

                sim = float(row["similarity"]) if "similarity" in row and pd.notna(row["similarity"]) else compute_char_similarity(pred_val, gt_val)
                em = compute_exact_match(pred_val, gt_val)
                tf1 = compute_token_f1(pred_val, gt_val)

                field_similarities.append(sim)
                field_exact_matches.append(em)
                field_success_80.append(1.0 if sim >= 0.80 else 0.0)
                field_token_f1s.append(tf1)

            mean_sim = round(float(np.mean(field_similarities)), 4)
            em_rate = round(float(np.mean(field_exact_matches)), 4)
            succ_80 = round(float(np.mean(field_success_80)), 4)
            mean_tf1 = round(float(np.mean(field_token_f1s)), 4)

            per_field_stats[field] = {
                "evaluated_instances": len(field_df),
                "exact_match_accuracy": em_rate,
                "mean_similarity": mean_sim,
                "success_rate_80": succ_80,
                "token_f1": mean_tf1,
                "decision_threshold": thresholds.get(field, 0.90),
            }

            all_similarities.extend(field_similarities)
            all_exact_matches.extend(field_exact_matches)
            all_success_80.extend(field_success_80)
            all_token_f1s.extend(field_token_f1s)

    macro_sim = round(float(np.mean(all_similarities)), 4)
    macro_em = round(float(np.mean(all_exact_matches)), 4)
    macro_succ_80 = round(float(np.mean(all_success_80)), 4)
    macro_token_f1 = round(float(np.mean(all_token_f1s)), 4)

    results = {
        "benchmark": "metadata_extraction",
        "model": "LayoutLM Multi-Label Token Classifier + Spatial Extraction",
        "checkpoint": "ml/models/layoutlm_multilabel/final",
        "dataset": "DocILE Annotated Benchmark / invoice_field_quality.csv",
        "total_documents_evaluated": int(filtered_df["document_id"].nunique()),
        "total_field_instances_evaluated": len(filtered_df),
        "evaluation_scope": {
            "token_level_validation_macro_f1": 0.8746,  # Empirically measured in notebook 03 validation
            "field_level_evaluation": {
                "exact_match_rate": macro_em,
                "mean_character_similarity": macro_sim,
                "success_rate_at_80_threshold": macro_succ_80,
                "macro_token_f1": macro_token_f1,
            }
        },
        "per_field_metrics": per_field_stats,
        "elapsed_time_seconds": round(eval_timer["elapsed_s"], 3),
    }

    save_results(results, "metadata_results.json")
    logger.info("Metadata Extraction Evaluation complete!")
    logger.info("Macro Exact Match: %.4f | Mean Similarity: %.4f | Success @ 0.80: %.4f", macro_em, macro_sim, macro_succ_80)
    return results


if __name__ == "__main__":
    run_metadata_evaluation()
