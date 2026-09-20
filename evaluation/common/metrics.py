# ============================================================
# DocuMind Evaluation Framework: Standardized Metrics
# ============================================================

import math
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


# ------------------------------------------------------------
# 1. Classification Metrics
# ------------------------------------------------------------

def compute_classification_metrics(
    y_true: List[Union[str, int]],
    y_pred: List[Union[str, int]],
    labels: Optional[List[Union[str, int]]] = None,
) -> Dict[str, Any]:
    """
    Computes Accuracy, Macro Precision, Macro Recall, Macro F1,
    Weighted F1, Per-class metrics, and Confusion Matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    per_class = {}
    for lbl in labels:
        lbl_str = str(lbl)
        if lbl_str in report:
            per_class[lbl_str] = {
                "precision": float(report[lbl_str]["precision"]),
                "recall": float(report[lbl_str]["recall"]),
                "f1_score": float(report[lbl_str]["f1-score"]),
                "support": int(report[lbl_str]["support"]),
            }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "labels": [str(l) for l in labels],
        "confusion_matrix": cm,
        "total_samples": len(y_true),
    }


# ------------------------------------------------------------
# 2. String & QA Matching Metrics (Exact Match, Token F1, Similarity)
# ------------------------------------------------------------

def normalize_text(text: Optional[str]) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # White space cleanup
    return " ".join(text.split())


def compute_exact_match(prediction: Optional[str], ground_truth: Optional[str]) -> float:
    """Calculates strict normalized Exact Match (1.0 or 0.0)."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: Optional[str], ground_truth: Optional[str]) -> float:
    """Computes SQuAD-style token-level overlap F1 score."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def compute_char_similarity(pred: Optional[str], gt: Optional[str]) -> float:
    """Calculates Levenshtein-based similarity ratio between 0.0 and 1.0."""
    from difflib import SequenceMatcher
    if pred is None or gt is None:
        return 1.0 if pred == gt else 0.0
    s1, s2 = str(pred).strip(), str(gt).strip()
    if not s1 and not s2:
        return 1.0
    return round(SequenceMatcher(None, s1, s2).ratio(), 4)


# ------------------------------------------------------------
# 3. Information Retrieval Metrics (Recall@K, MRR, nDCG@K)
# ------------------------------------------------------------

def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Recall@K: 1.0 if any ground truth item is retrieved in top-K, else 0.0."""
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = any(doc_id in ground_truth_ids for doc_id in top_k)
    return 1.0 if hits else 0.0


def compute_mrr(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Mean Reciprocal Rank (MRR): 1 / rank of first relevant retrieved item."""
    if not ground_truth_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K with binary relevance."""
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in ground_truth_ids:
            dcg += 1.0 / math.log2(i + 2)  # rank is (i + 1), log2(rank + 1) = log2(i + 2)

    # Ideal DCG: best possible DCG where relevant documents appear first
    num_relevant = min(len(ground_truth_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


# ------------------------------------------------------------
# 4. Agent Tool-Routing Metrics
# ------------------------------------------------------------

def compute_routing_metrics(eval_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes tool selection accuracy, single-tool accuracy, multi-tool routing accuracy,
    and unnecessary/invalid tool-call rate.
    """
    total = len(eval_items)
    if total == 0:
        return {}

    exact_matches = 0
    single_total = 0
    single_correct = 0
    multi_total = 0
    multi_correct = 0
    unnecessary_calls = 0

    for item in eval_items:
        expected = item["expected_tools"]
        predicted = item["predicted_tools"]

        # Exact list match
        if expected == predicted:
            exact_matches += 1

        is_multi = len(expected) > 1
        if is_multi:
            multi_total += 1
            if expected == predicted:
                multi_correct += 1
        else:
            single_total += 1
            if expected == predicted:
                single_correct += 1

        # Unnecessary tool calls (tools predicted that are not in expected)
        extra = set(predicted) - set(expected)
        if extra:
            unnecessary_calls += 1

    return {
        "total_queries": total,
        "tool_selection_accuracy": round(exact_matches / total, 4),
        "single_tool_accuracy": round(single_correct / single_total, 4) if single_total else 1.0,
        "multi_tool_accuracy": round(multi_correct / multi_total, 4) if multi_total else 1.0,
        "unnecessary_tool_call_rate": round(unnecessary_calls / total, 4),
    }
