# ============================================================
# DocuMind Evaluation: Document Classification Benchmark
# ============================================================

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup paths
EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from common.metrics import compute_classification_metrics
from common.utils import PROJECT_ROOT, ML_DIR, timer, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.classification")


def clean_model_text(text: str) -> str:
    """
    Standard preprocessing established in ml/notebooks/02_document_classification.ipynb:
    - Strip HTML tags
    - Strip Markdown image syntax
    - Strip OCR page-split markers
    - Normalize whitespace
    """
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = text.replace("<--- Page Split --->", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_classification_evaluation() -> Dict[str, Any]:
    """
    Evaluates DistilBERT document classifier vs TF-IDF + Logistic Regression baseline
    on the official test split (ml/processed/test.csv, 996 documents).

    Uses the model's native production inference pipeline:
    - Preprocessing: clean_model_text (matching 02_document_classification.ipynb)
    - Sliding-window chunking: max_length=512, stride=128 (matching documind_agent.py)
    - Document-level logit pooling: mean of chunk logits across each document
    """
    logger.info("Starting Document Classification Evaluation...")

    test_csv_path = ML_DIR / "processed" / "test.csv"
    train_csv_path = ML_DIR / "processed" / "train.csv"
    model_dir = ML_DIR / "models" / "distilbert_doc_classifier" / "final"

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_csv_path}")

    # 1. Load Data
    test_df = pd.read_csv(test_csv_path)
    logger.info("Loaded %d test samples from %s", len(test_df), test_csv_path)

    # 2. Evaluate TF-IDF + Logistic Regression Baseline
    logger.info("Evaluating TF-IDF + Logistic Regression baseline...")
    train_df = pd.read_csv(train_csv_path)
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(train_df["model_text"])
    X_test_tfidf = vectorizer.transform(test_df["model_text"])

    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_model.fit(X_train_tfidf, train_df["label"])
    
    with timer() as lr_timer:
        baseline_preds = lr_model.predict(X_test_tfidf).tolist()

    baseline_metrics = compute_classification_metrics(
        y_true=test_df["label"].tolist(),
        y_pred=baseline_preds,
        labels=sorted(lr_model.classes_.tolist()),
    )
    baseline_metrics["inference_latency_ms_per_doc"] = round(lr_timer["elapsed_ms"] / len(test_df), 3)

    # 3. Evaluate Fine-Tuned DistilBERT (Native Production Pipeline)
    logger.info("Evaluating DistilBERT from %s...", model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    with open(model_dir / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    label2id = mapping["label2id"]

    # Preprocess test text
    test_df["model_text_clean"] = test_df["model_text"].apply(clean_model_text)

    # Create sliding-window chunks (max_length=512, stride=128) matching documind_agent.py & notebook
    logger.info("Creating sliding-window chunks (max_length=512, stride=128)...")
    chunk_rows = []
    for i in range(len(test_df)):
        text = test_df.iloc[i]["model_text_clean"]
        doc_id = test_df.iloc[i]["document_id"]
        encoded = tokenizer(
            text,
            max_length=512,
            truncation=True,
            stride=128,
            return_overflowing_tokens=True,
        )
        for chunk_id in range(len(encoded["input_ids"])):
            chunk_rows.append({
                "doc_idx": i,
                "document_id": doc_id,
                "input_ids": encoded["input_ids"][chunk_id],
                "attention_mask": encoded["attention_mask"][chunk_id],
            })

    chunk_df = pd.DataFrame(chunk_rows)
    logger.info("Generated %d chunks from %d documents.", len(chunk_df), len(test_df))

    batch_size = 32
    all_logits = []

    with timer() as dl_timer:
        for i in range(0, len(chunk_df), batch_size):
            batch = chunk_df.iloc[i : i + batch_size]
            max_len = max(len(x) for x in batch["input_ids"])
            input_ids = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in batch["input_ids"]]
            attn_mask = [x + [0] * (max_len - len(x)) for x in batch["attention_mask"]]

            inp = torch.tensor(input_ids, dtype=torch.long, device=device)
            attn = torch.tensor(attn_mask, dtype=torch.long, device=device)

            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=attn).logits
                all_logits.append(out.cpu().numpy())

        logits_matrix = np.vstack(all_logits)
        logit_cols = [f"logit_{k}" for k in range(len(id2label))]
        logit_df = pd.DataFrame(logits_matrix, columns=logit_cols)
        logit_df["doc_idx"] = chunk_df["doc_idx"].values

        # Mean logit pooling per document
        mean_doc_logits = logit_df.groupby("doc_idx")[logit_cols].mean()
        mean_doc_logits = mean_doc_logits.reindex(range(len(test_df)))
        pred_ids = mean_doc_logits.values.argmax(axis=1)
        distilbert_preds = [id2label[pid] for pid in pred_ids]

    distilbert_metrics = compute_classification_metrics(
        y_true=test_df["label"].tolist(),
        y_pred=distilbert_preds,
        labels=sorted(list(id2label.values())),
    )
    distilbert_metrics["inference_latency_ms_per_doc"] = round(dl_timer["elapsed_ms"] / len(test_df), 3)

    # 4. Naive Truncation Ablation (Uncleaned Raw Text, Single 512-token Truncation)
    logger.info("Evaluating naive truncation ablation (uncleaned raw text, single 512-token chunk)...")
    naive_preds = []
    raw_texts = test_df["model_text"].tolist()
    with timer() as naive_timer:
        for i in range(0, len(raw_texts), batch_size):
            batch_texts = raw_texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits
                pids = torch.argmax(logits, dim=-1).cpu().numpy()
                naive_preds.extend([id2label[p] for p in pids])

    naive_metrics = compute_classification_metrics(
        y_true=test_df["label"].tolist(),
        y_pred=naive_preds,
        labels=sorted(list(id2label.values())),
    )
    naive_metrics["inference_latency_ms_per_doc"] = round(naive_timer["elapsed_ms"] / len(test_df), 3)

    results = {
        "benchmark": "document_classification",
        "dataset": {
            "path": "ml/processed/test.csv",
            "total_test_samples": len(test_df),
            "classes": sorted(list(id2label.values())),
        },
        "checkpoint": {
            "path": str(model_dir),
            "device": str(device),
            "label_mapping": {str(k): v for k, v in id2label.items()},
        },
        "pipeline_settings": {
            "primary": {
                "description": "Production Agent Pipeline (Cleaned Text + Sliding-Window Chunking + Mean Logit Pooling)",
                "preprocessing": "clean_model_text (HTML/Markdown image removal, page-split removal, whitespace normalization)",
                "max_length": 512,
                "stride": 128,
                "pooling": "mean_chunk_logits",
                "total_chunks_evaluated": len(chunk_df),
            },
            "ablation": {
                "description": "Naïve Truncation Baseline (Uncleaned Raw Text + Single 512-Token Window)",
                "preprocessing": "None (raw model_text)",
                "max_length": 512,
                "stride": None,
                "pooling": "first_window_only",
            },
        },
        "models": {
            "distilbert": {
                "name": "DistilBERT (distilbert-base-uncased)",
                "checkpoint": str(model_dir),
                "device": str(device),
                "metrics": distilbert_metrics,
            },
            "baseline_tfidf_lr": {
                "name": "TF-IDF + Logistic Regression",
                "metrics": baseline_metrics,
            },
        },
        "comparison": {
            "accuracy_delta": round(distilbert_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
            "macro_f1_delta": round(distilbert_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4),
            "weighted_f1_delta": round(distilbert_metrics["weighted_f1"] - baseline_metrics["weighted_f1"], 4),
        },
        "ablations": {
            "distilbert_naive_truncation_raw": {
                "name": "DistilBERT (Naïve Truncation, Uncleaned Raw Text)",
                "description": "Evaluates raw model_text truncated at 512 tokens without preprocessing or chunk pooling",
                "metrics": naive_metrics,
                "notes": "Reproduces earlier 98.80% accuracy / 99.20% macro F1 result caused by uncleaned OCR header artifacts pushing report bodies beyond token 512",
            },
        },
    }

    save_results(results, "classification_results.json")
    logger.info("Classification Evaluation complete!")
    logger.info("DistilBERT (Production Pipeline) Accuracy: %.4f | Macro F1: %.4f", distilbert_metrics["accuracy"], distilbert_metrics["macro_f1"])
    logger.info("DistilBERT (Naïve Ablation)      Accuracy: %.4f | Macro F1: %.4f", naive_metrics["accuracy"], naive_metrics["macro_f1"])
    logger.info("Baseline (TF-IDF + LogReg)       Accuracy: %.4f | Macro F1: %.4f", baseline_metrics["accuracy"], baseline_metrics["macro_f1"])
    return results


if __name__ == "__main__":
    run_classification_evaluation()
