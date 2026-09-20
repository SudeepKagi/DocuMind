# ============================================================
# DocuMind Evaluation Framework: Utilities
# ============================================================

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("documind.evaluation.utils")

# Base Path Resolvers
EVAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVAL_DIR.parent
ML_DIR = PROJECT_ROOT / "ml"
ML_SRC = ML_DIR / "src"
BACKEND_DIR = PROJECT_ROOT / "backend"
RESULTS_DIR = EVAL_DIR / "results"
DOCS_EVAL_DIR = PROJECT_ROOT / "docs" / "evaluation"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_EVAL_DIR.mkdir(parents=True, exist_ok=True)


class CustomJSONEncoder(json.JSONEncoder):
    """Encodes NumPy arrays, integers, floats, and dates cleanly into JSON."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, (datetime)):
            return obj.isoformat()
        return super().default(obj)


def save_results(data: Dict[str, Any], filename: str) -> Path:
    """Save structured evaluation output to evaluation/results/<filename>."""
    filepath = RESULTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=CustomJSONEncoder)
    logger.info("Saved evaluation results to %s", filepath)
    return filepath


def load_results(filename: str) -> Dict[str, Any]:
    """Load previously saved evaluation output."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Result file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer():
    """Execution timer context manager returning elapsed milliseconds."""
    t0 = time.perf_counter()
    res = {"elapsed_ms": 0.0, "elapsed_s": 0.0}
    try:
        yield res
    finally:
        t1 = time.perf_counter()
        res["elapsed_s"] = t1 - t0
        res["elapsed_ms"] = (t1 - t0) * 1000.0


def setup_ml_paths():
    """Safely adds ml/src and backend to sys.path without modifying source files."""
    for p in [ML_SRC, ML_DIR, BACKEND_DIR]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
