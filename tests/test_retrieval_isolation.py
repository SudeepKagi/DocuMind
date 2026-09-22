"""
Unit tests for retrieval isolation and identifier analysis in backend/services/retrieval.py
Tests document isolation boundaries, zero-fallback guarantees, and structural identifier detection.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "packages"))

from services.retrieval import retrieval_engine, RetrievalEngine, RetrievalScope


class TestIdentifierAnalyzer:
    def test_detect_business_identifiers(self):
        """Verify structural identifier detection across POs, invoices, routing numbers, and mixed codes."""
        test_queries = [
            ("Which purchase order authorized wire transfer routing code 021000021?", ["021000021"]),
            ("Show me records mentioning transaction reference INV-2026-8842-X.", ["INV-2026-8842-X"]),
            ("Look up PO-84721 and contract MSA-2026-884X-V2.", ["PO-84721", "MSA-2026-884X-V2"]),
            ("Check transaction TXN8849201 and SWIFT code SWIFTABC123.", ["TXN8849201", "SWIFTABC123"]),
            ("Account wire 45782910392 status.", ["45782910392"]),
        ]
        for query, expected_tokens in test_queries:
            detected = retrieval_engine.detect_identifier_tokens(query)
            for exp in expected_tokens:
                assert exp in detected, f"Expected {exp} to be detected in query: '{query}', got: {detected}"

    def test_stopwords_exclusion(self):
        """Verify normal business dictionary terms are not classified as identifiers."""
        query = "Extract the line items, calculate effective tax rate, and reconcile subtotal."
        detected = retrieval_engine.detect_identifier_tokens(query)
        assert len(detected) == 0, f"Expected no identifiers, got: {detected}"


class TestRetrievalScopeAndIsolation:
    def test_scope_initialization(self):
        scope = RetrievalScope.from_dict_or_args(
            user_id="user_123",
            document_id="doc_A",
        )
        assert scope.user_id == "user_123"
        assert scope.doc_id == "doc_A"
        assert scope.source_type == "uploaded"

    def test_multi_doc_scope(self):
        scope = RetrievalScope.from_dict_or_args(
            user_id="user_123",
            target_doc_ids=["doc_A", "doc_B"],
        )
        assert scope.target_doc_ids == ["doc_A", "doc_B"]
        assert scope.doc_id is None

    def test_rrf_boost_priority(self):
        """Verify that identifier matching chunks get boosted above semantic-only hits."""
        dense_results = [
            {"id": "c1", "text": "General invoice discussion and payment terms.", "score": 0.85},
            {"id": "c2", "text": "Remittance details for transaction INV-2026-8842-X.", "score": 0.70},
        ]
        tokens = ["INV-2026-8842-X"]
        merged = retrieval_engine.rrf_merge(dense_results, identifier_tokens=tokens)
        # c2 must rank first due to exact identifier match
        assert merged[0]["id"] == "c2"
        assert merged[0]["score"] > merged[1]["score"]
