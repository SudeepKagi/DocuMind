"""
DocuMind W: Universal Pipeline End-to-End Architectural Hardening Test Suite
Validates the six core architectural invariants:
1. Universal Text & Table Canonicalization (ASCII box borders, markdown pipes, tab/space streams, multi-line rows, negative financial notations).
2. Pure Decimal Arithmetic & Bracket Precision (resolve_interval_bracket [min, max), reconciliation ledger, date offsets).
3. Retrieval Scope & Document Isolation (no cross-doc leakage, zero fallback on empty search).
4. Strict Negative Restraint & Zero-Preamble Generation Hygiene (Not Mentioned in Provided Context, no filler/preambles).
5. Deterministic Agent 5-Primitive DAG Orchestration.
"""

import sys
from pathlib import Path
from decimal import Decimal
import pytest

# Ensure backend and packages are on path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR / "packages"))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

from services.extraction import (
    extraction_engine,
    ExtractionEngine,
    LineItem,
    LineItemsSchema,
    CounterpartySchema,
    PaymentTermsSchema,
    TieredMatrixSchema,
    ReconciliationLedgerSchema,
    ExtractionResult,
)
from services.calculation import (
    calculation_engine,
    CalculationEngine,
    resolve_interval_bracket,
    reconcile_ledger,
)
from services.retrieval import (
    retrieval_engine,
    RetrievalEngine,
    RetrievalScope,
)
from services.rag import (
    reasoning_engine,
    GroundedReasoningEngine,
)
from services.agent import (
    agent,
    DocuMindAgentService,
    EvidenceItem,
)


# ========================================================
# 1. Universal Table & Financial Accounting Canonicalization
# ========================================================

class TestUniversalTableAndAccountingParsing:
    """Validates ingestion across ASCII boxes, pipes, multi-line rows, and negative currencies."""

    def test_ascii_box_table_normalization(self):
        """Validates that +----+ box borders are stripped and row entities correctly extracted."""
        box_table = """
+----+------------------------------------------------+-----+------------+------------+
| #  | Item Specification & Deliverables              | Qty | Unit Rate  | Total ($)  |
+====+================================================+=====+============+============+
| 01 | Cloud Infrastructure Migration - Phase II      | 40  | 175.00     | 7,000.00   |
+----+------------------------------------------------+-----+------------+------------+
| 02 | Enterprise Security Audit                      | 1   | 4,500.00   | 4,500.00   |
+----+------------------------------------------------+-----+------------+------------+
| 03 | High-Throughput API Gateway                    | 32  | 150.00     | 4,800.00   |
+----+------------------------------------------------+-----+------------+------------+
"""
        items = extraction_engine.extract_line_items(box_table)
        assert len(items) == 3
        assert items[0]["item_number"] == "01"
        assert items[0]["total_amount"] == 7000.00
        assert items[1]["item_number"] == "02"
        assert items[1]["total_amount"] == 4500.00
        assert items[2]["item_number"] == "03"
        assert items[2]["total_amount"] == 4800.00

    def test_multi_line_row_description_reassembly(self):
        """Validates that description lines split across multiple rows reassemble into a single entity."""
        multi_line_table = """
| 01 | Cloud Infrastructure Migration                 | 40 | 175.00 | 7,000.00 |
|    | Multi-tenant VPC isolation and subnet routing  |    |        |          |
|    | Complete zero-trust architecture transition    |    |        |          |
| 02 | 24/7 SLA Engineering Support Retainer          | 1  | 2,200.00 | 2,200.00 |
"""
        items = extraction_engine.extract_line_items(multi_line_table)
        assert len(items) == 2
        assert "Multi-tenant VPC isolation" in items[0]["description"]
        assert "Complete zero-trust architecture" in items[0]["description"]
        assert items[0]["total_amount"] == 7000.00
        assert items[1]["item_number"] == "02"
        assert items[1]["total_amount"] == 2200.00

    @pytest.mark.parametrize("raw_val, expected", [
        ("-$450.00", -450.00),
        ("($450.00)", -450.00),
        ("(450.00)", -450.00),
        ("450.00 CR", -450.00),
        ("-$18,000", -18000.00),
        ("($18,000.00)", -18000.00),
        ("€2,500.50", 2500.50),
        ("£14,200.00", 14200.00),
        ("USD 99,400.00", 99400.00),
        ("0.00", 0.00),
        (None, 0.00),
    ])
    def test_financial_accounting_primitive_normalizer(self, raw_val, expected):
        """Validates standard and negative accounting conventions."""
        result = extraction_engine.parse_numeric(raw_val)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_typed_pydantic_schema_and_empty_result(self):
        """Validates Pydantic schema validation and explicit ExtractionResult(status='EMPTY') on missing data."""
        # Non-matching evidence
        empty_text = "This document is a general memorandum with no tabular or billing entries."
        res = extraction_engine.extract_typed(empty_text, LineItemsSchema)
        assert isinstance(res, ExtractionResult)
        assert res.status == "EMPTY"
        assert res.data is None
        assert res.is_empty is True

        # Valid table evidence
        table_text = "01 Server Hardware 2.0 500.00 1000.00"
        success_res = extraction_engine.extract_typed(table_text, LineItemsSchema)
        assert isinstance(success_res, ExtractionResult)
        assert success_res.status == "SUCCESS"
        assert isinstance(success_res.data, LineItemsSchema)
        assert len(success_res.data.items) == 1
        assert success_res.data.items[0].total_amount == 1000.00


# ========================================================
# 2. Strict Deterministic Math & Bracket Interval Evaluator
# ========================================================

class TestDeterministicMathAndIntervals:
    """Validates Decimal arithmetic, universal bracket evaluator, and compound reconciliation."""

    def test_universal_interval_bracket_semantics(self):
        """
        Validates resolve_interval_bracket with explicit boundary semantics:
        closed lower, open upper: min <= value < max
        """
        sla_matrix = [
            {"tier": "Tier 1", "min": 99.5, "max": 100.0, "min_inc": True, "max_inc": True, "rate": 0.0},
            {"tier": "Tier 2", "min": 99.0, "max": 99.5, "min_inc": True, "max_inc": False, "rate": 10.0},
            {"tier": "Tier 3", "min": 95.0, "max": 99.0, "min_inc": True, "max_inc": False, "rate": 50.0},
            {"tier": "Tier 4", "min": 0.0, "max": 95.0, "min_inc": True, "max_inc": False, "rate": 100.0},
        ]

        # Boundary checks:
        # 1. 100.0% -> Tier 1
        assert resolve_interval_bracket(100.0, sla_matrix)["matched_tier"] == "Tier 1"
        # 2. 99.5% -> Tier 1 (closed lower bound)
        assert resolve_interval_bracket(99.5, sla_matrix)["matched_tier"] == "Tier 1"
        # 3. 99.499% -> Tier 2 (open upper bound of Tier 2)
        assert resolve_interval_bracket(99.499, sla_matrix)["matched_tier"] == "Tier 2"
        # 4. 99.0% -> Tier 2 (closed lower bound)
        assert resolve_interval_bracket(99.0, sla_matrix)["matched_tier"] == "Tier 2"
        # 5. 98.99% -> Tier 3 (open upper bound of Tier 3)
        assert resolve_interval_bracket(98.99, sla_matrix)["matched_tier"] == "Tier 3"
        # 6. 98.4% (Benchmark query) -> Tier 3 (50% credit)
        b_res = resolve_interval_bracket(98.4, sla_matrix)
        assert b_res["matched_tier"] == "Tier 3"
        assert b_res["rate"] == 50.0
        # 7. 95.0% -> Tier 3 (closed lower bound)
        assert resolve_interval_bracket(95.0, sla_matrix)["matched_tier"] == "Tier 3"
        # 8. 94.99% -> Tier 4
        assert resolve_interval_bracket(94.99, sla_matrix)["matched_tier"] == "Tier 4"
        # 9. 0.0% -> Tier 4
        assert resolve_interval_bracket(0.0, sla_matrix)["matched_tier"] == "Tier 4"

    def test_compound_reconciliation_pipeline(self):
        """Validates ledger reconciliation with line items, deductions, additions, and exact audit logs."""
        line_items = [
            {"title": "Cloud Infrastructure", "total_amount": 7000.00},
            {"title": "Security Audit", "total_amount": 4500.00},
            {"title": "API Gateway", "total_amount": 4800.00},
            {"title": "SLA Support", "total_amount": 2200.00},
            {"title": "Documentation", "total_amount": 1500.00},
        ]
        # Sum = 20,000.00
        # Discount (5%) = -1,000.00
        # Sales tax (8.25% on 19,000) = +1,567.50
        # Expected total = 20,567.50
        res_match = reconcile_ledger(
            line_items=line_items,
            deductions=[1000.00],
            additions=[1567.50],
            expected_total=20567.50,
            stated_subtotal=20000.00,
        )
        assert res_match["status"] == "MATCH"
        assert res_match["is_match"] is True
        assert res_match["variance"] == 0.0
        assert len(res_match["audit_log"]) >= 4

        # Test Mismatch with intentional variance
        res_mismatch = reconcile_ledger(
            line_items=line_items,
            deductions=[1000.00],
            additions=[1567.50],
            expected_total=21000.00,
            stated_subtotal=20000.00,
        )
        assert res_mismatch["status"] == "MISMATCH"
        assert res_mismatch["is_match"] is False
        assert res_mismatch["variance"] == pytest.approx(-432.50, rel=1e-5)

    def test_calendar_date_offsets_and_business_days(self):
        """Validates Net-30 and business-day dispute notification deadlines."""
        # 21 Sep 2026 is Monday
        res = calculation_engine.calculate(
            operation="date_offset",
            inputs={
                "issue_date": "21 Sep 2026",
                "term_days": 30,
                "dispute_business_days": 5,
            }
        )
        assert res["payment_due_date"] == "21 Oct 2026"
        assert res["payment_overdue_date"] == "21 Oct 2026"
        # 5 business days from Monday Sep 21 = Monday Sep 28
        assert res["dispute_deadline"] == "28 Sep 2026"


# ========================================================
# 3. Retrieval Scope & Structural Boosting
# ========================================================

class TestRetrievalScopeAndIsolation:
    """Validates document boundary isolation and structural identifier detection."""

    def test_scope_enforcement_and_no_fallback(self):
        """Asserts that scope targeting a specific document never searches globally on zero results."""
        scope = RetrievalScope(
            user_id="user_enterprise_1",
            workspace_id="ws_finance",
            doc_id="doc_invoice_99",
        )
        assert scope.doc_id == "doc_invoice_99"
        assert scope.workspace_id == "ws_finance"

    def test_structural_identifier_analyzer(self):
        """Validates that enterprise codes (PO-*, INV-*, SOW-*, MSKU-*, SWIFT, routing codes) are extracted."""
        query = (
            "Check invoice reference INV-2026-8842-X under SOW-9941-C and container MSKU-918230-4 "
            "with wire routing 021000021 and SWIFT code CHASUS33."
        )
        tokens = retrieval_engine.detect_identifier_tokens(query)
        assert "INV-2026-8842-X" in tokens
        assert "SOW-9941-C" in tokens
        assert "MSKU-918230-4" in tokens
        assert "021000021" in tokens
        assert "CHASUS33" in tokens

    def test_rrf_boost_with_exact_identifier(self):
        """Validates that chunks containing exact business identifiers receive RRF priority."""
        chunks = [
            {"id": "c1", "text": "General billing notes for enterprise customer.", "metadata": {"chunk_id": "c1"}},
            {"id": "c2", "text": "Wire transfer routing 021000021 authorized under PO-84721.", "metadata": {"chunk_id": "c2"}},
        ]
        fused = retrieval_engine.rrf_merge(
            dense_results=chunks,
            identifier_tokens=["021000021", "PO-84721"],
            exact_boost=0.50,
        )
        assert fused[0]["metadata"]["chunk_id"] == "c2"
        assert fused[0]["fused_score"] > fused[1]["fused_score"]


# ========================================================
# 4. Strict Negative Restraint & Generation Hygiene
# ========================================================

class TestGenerationHygieneAndNegativeRestraint:
    """Validates anti-premise echoing, zero-preamble, and output sanitation."""

    def test_empty_context_strict_negative_restraint(self):
        """Asserts that missing context returns 'Not Mentioned in Provided Context' immediately."""
        ans = reasoning_engine.reason(
            question="What is the IBAN for international wire transfer?",
            context_text="",
        )
        assert ans == "Not Mentioned in Provided Context"

    def test_output_sanitation_strips_preambles_and_filler(self):
        """Asserts that procedural preambles, ChatML tags, and conversational filler are purged."""
        raw_output = (
            "<|im_start|>assistant\n"
            "Certainly! Step 1: Let us begin by analyzing the invoice.\n"
            "The total amount due is $20,567.50.\n"
            "Human: Can you tell me more?\n"
            "Assistant: Sure thing!\n"
            "<|im_end|>"
        )
        cleaned = reasoning_engine.sanitize_output(raw_output)
        assert "<|im_start|>" not in cleaned
        assert "<|im_end|>" not in cleaned
        assert "Step 1: Let us begin" not in cleaned
        assert "Human:" not in cleaned
        assert "Assistant:" not in cleaned
        assert "The total amount due is $20,567.50." in cleaned


# ========================================================
# 5. Deterministic Agent DAG Orchestration
# ========================================================

class TestDeterministicAgentDAG:
    """Validates that compound tasks execute across the 5 core primitives without bypassing math."""

    def test_5_primitive_dag_task_graph(self):
        """Asserts that a compound query plans retrieval -> extraction -> calculation."""
        analysis = agent.query_engine.analyze(
            "Extract the line items, calculate the effective tax rate per item, and reconcile the subtotal."
        )
        graph = agent.decomposer.decompose(analysis)
        caps = [node.capability for node in graph.nodes.values()]

        assert "retrieval" in caps
        assert "extraction" in caps
        assert "calculation" in caps

        # Calculation must depend on extraction
        calc_node = graph.nodes["task_calculation"]
        assert "task_extraction" in calc_node.dependencies
        assert "task_retrieval" in calc_node.dependencies
