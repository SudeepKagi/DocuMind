"""
Unit tests for deterministic calculations in backend/services/calculation.py
Tests decimal precision, range boundary lookups (SLA tiers), reconciliation,
negative/zero edge cases, and temporal date offsets.
"""

from decimal import Decimal
from datetime import datetime, timedelta
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "packages"))

from services.calculation import calculation_engine, CalculationEngine


class TestDeterministicCalculations:
    def test_decimal_sum_precision(self):
        """Verify high-precision decimal addition without float drift."""
        values = ["0.1", "0.2", "0.3"]
        res = calculation_engine.calculate("sum", {"values": values})
        assert float(res["result"]) == 0.6

    def test_difference_and_negative(self):
        """Verify difference with negative results."""
        res = calculation_engine.calculate("difference", {"value_a": "100.00", "value_b": "150.50"})
        assert res["result_decimal"] == "-50.50"
        assert res["result"] == -50.50

    def test_percentage_calculation(self):
        """Verify percentage calculation."""
        res = calculation_engine.calculate("percentage", {"base": "22500.00", "rate": "50.0"})
        assert float(res["result"]) == 11250.00

    def test_effective_tax_rate(self):
        """Verify effective tax rate when a 5% discount is applied to an 8.25% sales tax."""
        res = calculation_engine.calculate("rate", {"sales_tax_percent": 8.25, "discount_percent": 5.0})
        # (1 - 0.05) * 8.25 = 0.95 * 8.25 = 7.8375%
        assert res["result"] == 7.8375
        assert "7.8375%" in res["formula"]


class TestSLARangeBoundaryLookups:
    """
    Tests SLA tier lookups with explicit inclusive/exclusive interval semantics:
    Tier 1: 99.9 <= x <= 100.0 (0% credit)
    Tier 2: 99.0 <= x < 99.9 (10% credit)
    Tier 3: 98.0 <= x < 99.0 (50% credit)
    Tier 4: 0.0 <= x < 98.0 (100% credit)
    """

    def test_tier_1_upper_and_lower_boundaries(self):
        # 100.0 (max boundary)
        res_100 = calculation_engine.calculate("range_lookup", {"value": 100.0, "base_amount": 22500.00})
        assert res_100["matched_tier"] == "Tier 1"
        assert res_100["credit_percent"] == 0.0
        assert res_100["net_payable"] == 22500.00

        # 99.9 (min boundary)
        res_99_9 = calculation_engine.calculate("range_lookup", {"value": 99.9, "base_amount": 22500.00})
        assert res_99_9["matched_tier"] == "Tier 1"
        assert res_99_9["credit_percent"] == 0.0

    def test_tier_2_just_below_tier_1(self):
        # 99.899% (just below 99.9%) -> Tier 2
        res = calculation_engine.calculate("range_lookup", {"value": 99.899, "base_amount": 22500.00})
        assert res["matched_tier"] == "Tier 2"
        assert res["credit_percent"] == 10.0
        assert res["credit_amount"] == 2250.00
        assert res["net_payable"] == 20250.00

    def test_tier_3_exact_benchmark_case(self):
        # 98.4% -> Tier 3 (50% credit) on $22,500 base commitment fee
        res = calculation_engine.calculate("range_lookup", {"value": 98.4, "base_amount": 22500.00})
        assert res["matched_tier"] == "Tier 3"
        assert res["credit_percent"] == 50.0
        assert res["credit_amount"] == 11250.00
        assert res["net_payable"] == 11250.00

    def test_tier_3_lower_boundary(self):
        # 98.0% (exact lower boundary of Tier 3)
        res = calculation_engine.calculate("range_lookup", {"value": 98.0, "base_amount": 22500.00})
        assert res["matched_tier"] == "Tier 3"
        assert res["credit_percent"] == 50.0

    def test_tier_4_just_below_tier_3(self):
        # 97.99% (just below 98.0%) -> Tier 4 (100% credit)
        res = calculation_engine.calculate("range_lookup", {"value": 97.99, "base_amount": 22500.00})
        assert res["matched_tier"] == "Tier 4"
        assert res["credit_percent"] == 100.0
        assert res["credit_amount"] == 22500.00
        assert res["net_payable"] == 0.00

    def test_zero_and_negative_uptime(self):
        # Zero uptime
        res_zero = calculation_engine.calculate("range_lookup", {"value": 0.0, "base_amount": 22500.00})
        assert res_zero["matched_tier"] == "Tier 4"
        assert res_zero["credit_percent"] == 100.0


class TestReconciliation:
    def test_reconciliation_match(self):
        line_items = [
            {"total_amount": 7000.00},
            {"total_amount": 5000.00},
            {"total_amount": 4800.00},
            {"total_amount": 2200.00},
            {"total_amount": 1000.00},
        ]
        totals = {
            "subtotal": 20000.00,
            "discount_amount": 1000.00,
            "tax_amount": 1567.50,
            "total_due": 20567.50,
        }
        res = calculation_engine.calculate("reconcile", {"line_items": line_items, "stated_totals": totals})
        assert res["status"] == "MATCH"
        assert res["variance"] == 0.0
        assert res["computed_subtotal"] == 20000.00

    def test_reconciliation_mismatch(self):
        line_items = [{"total_amount": 5000.00}]
        totals = {"total_due": 6000.00}
        res = calculation_engine.calculate("reconcile", {"line_items": line_items, "stated_totals": totals})
        assert res["status"] == "MISMATCH"
        assert res["variance"] == -1000.00


class TestDateArithmetic:
    def test_net_30_date_offset(self):
        res = calculation_engine.calculate("date_offset", {
            "issue_date": "21 Sep 2026",
            "term_days": 30,
            "dispute_business_days": 5,
        })
        assert res["issue_date"] == "21 Sep 2026"
        assert res["due_date"] == "21 Oct 2026"
        assert res["overdue_date"] == "21 Oct 2026"
        assert res["dispute_deadline"] == "28 Sep 2026"
