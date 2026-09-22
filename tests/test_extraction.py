"""
Unit tests for schema-driven extraction in backend/services/extraction.py
Tests schemas, accounting negative values, relative line item indexing, and negative restraint.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "packages"))

from services.extraction import (
    extraction_engine,
    ExtractionEngine,
    LineItemsSchema,
    PaymentSchema,
    CounterpartySchema,
    ContractTermsSchema,
    ProjectBudgetSchema,
    InvoiceSchema,
)


class TestNumericAndAccountingParsing:
    def test_negative_accounting_formats(self):
        """Verify parsing of various accounting negative formats."""
        cases = [
            ("-$450.00", -450.0),
            ("($450.00)", -450.0),
            ("(450.00)", -450.0),
            ("450.00 CR", -450.0),
            ("450.00", 450.0),
            ("$1,234.56", 1234.56),
        ]
        for raw, expected in cases:
            parsed = extraction_engine.parse_numeric(raw)
            assert parsed == expected, f"Failed for {raw}: got {parsed}, expected {expected}"


SAMPLE_EVIDENCE = """
COMMERCIAL TAX INVOICE
Nexus Global Dynamics LLC
500 Financial Way, 18th Floor, New York, NY 10005
Billed To:
Apex Enterprise Solutions Inc.
100 Tech Park Boulevard, Suite 450, San Jose, CA 95110
Invoice Date: 21 Sep 2026

Line Items:
01 | Cloud Infrastructure Architecture & High-Availability Clusters | 10 | $700.00 | $7,000.00
02 | Enterprise Zero-Trust Security Audit & Compliance Validation | 20 | $250.00 | $5,000.00
03 | High-Throughput API Gateway Integration & Microservices Mesh | 32 | $150.00 | $4,800.00
04 | Distributed Event Streaming Engine Setup & Performance Tuning | 22 | $100.00 | $2,200.00
05 | 24/7 Dedicated SRE Support & Continuous Infrastructure Monitoring | 5 | $200.00 | $1,000.00

Subtotal: $20,000.00
Contract Partner Discount (5%): -$1,000.00
Taxable Baseline: $19,000.00
Sales Tax (8.25%): $1,567.50
Total Due: $20,567.50

Payment Details:
Bank: Global Commercial Reserve
Routing: 021000021
SWIFT: GCRBUS33
Payment Terms: Net 30 days. Contractual late payment fee of 1.5% per month.
"""


class TestSchemaExtraction:
    def test_line_items_schema(self):
        res = extraction_engine.extract(SAMPLE_EVIDENCE, schema=LineItemsSchema())
        items = res.get("line_items", [])
        assert len(items) == 5
        assert items[0]["unit_rate"] == 700.00
        assert items[0]["total_amount"] == 7000.00
        assert items[2]["item_number"] == "03"
        assert items[2]["unit_rate"] == 150.00

    def test_relative_item_targeting(self):
        res = extraction_engine.extract(SAMPLE_EVIDENCE, schema="line_items", target_item_index=3)
        targeted = res.get("targeted_line_item")
        assert targeted is not None
        assert targeted["item_number"] == "03"
        assert targeted["unit_rate"] == 150.00
        assert targeted["total_amount"] == 4800.00

    def test_payment_schema_and_negative_restraint(self):
        res = extraction_engine.extract(SAMPLE_EVIDENCE, schema=PaymentSchema())
        assert res.get("routing_code") == "021000021"
        assert "1.5%" in res.get("penalty_rate", "")
        # IBAN is absent in this US domestic text -> must return 'Not Provided'
        assert res.get("iban") == "Not Provided"

    def test_counterparty_schema(self):
        res = extraction_engine.extract(SAMPLE_EVIDENCE, schema=CounterpartySchema())
        assert "Nexus Global Dynamics LLC" in res.get("issuing_entity", "")
        assert "Apex Enterprise Solutions" in res.get("invoiced_entity", "")
        assert "New York" in res.get("remittance_address", "")
        assert "San Jose" in res.get("recipient_address", "")
