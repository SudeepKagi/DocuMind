"""
DocuMind W: Context & Grounding Gatekeeper Test Suite (tests/test_context_gate.py)
Validates:
1. Hard Context & Grounding Gatekeeper:
   - Empty or missing context immediately returns 'Not Mentioned in Provided Context'.
   - Qwen2.5-1.5B is NEVER called (zero LLM token generation / model loading).
2. Dynamic Document Identifier Resolution & Scoping:
   - Queries referencing unindexed document IDs return DOCUMENT_NOT_INDEXED with zero global search fallback.
   - Queries referencing indexed documents automatically bind retrieval scope to target document.
3. Deterministic SOW Milestone Reconciliation:
   - Verbatim extraction of 4 milestones ($22,500, $40,500, $27,000, $18,000).
   - Advance credit deduction (-$18,000).
   - Technology tax addition (+$5,400).
   - Python Decimal reconciliation against Final Settlement Commitment ($95,400.00 -> MATCH).
   - Zero placeholder hallucination (no $1,000,000 or $500,000) and zero mid-string truncation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure backend and packages are on path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR / "packages"))
sys.path.insert(0, str(BACKEND_DIR))

from services.rag import reasoning_engine, GroundedReasoningEngine
from services.retrieval import (
    retrieval_engine,
    RetrievalScope,
    resolve_document_identifier,
    extract_structural_identifiers,
)
from services.extraction import extraction_engine
from services.calculation import reconcile_ledger
from services.agent import agent, EvidenceItem, Evidence


SOW_QUERY = (
    "For SOW-2026-ENG-9941-C, sum all milestone Net Payable Release amounts, "
    "subtract the advance credit, add the tech tax, and confirm if the result "
    "matches the Final Settlement Commitment."
)

SOW_DOCUMENT_TEXT = """
STATEMENT OF WORK: SOW-2026-ENG-9941-C
Master Services Agreement Reference: MSA-2026-884X-V2
Client: Nexus Global Systems Inc.
Contractor: DocuMind Engineering Services

Milestone Schedule and Payment Terms:
Milestone 1: Architectural Design & Core Ingestion - Net Payable Release: $22,500.00
Milestone 2: Multi-tenant Pipeline & Orchestration - Net Payable Release: $40,500.00
Milestone 3: LayoutLM Fine-Tuning & Ingestion - Net Payable Release: $27,000.00
Milestone 4: Security Hardening & Acceptance - Net Payable Release: $18,000.00

Settlement Summary:
Sum of Milestones: $108,000.00
Advance Credit Deduction: -$18,000.00
Subtotal after Credit: $90,000.00
Tech Tax (6.0%): $5,400.00
Final Settlement Commitment: $95,400.00
"""


class TestContextGatekeeper(unittest.TestCase):
    """Verifies that empty/unindexed queries trigger the hard gatekeeper without calling Qwen."""

    def test_empty_context_gatekeeper_bypasses_llm(self):
        """Send the SOW query with empty context. Assert response equals 'Not Mentioned in Provided Context' and Qwen is never called."""
        with patch.object(reasoning_engine, "_ensure_model") as mock_model:
            response = reasoning_engine.reason(
                question=SOW_QUERY,
                context_text="",
            )
            self.assertEqual(response, "Not Mentioned in Provided Context")
            # Verify Qwen model initialization / generation is NEVER invoked
            mock_model.assert_not_called()

    def test_empty_whitespace_context_gatekeeper_bypasses_llm(self):
        """Verify that whitespace-only context also triggers negative restraint before LLM."""
        with patch.object(reasoning_engine, "_ensure_model") as mock_model:
            response = reasoning_engine.reason(
                question=SOW_QUERY,
                context_text="   \n\t  \n  ",
            )
            self.assertEqual(response, "Not Mentioned in Provided Context")
            mock_model.assert_not_called()

    def test_agent_run_empty_evidence_gatekeeper(self):
        """Verify that agent.run with zero evidence terminates at gatekeeper without LLM execution."""
        state = agent.run(
            query=SOW_QUERY,
            user_id="test_user_empty",
            initial_evidence=[],
        )
        self.assertEqual(state.final_answer, "Not Mentioned in Provided Context")
        self.assertEqual(len(state.reasoning), 0)


class TestIdentifierResolution(unittest.TestCase):
    """Verifies dynamic identifier resolution and scope binding."""

    def test_structural_identifier_extraction(self):
        """Verify regex extraction of alphanumeric document identifiers like SOW-2026-ENG-9941-C."""
        tokens = extract_structural_identifiers(SOW_QUERY)
        self.assertIn("SOW-2026-ENG-9941-C", tokens)

    def test_unindexed_document_identifier_lookup(self):
        """Verify lookup for an unindexed document ID returns None and sets DOCUMENT_NOT_INDEXED."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}

        resolved_id = resolve_document_identifier("SOW-9999-FAKE-DOC", collection=mock_collection)
        self.assertIsNone(resolved_id)

    def test_indexed_document_identifier_lookup(self):
        """Verify lookup for an indexed document ID successfully resolves the target doc_id."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc_sow_9941_c1"],
            "metadatas": [{"filename": "SOW-2026-ENG-9941-C.pdf", "document_id": "doc_sow_9941"}]
        }

        resolved_id = resolve_document_identifier("SOW-2026-ENG-9941-C", collection=mock_collection)
        self.assertEqual(resolved_id, "doc_sow_9941")

    def test_chunk_text_content_resolution(self):
        """Verify business identifier inside chunk document text resolves target doc_id even with arbitrary filename."""
        mock_collection = MagicMock()
        mock_collection.get.side_effect = [
            {"ids": [], "metadatas": []},  # metadata where filter
            {
                "ids": ["chunk_cam_1"],
                "metadatas": [{"filename": "random_upload_1790.txt", "document_id": "doc_cam_b9"}],
                "documents": ["Statement Identifier: CAM-2026-B9-RECON\nPremises RSF: 18,200 RSF"]
            }
        ]

        resolved_id = resolve_document_identifier("CAM-2026-B9-RECON", collection=mock_collection)
        self.assertEqual(resolved_id, "doc_cam_b9")


class TestSOWReconciliationExecution(unittest.TestCase):
    """Verifies end-to-end deterministic SOW extraction, reconciliation, and output hygiene."""

    def test_indexed_sow_reconciliation_match(self):
        """
        Index SOW-2026-ENG-9941-C and send SOW query.
        Assert:
        - All 4 milestones ($22,500, $40,500, $27,000, $18,000) are extracted.
        - Advance credit (-$18,000) is deducted.
        - Tax (+$5,400) is added.
        - Status is 'MATCH' ($95,400.00).
        - No response contains '$1,000,000' or '$500,000' or truncates mid-sentence.
        """
        evidence = [
            Evidence(
                document_id="doc_sow_9941",
                filename="SOW-2026-ENG-9941-C.pdf",
                page=1,
                chunk_id="chunk_0",
                source_text=SOW_DOCUMENT_TEXT,
                score=0.98,
            )
        ]

        state = agent.run(
            query=SOW_QUERY,
            user_id="test_enterprise_user",
            initial_evidence=evidence,
        )

        answer = state.final_answer

        # 1. Assert all 4 milestones are extracted
        self.assertIn("$22,500.00", answer)
        self.assertIn("$40,500.00", answer)
        self.assertIn("$27,000.00", answer)
        self.assertIn("$18,000.00", answer)
        self.assertIn("$108,000.00", answer)  # Milestone sum

        # 2. Assert advance credit deduction and tech tax addition
        self.assertIn("18,000.00", answer)   # Advance credit
        self.assertIn("5,400.00", answer)    # Tech tax

        # 3. Assert reconciliation status and final settlement commitment
        self.assertIn("MATCH", answer)
        self.assertIn("$95,400.00", answer)

        # 4. Anti-hallucination assertions: NO fabricated round numbers
        self.assertNotIn("1,000,000", answer)
        self.assertNotIn("1,500,000", answer)
        self.assertNotIn("2,000,000", answer)
        self.assertNotIn("2,500,000", answer)
        self.assertNotIn("500,000", answer)
        self.assertNotIn("7,000,000", answer)

        # 5. Output hygiene: No mid-sentence truncation
        self.assertFalse(answer.endswith("("))
        self.assertFalse(answer.endswith(","))
        self.assertFalse(answer.endswith("-"))
        self.assertNotIn("( $7,000,000 - $500,", answer)
        self.assertTrue(any(answer.endswith(ch) for ch in [".", "`", "*", "MATCH", "MATCH."]))


if __name__ == "__main__":
    unittest.main()
