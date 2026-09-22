"""
Unit tests for generation termination and output sanitation in backend/services/rag.py
Tests stopping sequence criteria, role turn stripping, conversational filler cleanup, and premise protection.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "packages"))

from services.rag import reasoning_engine, GroundedReasoningEngine


class TestOutputSanitation:
    def test_strip_hallucinated_role_turns(self):
        """Verify that hallucinated role turns like Human:/User:/Assistant: are truncated immediately."""
        raw = "The total invoice due is $20,567.50.\n\nHuman: Can you also tell me who signed it?\nAssistant: It was signed by John."
        cleaned = reasoning_engine.sanitize_output(raw)
        assert cleaned == "The total invoice due is $20,567.50."
        assert "Human:" not in cleaned
        assert "Assistant:" not in cleaned

    def test_strip_conversational_filler_and_pleasantries(self):
        """Verify removal of pleasantries and conversational filler."""
        raw = "Certainly! Here is the information: The payment terms specify Net 30 calendar days. I hope this helps!"
        cleaned = reasoning_engine.sanitize_output(raw)
        assert "Certainly!" not in cleaned
        assert "I hope this helps!" not in cleaned
        assert "The payment terms specify Net 30 calendar days." in cleaned

    def test_strip_apologies_and_meta_notes(self):
        raw = "I apologize for any confusion. The contractual late-payment fee is 1.5% per month.\n\nNote: This answer adheres to the strict context rules."
        cleaned = reasoning_engine.sanitize_output(raw)
        assert "I apologize" not in cleaned
        assert "Note: This answer" not in cleaned
        assert "1.5% per month" in cleaned

    def test_strip_chatml_tags(self):
        raw = "<|im_start|>assistant\nThe issuing entity is Nexus Global Dynamics LLC.<|im_end|>"
        cleaned = reasoning_engine.sanitize_output(raw)
        assert "<|im_start|>" not in cleaned
        assert "<|im_end|>" not in cleaned
        assert "The issuing entity is Nexus Global Dynamics LLC." in cleaned
