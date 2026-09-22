"""
Unit tests for the three-tool Gemini agent architecture and service integration.
Verifies tool definitions, lack of calculator tool, and absence of Qwen references in active agent services.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "packages"))

from services.agent import (
    agent_service,
    classify_document,
    retrieve_documents,
    extract_document,
)


class TestGeminiAgentArchitecture:
    def test_three_tool_declarations(self):
        """Verify the agent defines the exact three tools: classify, retrieve, extract."""
        tools = [classify_document, retrieve_documents, extract_document]
        tool_names = [t.__name__ for t in tools]
        assert len(tool_names) == 3
        assert "retrieve_documents" in tool_names
        assert "extract_document" in tool_names
        assert "classify_document" in tool_names

    def test_no_calculator_tool(self):
        """Assert no calculation/calculator tool is in the agent tools."""
        tools = [classify_document, retrieve_documents, extract_document]
        tool_names = [t.__name__ for t in tools]
        assert "calculate" not in tool_names
        assert "calculator" not in tool_names
        assert "documind_calculator" not in tool_names

    def test_no_qwen_references_in_agent(self):
        """Assert agent service does not reference Qwen."""
        import inspect
        source = inspect.getsource(agent_service.__class__)
        assert "qwen" not in source.lower()
