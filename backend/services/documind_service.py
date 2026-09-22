import sys
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("documind.service")

# Ensure ML source path is available for importing existing DocuMind agent
ROOT_DIR = Path(__file__).resolve().parents[2]
ML_SRC = ROOT_DIR / "ml" / "src"

if str(ML_SRC) not in sys.path:
    sys.path.append(str(ML_SRC))

try:
    from documind_agent import (
        run_agent,
        classify_document,
        extract_metadata,
        search_documents,
        answer_question,
        embedding_model,
    )
    logger.info("Successfully imported DocuMind ML agent from %s", ML_SRC)
except ImportError as e:
    logger.error("Failed to import documind_agent from %s: %s", ML_SRC, e)
    raise


class DocuMindService:
    """
    Application service wrapper around the DocuMind ML/Agent system.
    """

    @property
    def embedding_model(self):
        return embedding_model

    @staticmethod
    def run_agent(question: str) -> Dict[str, Any]:
        """Execute the multi-tool agent planner and orchestration."""
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "status": "error",
                "message": "Question cannot be empty.",
                "tools_used": [],
                "results": [],
                "final_answer": "Please provide a question about your documents.",
            }
        return run_agent(cleaned_question)

    run_agent_query = run_agent

    @staticmethod
    def classify(question: str) -> Dict[str, Any]:
        """Classify document type using DistilBERT."""
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "status": "error",
                "message": "Question cannot be empty."
            }
        return classify_document(cleaned_question)

    @staticmethod
    def extract(question: str) -> Dict[str, Any]:
        """Extract invoice metadata (total gross amount, amount due)."""
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "status": "error",
                "message": "Question cannot be empty."
            }
        return extract_metadata(cleaned_question)

    extract_metadata = extract

    @staticmethod
    def search(question: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform hybrid BM25 + BGE semantic search or exact enumeration."""
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "status": "error",
                "message": "Search query cannot be empty.",
                "count": 0,
                "results": []
            }
        return search_documents(cleaned_question, top_k=top_k)

    @staticmethod
    def qa(question: str) -> Dict[str, Any]:
        """Grounded question answering with retrieval and Gemini."""
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "status": "error",
                "message": "Question cannot be empty."
            }
        return answer_question(cleaned_question)


documind_service = DocuMindService()
