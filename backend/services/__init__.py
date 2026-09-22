"""
Consolidated backend services for DocuMind W.
Uses PEP 562 lazy attribute loading so that importing lightweight modules (extraction,
calculation, retrieval) or running tests does not eagerly instantiate heavy GPU ML models.
"""

from typing import Any

__all__ = [
    "documind_service",
    "DocuMindService",
    "DocumentIngestionService",
    "uploaded_rag_service",
    "UploadedDocumentRAGService",
    "agent",
    "gemini_agent",
    "AgentService",
    "retrieval_engine",
    "RetrievalEngine",
    "retrieve_documents",
    "extraction_engine",
    "ExtractionEngine",
    "extract_document",
    "classify_document",
    "reasoning_engine",
    "GroundedReasoningEngine",
]


def __getattr__(name: str) -> Any:
    if name in ("documind_service", "DocuMindService"):
        from .documind_service import documind_service, DocuMindService
        return documind_service if name == "documind_service" else DocuMindService
    elif name in ("document_ingestion", "DocumentIngestionService"):
        from .document_ingestion import DocumentIngestionService
        return DocumentIngestionService
    elif name in ("uploaded_rag_service", "UploadedDocumentRAGService"):
        from .uploaded_rag import uploaded_rag_service, UploadedDocumentRAGService
        return uploaded_rag_service if name == "uploaded_rag_service" else UploadedDocumentRAGService
    elif name in ("agent", "gemini_agent", "AgentService"):
        from .agent import agent_service
        return agent_service
    elif name in ("retrieval_engine", "RetrievalEngine"):
        from .retrieval import retrieval_engine, RetrievalEngine
        return retrieval_engine if name == "retrieval_engine" else RetrievalEngine
    elif name == "retrieve_documents":
        from .retrieval import retrieve_documents
        return retrieve_documents
    elif name in ("extraction_engine", "ExtractionEngine"):
        from .extraction import extraction_engine, ExtractionEngine
        return extraction_engine if name == "extraction_engine" else ExtractionEngine
    elif name == "extract_document":
        from .extraction import extract_document
        return extract_document
    elif name == "classify_document":
        from .agent import classify_document
        return classify_document
    elif name in ("reasoning_engine", "GroundedReasoningEngine"):
        from .rag import reasoning_engine, GroundedReasoningEngine
        return reasoning_engine if name == "reasoning_engine" else GroundedReasoningEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


