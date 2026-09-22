import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any, Dict

from db import get_db, get_or_create_dev_user, User
from schemas.requests import (
    AgentRequest,
    ClassificationRequest,
    ExtractionRequest,
    SearchRequest,
    QARequest,
    DocumentRequest,
)
from services.agent import agent_orchestrator
from services.documind_service import documind_service

logger = logging.getLogger("documind.routes.agent")

router = APIRouter(tags=["Agent & Intelligence"])


def get_current_user(db: Session = Depends(get_db)) -> User:
    """Resolve current user for agent multi-tenant retrieval."""
    return get_or_create_dev_user(db)


@router.post("/agent", status_code=status.HTTP_200_OK)
def run_agent_endpoint(
    request: AgentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Execute the DocuMind Agent planner across classification, metadata extraction,
    corpus hybrid search, and uploaded-document ChromaDB RAG.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field must not be empty."
        )

    try:
        response = agent_orchestrator.orchestrate(
            db=db,
            user_id=current_user.id,
            question=request.question.strip(),
        )
        return response
    except Exception as e:
        logger.exception("Error executing DocuMind agent: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}"
        )


@router.post("/classify", status_code=status.HTTP_200_OK)
def classify_endpoint(request: DocumentRequest) -> Dict[str, Any]:
    """
    Classify a document into Contract, Email, Invoice, Purchase Order, or Report
    using the fine-tuned DistilBERT model.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field must not be empty."
        )

    try:
        response = documind_service.classify(request.question)
        return response
    except Exception as e:
        logger.exception("Error executing document classification: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@router.post("/extract", status_code=status.HTTP_200_OK)
def extract_endpoint(request: DocumentRequest) -> Dict[str, Any]:
    """
    Extract structured metadata (total gross amount, amount due) from invoices.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field must not be empty."
        )

    try:
        response = documind_service.extract(request.question)
        return response
    except Exception as e:
        logger.exception("Error executing metadata extraction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata extraction failed: {str(e)}"
        )


@router.post("/search", status_code=status.HTTP_200_OK)
def search_endpoint(request: SearchRequest) -> Dict[str, Any]:
    """
    Perform hybrid BM25 lexical + BGE semantic search or exact document enumeration.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field must not be empty."
        )

    try:
        response = documind_service.search(request.question, top_k=request.top_k)
        return response
    except Exception as e:
        logger.exception("Error executing document search: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/qa", status_code=status.HTTP_200_OK)
def qa_endpoint(request: DocumentRequest) -> Dict[str, Any]:
    """
    Answer questions with grounded RAG using BGE/BM25 retrieval and Gemini reasoning.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field must not be empty."
        )

    try:
        response = documind_service.qa(request.question)
        return response
    except Exception as e:
        logger.exception("Error executing grounded QA: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QA failed: {str(e)}"
        )
