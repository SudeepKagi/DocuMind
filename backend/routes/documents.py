import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from db import get_db, get_or_create_dev_user, User
from schemas.documents import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentQARequest,
    DocumentQAResponse,
    DocumentSummary,
)
from services.uploaded_rag import uploaded_rag_service

logger = logging.getLogger("documind.routes.documents")

router = APIRouter(prefix="/documents", tags=["Uploaded Documents & QA"])


def get_current_user(db: Session = Depends(get_db)) -> User:
    """
    Dependency resolving the current user for document ownership.
    In the current MVP prior to authentication, resolves the default development user.
    """
    return get_or_create_dev_user(db)


@router.post("/upload", response_model=DocumentSummary, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a document (PDF, DOCX, TXT, EML) to be parsed, chunked,
    and indexed with BGE embeddings. Metadata and ownership are persisted in PostgreSQL.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file.filename}' is empty.",
            )

        summary = uploaded_rag_service.ingest_file(
            db=db,
            user_id=current_user.id,
            filename=file.filename,
            file_bytes=content,
        )
        return summary

    except ValueError as ve:
        logger.warning("Validation error uploading %s: %s", file.filename, ve)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.exception("Error ingesting uploaded document %s: %s", file.filename, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded document: {str(e)}",
        )


@router.get("", response_model=DocumentListResponse, status_code=status.HTTP_200_OK)
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve all uploaded documents belonging to the current user and their processing status from PostgreSQL.
    """
    try:
        docs = uploaded_rag_service.list_documents(db=db, user_id=current_user.id)
        return {
            "status": "success",
            "documents": docs,
            "total": len(docs),
        }
    except Exception as e:
        logger.exception("Error listing uploaded documents: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        )


@router.get("/{document_id}", response_model=DocumentDetailResponse, status_code=status.HTTP_200_OK)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve full metadata and extracted chunks for a specific uploaded document owned by the current user.
    """
    doc = uploaded_rag_service.get_document(db=db, user_id=current_user.id, document_id=document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    summary = {
        "document_id": doc["document_id"],
        "filename": doc["filename"],
        "file_type": doc["file_type"],
        "file_size": doc["file_size"],
        "upload_time": doc["upload_time"],
        "page_count": doc.get("page_count"),
        "extraction_status": doc["extraction_status"],
        "chunk_count": doc["chunk_count"],
    }

    chunks = [
        {
            "chunk_id": c["chunk_id"],
            "page": c.get("page"),
            "text": c["text"],
            "word_count": c.get("word_count"),
        }
        for c in doc.get("chunks", [])
    ]

    return {
        "status": "success",
        "document": summary,
        "chunks": chunks,
    }


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete an uploaded document record from PostgreSQL, its chunks, and its embeddings.
    """
    deleted = uploaded_rag_service.delete_document(
        db=db,
        user_id=current_user.id,
        document_id=document_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found or already deleted.",
        )
    return {
        "status": "success",
        "message": f"Document '{document_id}' deleted successfully.",
    }


@router.post("/{document_id}/qa", response_model=DocumentQAResponse, status_code=status.HTTP_200_OK)
def document_qa(
    document_id: str,
    request: DocumentQARequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Ask a question specifically about an uploaded document.
    Retrieval and QA are strictly isolated to this document and its owner.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field must not be empty.",
        )

    try:
        response = uploaded_rag_service.answer_question(
            db=db,
            user_id=current_user.id,
            document_id=document_id,
            question=request.question.strip(),
        )
        return response

    except ValueError as ve:
        logger.warning("QA validation error for %s: %s", document_id, ve)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve),
        )
    except Exception as e:
        logger.exception("Error executing QA on document %s: %s", document_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question for document: {str(e)}",
        )
