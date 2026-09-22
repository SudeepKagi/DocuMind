import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sqlalchemy.orm import Session

from .document_ingestion import DocumentIngestionService
from .chroma_service import chroma_service
from db.models import Document

logger = logging.getLogger("documind.services.uploaded_rag")

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
ML_SRC = ROOT_DIR / "ml" / "src"

if str(ML_SRC) not in sys.path:
    sys.path.append(str(ML_SRC))

# Import existing ML models loaded by DocuMind agent
try:
    from documind_agent import (
        embedding_model,
        device,
    )
    logger.info("Successfully linked to active DocuMind ML models for ChromaDB Uploaded RAG")
except ImportError as e:
    logger.error("Failed to link to DocuMind agent models: %s", e)
    raise


class UploadedDocumentRAGService:
    """
    Manages document upload ingestion, text extraction, chunking, BGE vector embeddings,
    persistent ChromaDB storage, and grounded multi-document question answering.
    """

    def __init__(self):
        self.storage_dir = BACKEND_DIR / "storage"
        self.uploads_dir = self.storage_dir / "uploads"
        self.processed_dir = self.storage_dir / "processed"

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def ingest_file(
        self,
        db: Session,
        user_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> Dict[str, Any]:
        """
        Ingest an uploaded document into PostgreSQL and ChromaDB:
        Upload -> Save raw file -> Create PostgreSQL record ('processing') ->
        Extract -> Chunk -> Embed with active BGE -> Insert into ChromaDB ->
        Save processed JSON -> Update PostgreSQL status ('processed')
        """
        if not DocumentIngestionService.is_supported(filename):
            raise ValueError(
                f"Unsupported file extension for '{filename}'. Supported: PDF, DOCX, TXT, EML"
            )

        if not file_bytes or len(file_bytes) == 0:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        file_type = DocumentIngestionService.get_file_type(filename)
        safe_filename = re.sub(r"[^\w\.-]", "_", filename)
        upload_path = self.uploads_dir / f"{document_id}_{safe_filename}"
        upload_time = datetime.now(timezone.utc)
        file_size = len(file_bytes)

        # 1. Save raw uploaded file
        with open(upload_path, "wb") as f:
            f.write(file_bytes)

        # 2. Create PostgreSQL document record with 'processing' status
        doc_record = Document(
            id=document_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            upload_time=upload_time,
            page_count=None,
            extraction_status="processing",
            chunk_count=0,
            storage_path=str(upload_path),
            processed_path=None,
        )
        db.add(doc_record)
        db.commit()
        db.refresh(doc_record)

        try:
            # 3. Extract text and pages
            pages_data, page_count = DocumentIngestionService.extract_text(upload_path)

            # 4. Chunk text
            chunks = []
            if pages_data:
                chunks = DocumentIngestionService.chunk_document(
                    document_id=document_id,
                    pages_data=pages_data,
                    target_words=500,
                    overlap_words=50,
                )

            chunk_count = len(chunks)

            # 5. Generate BGE embeddings and insert into ChromaDB
            if chunk_count > 0:
                chunk_texts = [c["text"] for c in chunks]
                embeddings = embedding_model.encode(
                    chunk_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                # Store into persistent ChromaDB
                chroma_service.add_chunks(
                    document_id=document_id,
                    user_id=user_id,
                    filename=filename,
                    file_type=file_type,
                    chunks=chunks,
                    embeddings=embeddings,
                )

            # 6. Save processed document JSON metadata and chunk texts (for UI inspection)
            processed_data = {
                "document_id": document_id,
                "user_id": user_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": file_size,
                "upload_time": upload_time.isoformat(),
                "page_count": page_count,
                "extraction_status": "processed",
                "chunk_count": chunk_count,
                "storage_path": str(upload_path),
                "chunks": chunks,
            }

            doc_json_path = self.processed_dir / f"{document_id}.json"
            with open(doc_json_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2)

            # 7. Update PostgreSQL record to 'processed'
            doc_record.extraction_status = "processed"
            doc_record.page_count = page_count
            doc_record.chunk_count = chunk_count
            doc_record.processed_path = str(doc_json_path)
            db.commit()
            db.refresh(doc_record)

            logger.info(
                "Successfully ingested document %s (%s) with %d chunks into PostgreSQL and ChromaDB",
                document_id,
                filename,
                chunk_count,
            )

            return {
                "document_id": doc_record.id,
                "filename": doc_record.filename,
                "file_type": doc_record.file_type,
                "file_size": doc_record.file_size,
                "upload_time": doc_record.upload_time.isoformat(),
                "page_count": doc_record.page_count,
                "extraction_status": doc_record.extraction_status,
                "chunk_count": doc_record.chunk_count,
            }

        except Exception as e:
            logger.exception("Error processing document %s: %s", document_id, e)
            doc_record.extraction_status = "failed"
            db.commit()
            # Cleanup any vectors if partially indexed
            chroma_service.delete_document(document_id, user_id)
            raise

    def list_documents(self, db: Session, user_id: str) -> List[Dict[str, Any]]:
        """
        List all documents belonging strictly to the user from PostgreSQL.
        """
        docs = (
            db.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .all()
        )
        return [
            {
                "document_id": d.id,
                "filename": d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "upload_time": (
                    d.upload_time.isoformat()
                    if hasattr(d.upload_time, "isoformat")
                    else str(d.upload_time)
                ),
                "page_count": d.page_count,
                "extraction_status": d.extraction_status,
                "chunk_count": d.chunk_count,
            }
            for d in docs
        ]

    def get_document(
        self,
        db: Session,
        user_id: str,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get full details and chunk list for a specific document owned by the user.
        """
        doc = (
            db.query(Document)
            .filter(Document.id == document_id, Document.user_id == user_id)
            .first()
        )
        if not doc:
            return None

        chunks = []
        if doc.processed_path and Path(doc.processed_path).exists():
            try:
                with open(doc.processed_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chunks = data.get("chunks", [])
            except Exception as e:
                logger.error("Error reading chunks for %s: %s", document_id, e)

        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "upload_time": (
                doc.upload_time.isoformat()
                if hasattr(doc.upload_time, "isoformat")
                else str(doc.upload_time)
            ),
            "page_count": doc.page_count,
            "extraction_status": doc.extraction_status,
            "chunk_count": doc.chunk_count,
            "storage_path": doc.storage_path,
            "processed_path": doc.processed_path,
            "chunks": chunks,
        }

    def delete_document(
        self,
        db: Session,
        user_id: str,
        document_id: str,
    ) -> bool:
        """
        Delete document record from PostgreSQL, delete ChromaDB vectors,
        and remove physical files from storage.
        """
        doc = (
            db.query(Document)
            .filter(Document.id == document_id, Document.user_id == user_id)
            .first()
        )
        if not doc:
            return False

        # 1. Remove vectors from ChromaDB
        chroma_service.delete_document(document_id, user_id)

        # 2. Remove raw upload file
        if doc.storage_path and Path(doc.storage_path).exists():
            try:
                Path(doc.storage_path).unlink()
            except Exception as e:
                logger.warning("Could not delete upload file %s: %s", doc.storage_path, e)

        # 3. Remove processed JSON
        if doc.processed_path and Path(doc.processed_path).exists():
            try:
                Path(doc.processed_path).unlink()
            except Exception as e:
                logger.warning("Could not delete doc JSON %s: %s", doc.processed_path, e)

        # 4. Remove record from PostgreSQL
        db.delete(doc)
        db.commit()

        logger.info("Deleted document %s from PostgreSQL, ChromaDB, and disk", document_id)
        return True

    @staticmethod
    def detect_question_type(question: str) -> str:
        """
        Detects query intent to adapt retrieval depth, prompt instructions, and token budgets:
        - COMPARISON: Cross-document, resume vs JD, skill gaps, or differences
        - CALCULATION: Formula, fees, mathematical computations
        - SUMMARY: High-level overview, synopsis, comprehensive summary
        - ENUMERATION: Complete listings, skills, requirements, items
        - EXPLANATION: How/why concepts, contextual descriptions
        - FACTUAL: Direct factual answers (what is, who is, when is)
        """
        q = question.lower().strip()

        # 1. Comparison
        if any(k in q for k in [
            "compare", "comparison", "difference", "differences", "versus", "vs",
            "how does", "match my", "match against", "match the", "fit for",
            "skills are missing", "what skills are missing",
            "skill gap", "skill gaps", "missing skill", "missing skills",
            "against"
        ]):
            return "COMPARISON"

        # 2. Calculation
        if any(k in q for k in [
            "calculate", "computation", "compute", "how is the fee",
            "how is the total", "formula", "breakdown of the fee", "what is the total"
        ]):
            return "CALCULATION"

        # 3. Summary
        if any(k in q for k in [
            "summarize", "summary", "give me a summary", "briefly describe",
            "high level overview", "overview of", "synopsis"
        ]):
            return "SUMMARY"

        # 4. Enumeration
        if any(k in q for k in [
            "list", "what are all", "find all", "which documents",
            "what are the skills", "what skills are", "what skills",
            "what projects are", "what projects", "which projects",
            "enumerate", "all requirements", "all skills", "all projects"
        ]):
            return "ENUMERATION"

        # 5. Explanation
        if any(k in q for k in [
            "explain", "describe", "what does this document contain",
            "tell me about", "why is", "why does", "how do"
        ]):
            return "EXPLANATION"

        # 6. Factual
        return "FACTUAL"

    def query_uploaded_rag(
        self,
        db: Session,
        user_id: str,
        question: str,
        n_results: int = 6,
        document_id: Optional[str] = None,
        target_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute grounded RAG across user's uploaded documents using ChromaDB vectors.
        Adapts retrieval depth, prompt instructions, and token budgets dynamically based on query intent.
        """
        q_type = self.detect_question_type(question)
        logger.info("Uploaded RAG intent detected: %s for query: '%s'", q_type, question)

        # 1. Embed query with existing BGE model
        query_embedding = embedding_model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # 2. Retrieve top-k relevant chunks per document from ChromaDB
        hits = []
        if target_doc_ids and len(target_doc_ids) > 1:
            # Multi-document comparison or broad query: retrieve top chunks from EACH document
            per_doc_k = 5 if q_type == "COMPARISON" else 3
            for d_id in target_doc_ids:
                doc_hits = chroma_service.query_vectors(
                    query_embedding=query_embedding,
                    user_id=user_id,
                    n_results=per_doc_k,
                    document_id=d_id,
                )
                hits.extend(doc_hits)
        else:
            # Single-document or general collection query: adjust depth by question type
            if q_type in ("SUMMARY", "ENUMERATION"):
                n_res = 7
            elif q_type == "COMPARISON":
                n_res = 8
            elif q_type in ("EXPLANATION", "CALCULATION"):
                n_res = 5
            else:  # FACTUAL
                n_res = 4

            hits = chroma_service.query_vectors(
                query_embedding=query_embedding,
                user_id=user_id,
                n_results=n_res,
                document_id=document_id,
            )

        if not hits:
            return {
                "status": "success",
                "question": question,
                "answer": "Not Mentioned in Provided Context",
                "documents": [],
                "sources": [],
            }

        # 3. Deduplicate chunks while preserving ranking
        seen_chunk_ids = set()
        deduped_hits = []
        for h in hits:
            c_id = h["metadata"].get("chunk_id", h["id"])
            if c_id not in seen_chunk_ids:
                seen_chunk_ids.add(c_id)
                deduped_hits.append(h)

        # 4. Build structured evidence context
        evidence_lines = []
        sources = []
        unique_docs = set()

        for i, hit in enumerate(deduped_hits):
            meta = hit["metadata"]
            filename = meta.get("filename", "Uploaded Document")
            page = meta.get("page", 1)
            chunk_id = meta.get("chunk_id", str(i))
            doc_id = meta.get("document_id", "")
            unique_docs.add(filename)

            evidence_lines.append(
                f"--- SOURCE {i + 1} (File: {filename}, Page: {page}, Chunk: {chunk_id}) ---\n"
                f"{hit['text'][:2200]}\n"
            )

            sources.append({
                "filename": filename,
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "page": page,
                "score": hit["score"],
                "text": hit["text"][:350] + ("..." if len(hit["text"]) > 350 else ""),
            })

        # 4b. Grounded Generation using Unified Gemini Agent
        from .agent import agent_service

        context_chunks = [
            {
                "chunk_id": str(h["metadata"].get("chunk_id", i)),
                "document_id": h["metadata"].get("document_id", ""),
                "filename": h["metadata"].get("filename", "Uploaded Document"),
                "page": h["metadata"].get("page", 1),
                "text": h["text"],
                "score": h["score"],
            }
            for i, h in enumerate(deduped_hits)
        ]

        answer = agent_service.answer_document_qa(
            question=question,
            context_chunks=context_chunks,
            document_id=document_id,
        )

        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "documents": list(unique_docs),
            "sources": sources,
        }

    def answer_question(
        self,
        db: Session,
        user_id: str,
        document_id: str,
        question: str,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Legacy endpoint compatibility: Answers a question targeting a specific document_id.
        """
        res = self.query_uploaded_rag(
            db=db,
            user_id=user_id,
            question=question,
            n_results=top_k,
            document_id=document_id,
        )
        return {
            "status": res["status"],
            "document_id": document_id,
            "question": question,
            "answer": res["answer"],
            "sources": res["sources"],
        }


uploaded_rag_service = UploadedDocumentRAGService()
