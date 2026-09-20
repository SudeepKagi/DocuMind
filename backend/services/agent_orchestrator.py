import logging
import re
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from db.models import Document
from .uploaded_rag import uploaded_rag_service
from .documind_service import documind_service

logger = logging.getLogger("documind.services.orchestrator")


class AgentOrchestrationService:
    """
    Application-layer agent orchestrator that intelligently routes user questions
    to the existing 6,638-document corpus, user-uploaded documents (ChromaDB),
    or both when comparative synthesis is requested.
    """

    CORPUS_PATTERNS = [
        re.compile(r"\b(invoice|contract|report|purchase[-_ ]?order)[-_ ]?\d+\b", re.IGNORECASE),
        re.compile(r"\bdoc_\d{4}\b", re.IGNORECASE),
        re.compile(r"\b(termination fee|gross amount|amount due|vendor name)\b", re.IGNORECASE),
    ]

    UPLOADED_KEYWORDS = [
        "resume", "cv", "internship", "jd", "job description", "sudeep",
        "uploaded", "my document", "my documents", "my file", "my files",
        "candidate", "applicant", "education", "experience", "gpa", "degree",
        "skills", "qualification", "qualifications", "hiring",
    ]

    def _determine_mode(
        self,
        question: str,
        uploaded_docs: List[Document],
    ) -> str:
        """
        Classifies query intent into: 'UPLOADED', 'CORPUS', or 'BOTH'.
        """
        q_lower = question.lower()

        # If user has no uploaded documents, default to CORPUS
        if not uploaded_docs:
            return "CORPUS"

        # Check for explicit cross-system comparison (BOTH)
        is_cross_compare = (
            ("compare" in q_lower or "difference" in q_lower or "versus" in q_lower or "vs" in q_lower)
            and ("corpus" in q_lower or "database" in q_lower or "existing" in q_lower or "contracts" in q_lower)
            and any(k in q_lower for k in ["uploaded", "resume", "my document", "jd"])
        )
        if is_cross_compare:
            return "BOTH"

        # Check if question matches uploaded filenames or keywords
        has_uploaded_keyword = any(k in q_lower for k in self.UPLOADED_KEYWORDS)
        
        # Check against uploaded filenames
        has_filename_match = False
        for doc in uploaded_docs:
            clean_name = re.sub(r"\.[a-zA-Z0-9]+$", "", doc.filename.lower())
            tokens = re.split(r"[\s_()\-]+", clean_name)
            if any(len(t) > 2 and t in q_lower for t in tokens):
                has_filename_match = True
                break

        # Check if question matches corpus document patterns
        has_corpus_pattern = any(p.search(question) for p in self.CORPUS_PATTERNS)

        if (has_uploaded_keyword or has_filename_match) and not has_corpus_pattern:
            return "UPLOADED"

        if has_corpus_pattern and not (has_uploaded_keyword or has_filename_match):
            return "CORPUS"

        # If question contains comparison across uploaded docs (e.g. "compare my resume with the internship JD")
        if "compare" in q_lower and (has_uploaded_keyword or has_filename_match):
            return "UPLOADED"

        # Default fallback logic
        if has_uploaded_keyword or has_filename_match:
            return "UPLOADED"

        return "CORPUS"

    def orchestrate(
        self,
        db: Session,
        user_id: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Main Agent entry point: routes to the proper retrieval system and synthesizes response.
        """
        uploaded_docs = (
            db.query(Document)
            .filter(
                Document.user_id == user_id,
                Document.extraction_status == "processed",
            )
            .all()
        )

        mode = self._determine_mode(question, uploaded_docs)
        logger.info("Agent query '%s' classified as mode: %s (user: %s, uploaded docs: %d)", question, mode, user_id, len(uploaded_docs))

        if mode == "UPLOADED":
            q_lower = question.lower()
            matched_docs = []
            for doc in uploaded_docs:
                clean_name = re.sub(r"\.[a-zA-Z0-9]+$", "", doc.filename.lower())
                tokens = [t for t in re.split(r"[\s_()\-]+", clean_name) if len(t) > 2]
                if any(t in q_lower for t in tokens) or (clean_name in q_lower) or (doc.filename.lower() in q_lower):
                    matched_docs.append(doc)

            is_comparison = any(w in q_lower for w in [
                "compare", "difference", "differences", "versus", "vs",
                "both", "all", "between", "match", "fit for", "missing",
                "relevant to", "against"
            ])

            target_doc_id = None
            target_doc_ids = None

            if len(matched_docs) == 1 and not is_comparison:
                target_doc_id = matched_docs[0].id
                logger.info("Direct single-document query targeted: %s (%s)", matched_docs[0].filename, target_doc_id)
            elif len(matched_docs) > 1:
                target_doc_ids = [d.id for d in matched_docs]
                logger.info("Multi-document query targeted: %s", [d.filename for d in matched_docs])
            elif not matched_docs and any(w in q_lower for w in ["documents", "files", "all", "uploaded", "everything", "enterprise"]):
                target_doc_ids = [d.id for d in uploaded_docs]
                logger.info("Broad multi-document query targeted across %d documents", len(target_doc_ids))

            # Query ChromaDB across uploaded documents
            rag_res = uploaded_rag_service.query_uploaded_rag(
                db=db,
                user_id=user_id,
                question=question,
                n_results=6,
                document_id=target_doc_id,
                target_doc_ids=target_doc_ids,
            )
            return {
                "question": question,
                "tools_used": ["uploaded_rag"],
                "results": {
                    "tool": "uploaded_rag",
                    "status": "success",
                    "mode": "UPLOADED",
                    "documents": rag_res.get("documents", []),
                    "sources": rag_res.get("sources", []),
                },
                "final_answer": rag_res.get("answer", ""),
            }

        elif mode == "CORPUS":
            # Delegate to existing 6,638-document corpus agent implementation
            corpus_res = documind_service.run_agent(question)
            return corpus_res

        else:  # BOTH
            # Query both ChromaDB and corpus hybrid search
            uploaded_res = uploaded_rag_service.query_uploaded_rag(
                db=db,
                user_id=user_id,
                question=question,
                n_results=4,
            )
            corpus_search = documind_service.search(query=question, top_k=2)

            # Combine sources
            all_sources = list(uploaded_res.get("sources", []))
            for res in corpus_search.get("results", []):
                all_sources.append({
                    "filename": f"Corpus: {res.get('document_id')}",
                    "document_id": res.get("document_id"),
                    "chunk_id": str(res.get("chunk_id", 0)),
                    "page": res.get("page", 1),
                    "score": res.get("score", 0.0),
                    "text": res.get("text", "")[:350] + "...",
                })

            return {
                "question": question,
                "tools_used": ["uploaded_rag", "corpus_hybrid_search"],
                "results": {
                    "tool": "multi_source_synthesis",
                    "status": "success",
                    "mode": "BOTH",
                    "documents": uploaded_res.get("documents", []) + [r.get("document_id") for r in corpus_search.get("results", [])],
                    "sources": all_sources,
                },
                "final_answer": uploaded_res.get("answer", ""),
            }


agent_orchestrator = AgentOrchestrationService()
