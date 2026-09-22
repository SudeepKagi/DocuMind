"""
DocuMind: Model-Driven Enterprise Document Intelligence Agent
Powered by Google GenAI (Gemini API) and official google-genai Python SDK.

Autonomous tool calling across 3 core document intelligence tools:
1. classify_document(document_id, text)
2. retrieve_documents(query, scope, top_k)
3. extract_document(evidence, schema)

Eliminates handwritten query planners, keyword routers, phrase matching,
and calculation.py. Gemini handles reasoning, in-context arithmetic, SLA rules,
and factual grounding.
"""

import os
import re
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from pathlib import Path

# Load environment
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from google import genai
from google.genai import types
from google.genai import errors

from .retrieval import retrieve_documents as internal_retrieve
from .extraction import extract_document as internal_extract
from .documind_service import documind_service

logger = logging.getLogger("documind.services.agent")

# Model configuration
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
MAX_TOOL_ITERATIONS = 5
MAX_RETRY_ATTEMPTS = 3


# ========================================================
# 1. Three Core Agent Tools
# ========================================================

def classify_document(
    document_id: Optional[str] = None,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classifies a document into Contract, Invoice, Purchase Order, Report, or Email
    using the fine-tuned DistilBERT document classification model.
    """
    try:
        query_text = text or document_id or ""
        res = documind_service.classify(query_text)
        return {
            "status": "success",
            "tool": "classify_document",
            "document_id": res.get("document_id", document_id),
            "predicted_class": res.get("predicted_class", "Unknown"),
            "confidence": res.get("confidence", 0.0),
            "probabilities": res.get("probabilities", {}),
        }
    except Exception as e:
        logger.error("classify_document error: %s", e)
        return {
            "status": "error",
            "tool": "classify_document",
            "error": str(e),
            "predicted_class": "Unknown",
        }


def retrieve_documents(
    query: str,
    scope: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieves relevant document chunks from the 223K+ chunk corpus or uploaded documents.
    Supports optional scope filter (e.g. {"document_type": "Contract"} or {"document_id": "doc_xyz"}).
    Applies hybrid BM25S + BGE-small retrieval, identifier priority, and context expansion.
    """
    try:
        hits = internal_retrieve(query=query, scope=scope, top_k=top_k)
        formatted_hits = []
        for h in hits:
            formatted_hits.append({
                "document_id": h.get("document_id", ""),
                "filename": h.get("filename", ""),
                "page": h.get("page", 1),
                "chunk_id": h.get("chunk_id", ""),
                "text": h.get("text", "")[:3000],
                "score": round(float(h.get("score", 0.0)), 4),
                "source_type": h.get("source_type", "corpus"),
            })
        return {
            "status": "success",
            "tool": "retrieve_documents",
            "query": query,
            "count": len(formatted_hits),
            "results": formatted_hits,
        }
    except Exception as e:
        logger.error("retrieve_documents error: %s", e)
        return {
            "status": "error",
            "tool": "retrieve_documents",
            "query": query,
            "count": 0,
            "results": [],
            "error": str(e),
        }


def extract_document(
    evidence: str,
    schema: Union[Dict[str, Any], List[str], str],
) -> Dict[str, Any]:
    """
    Extracts structured schema fields, itemized tables, or counterparties from evidence text.
    The schema is supplied dynamically (e.g. {"vendor_name": "string", "invoice_total": "number"}).
    """
    try:
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except Exception:
                pass

        res = internal_extract(evidence=evidence, schema=schema)
        return {
            "status": "success",
            "tool": "extract_document",
            "extracted_data": res,
        }
    except Exception as e:
        logger.error("extract_document error: %s", e)
        return {
            "status": "error",
            "tool": "extract_document",
            "error": str(e),
            "extracted_data": {},
        }


# ========================================================
# 2. System Instruction for Grounded Gemini Agent
# ========================================================

AGENT_SYSTEM_INSTRUCTION = """You are DocuMind, an enterprise document intelligence agent.
You interpret natural-language requests and select from document intelligence tools:
- classify_document: identify document type (Contract, Invoice, PO, Report, Email)
- retrieve_documents: search corpus or uploaded documents with optional scope (e.g. {"document_type": "Contract"} or {"document_id": "doc_xyz"})
- extract_document: extract structured schema fields or tables from evidence

CRITICAL GROUNDING AND REASONING RULES:
1. Strict Factual Grounding:
   - Answer exclusively using facts present in the retrieved evidence or structured extractions.
   - Do NOT invent unsupported facts, documents, dates, amounts, or entities.
   - Do NOT affirm or echo the user's premise before verifying it in the evidence.
   - Distinguish supplied facts from inferred conclusions.

2. Missing Information & Negative Restraint:
   - If a requested fact, amount, or rule is NOT mentioned in the evidence, you MUST state:
     "Not Mentioned in Provided Context"
   - For compound questions: if one fact is present but another is missing, answer the present fact clearly, and explicitly state "Not Mentioned in Provided Context" for the missing fact. Do not reject or terminate the whole response.

3. Mathematical & Arithmetic Reasoning:
   - You are the reasoning and calculation engine. Perform arithmetic, percentages, discounts, effective taxes, SLA service credits, date offsets, and reconciliation directly from the rules and numbers stated in the documents.
   - When evaluating reconciliation of milestones, credits, taxes, and final settlement commitment: itemize each milestone amount, the milestone sum, advance credit deduction, tax addition, and explicitly state whether the calculated total matches the final commitment (using status MATCH or MISMATCH).
   - Show your reasoning concisely when explaining calculations.

4. Enumeration & Listing:
   - When asked to list or find documents mentioning a topic, call retrieve_documents with the query and optional scope={"document_type": "..."}. Enumerate the matching documents found in the evidence.
"""



# ========================================================
# 3. Agent Service & Execution Engine
# ========================================================

# ========================================================
# 4. Backward Compatibility Data Contracts
# ========================================================

from dataclasses import dataclass, field

@dataclass
class EvidenceItem:
    """Standard document evidence object preserving provenance across the pipeline."""
    document_id: str = ""
    chunk_id: str = ""
    page: int = 1
    text: str = ""
    source_type: str = "uploaded"
    filename: str = "Document"
    score: float = 0.0
    source_text: Optional[str] = None

    def __post_init__(self):
        if not self.text and self.source_text:
            self.text = self.source_text
        elif not self.source_text and self.text:
            self.source_text = self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "text": self.text[:350] + ("..." if len(self.text) > 350 else ""),
            "score": self.score,
            "source_type": self.source_type,
        }

Evidence = EvidenceItem


@dataclass
class AgentState:
    """State contract preserved for test harnesses and backward compatibility."""
    query: str = ""
    user_id: str = "dev_user_001"
    evidence: List[EvidenceItem] = field(default_factory=list)
    final_answer: str = ""
    reasoning: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "COMPLETED"


# ========================================================
# 5. Agent Service Class Extensions
# ========================================================

class AgentService:
    """
    Hosted Gemini Agent Orchestrator.
    Manages client instantiation, tool definitions, tool calling execution loop,
    provenance tracking, and formatted response assembly for DocuMind.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None

    def get_model_name(self) -> str:
        """Configurable model through GEMINI_MODEL environment variable."""
        return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()

    def get_client(self) -> genai.Client:
        """Lazy initialization of Google GenAI Client with GEMINI_API_KEY."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured in backend/.env. "
                "Please set GEMINI_API_KEY to use the DocuMind hosted Gemini agent."
            )
        if self._client is None:
            self._client = genai.Client(api_key=api_key)
        return self._client

    def run_agent(
        self,
        question: str,
        db: Optional[Session] = None,
        user_id: str = "dev_user_001",
    ) -> Dict[str, Any]:
        """
        Executes the hosted Gemini agent loop with tool calling.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "status": "error",
                "question": question,
                "tools_used": [],
                "results": [],
                "sources": [],
                "final_answer": "Please provide a question about your documents.",
            }

        client = self.get_client()
        primary_model = self.get_model_name()
        preferred_candidates = [
            primary_model,
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.8-flash",
        ]
        candidate_models = []
        for m in preferred_candidates:
            if m and m not in candidate_models:
                candidate_models.append(m)

        available_tools = [classify_document, retrieve_documents, extract_document]

        response = None
        last_error = None
        chat: Optional[Any] = None

        for model_cand in candidate_models:
            try:
                chat = client.chats.create(
                    model=model_cand,
                    config=types.GenerateContentConfig(
                        tools=available_tools,
                        system_instruction=AGENT_SYSTEM_INSTRUCTION,
                        temperature=0.05,
                    )
                )

                for attempt in range(MAX_RETRY_ATTEMPTS):
                    try:
                        response = chat.send_message(cleaned_question)
                        break
                    except errors.ClientError as ce:
                        last_error = ce
                        err_str = str(ce)
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                            logger.warning("[%s] 429 quota reached, switching to fallback candidate model...", model_cand)
                            break
                        else:
                            raise
                    except errors.ServerError as se:
                        last_error = se
                        logger.warning("[%s] 503 ServerError. Retrying (attempt %d/%d)...", model_cand, attempt + 1, MAX_RETRY_ATTEMPTS)
                        time.sleep(1.5)
                    except Exception as net_err:
                        last_error = net_err
                        logger.warning("[%s] Network exception (%s). Retrying...", model_cand, net_err)
                        time.sleep(1)
                if response:
                    logger.info("Successfully received agent response using model: %s", model_cand)
                    break
            except Exception as e:
                last_error = e
                logger.warning("Error with model %s: %s", model_cand, e)

        if not response or chat is None:
            logger.error("Gemini request failed across candidate models: %s", last_error)
            # Local Grounded Fallback: search ChromaDB / corpus directly
            try:
                hits = internal_retrieve(query=cleaned_question, top_k=5)
                if hits:
                    evidence_texts = []
                    fallback_sources = []
                    seen = set()
                    for h in hits:
                        cid = h.get("chunk_id") or h.get("document_id", "")
                        if cid not in seen:
                            seen.add(cid)
                            fname = h.get("filename", "Indexed Document")
                            pg = h.get("page", 1)
                            txt = h.get("text", "").strip()
                            evidence_texts.append(f"### {fname} (Page {pg})\n\n{txt}")
                            fallback_sources.append({
                                "document_id": h.get("document_id", ""),
                                "filename": fname,
                                "page": pg,
                                "chunk_id": cid,
                                "text": txt[:350],
                                "score": round(float(h.get("score", 0.0)), 4),
                                "source_type": h.get("source_type", "uploaded"),
                            })

                    summary_header = (
                        "> **System Notice:** The remote AI model is temporarily busy. "
                        "Here is the verified context extracted directly from your indexed documents:\n\n"
                    )
                    return {
                        "status": "success",
                        "question": cleaned_question,
                        "tools_used": ["retrieve_documents"],
                        "results": [{
                            "tool": "retrieve_documents",
                            "status": "success",
                            "count": len(hits),
                            "results": hits,
                        }],
                        "sources": fallback_sources,
                        "final_answer": summary_header + "\n\n---\n\n".join(evidence_texts),
                    }
            except Exception as ret_err:
                logger.error("Fallback retrieval failed: %s", ret_err)

            return {
                "status": "error",
                "question": cleaned_question,
                "tools_used": [],
                "results": [],
                "sources": [],
                "final_answer": "The document intelligence service is temporarily experiencing high traffic. Please retry in a few moments.",
            }

        final_text = response.text or ""

        # Extract execution trace and sources from chat history
        tools_used: List[str] = []
        activity_results: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        seen_source_chunks = set()

        history = chat.get_history() if chat is not None else []
        for msg in history:
            parts = getattr(msg, "parts", None) or []
            for part in parts:
                if hasattr(part, "function_call") and part.function_call:
                    t_name = getattr(part.function_call, "name", None)
                    if t_name and str(t_name) not in tools_used:
                        tools_used.append(str(t_name))

                if hasattr(part, "function_response") and part.function_response:
                    resp_dict = getattr(part.function_response, "response", {})
                    t_name = getattr(part.function_response, "name", "tool")

                    card = {
                        "tool": t_name,
                        "status": "success",
                    }
                    if isinstance(resp_dict, dict):
                        card.update(resp_dict.get("result", resp_dict))

                        res_list = resp_dict.get("results") or resp_dict.get("result", {}).get("results", [])
                        if isinstance(res_list, list):
                            for r in res_list:
                                if isinstance(r, dict):
                                    cid = r.get("chunk_id") or r.get("document_id", "")
                                    if cid not in seen_source_chunks:
                                        seen_source_chunks.add(cid)
                                        sources.append({
                                            "document_id": r.get("document_id", "Doc"),
                                            "filename": r.get("filename", "Document"),
                                            "page": r.get("page", 1),
                                            "chunk_id": cid,
                                            "text": r.get("text", "")[:350],
                                            "score": r.get("score", 0.0),
                                            "source_type": r.get("source_type", "corpus"),
                                        })

                    activity_results.append(card)

        if not tools_used:
            tools_used = ["gemini_reasoning"]

        return {
            "status": "success",
            "question": cleaned_question,
            "tools_used": tools_used,
            "results": activity_results,
            "sources": sources,
            "final_answer": final_text.strip(),
        }

    def run(
        self,
        query: str,
        user_id: str = "dev_user_001",
        initial_evidence: Optional[List[Any]] = None,
        db: Optional[Session] = None,
    ) -> AgentState:
        """
        Compatibility method for test harnesses.
        Returns AgentState with final_answer, tools_used, and results.
        """
        # Context gatekeeper: If explicitly passed empty evidence list, return negative restraint immediately
        if initial_evidence is not None and len(initial_evidence) == 0:
            return AgentState(
                query=query,
                user_id=user_id,
                evidence=[],
                final_answer="Not Mentioned in Provided Context",
                reasoning=[],
                tools_used=[],
                results=[],
                status="COMPLETED",
            )

        query_text = query
        ev_items = []
        if initial_evidence:
            for item in initial_evidence:
                if isinstance(item, EvidenceItem):
                    ev_items.append(item)
                elif isinstance(item, dict):
                    ev_items.append(EvidenceItem(**item))
            context_blocks = [e.text or e.source_text for e in ev_items if (e.text or e.source_text)]
            if context_blocks:
                query_text = f"{query}\n\nSupplied Document Evidence:\n" + "\n\n".join(context_blocks)

        resp = self.run_agent(question=query_text, db=db, user_id=user_id)

        return AgentState(
            query=query,
            user_id=user_id,
            evidence=ev_items,
            final_answer=resp.get("final_answer", ""),
            reasoning=[resp.get("final_answer", "")],
            tools_used=resp.get("tools_used", []),
            results=resp.get("results", []),
            status="COMPLETED",
        )

    def answer_document_qa(
        self,
        question: str,
        context_chunks: Union[List[Dict[str, Any]], str],
        document_id: Optional[str] = None,
    ) -> str:
        """
        Direct grounded question answering using Gemini with strict factual grounding and negative restraint.
        Used for uploaded document QA and corpus-level QA.
        """
        if not context_chunks:
            return "Not Mentioned in Provided Context"

        evidence_blocks = []
        if isinstance(context_chunks, str):
            if context_chunks.strip():
                evidence_blocks.append(context_chunks.strip())
        else:
            for i, c in enumerate(context_chunks):
                txt = c.get("text", "").strip()
                pg = c.get("page", 1)
                cid = c.get("chunk_id", f"chunk_{i}")
                if txt:
                    evidence_blocks.append(f"--- EVIDENCE {i+1} (Page {pg}, ID: {cid}) ---\n{txt}")

        if not evidence_blocks:
            return "Not Mentioned in Provided Context"

        prompt = (
            f"You are DocuMind, an enterprise document intelligence assistant.\n"
            f"Answer the user's question strictly using ONLY the provided evidence below.\n\n"
            f"RULES:\n"
            f"1. Strict Factual Grounding: Answer exclusively using facts present in the evidence. "
            f"Do not invent facts, dates, amounts, or entities.\n"
            f"2. If the requested information is not mentioned in the evidence, output ONLY: 'Not Mentioned in Provided Context'.\n"
            f"3. Perform any required arithmetic, calculations, or comparisons directly from the numbers in the evidence.\n"
            f"4. Be direct, concise, and clear.\n\n"
            f"EVIDENCE:\n" + "\n\n".join(evidence_blocks) + "\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )

        client = self.get_client()
        model_name = self.get_model_name()
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.05,
                )
            )
            return (resp.text or "").strip()
        except Exception as e:
            logger.error("Gemini document QA error: %s", e)
            try:
                resp = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.05)
                )
                return (resp.text or "").strip()
            except Exception as e2:
                logger.error("Gemini fallback document QA error: %s", e2)
                return "The document intelligence service is temporarily experiencing high traffic. Please retry in a few moments."


    orchestrate = run_agent



agent_service = AgentService()
agent = agent_service
agent_orchestrator = agent_service
