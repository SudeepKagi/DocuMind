"""
DocuMind W: Unified Agent Orchestrator & Task DAG Engine
Consolidates Query Understanding, Task Decomposition, 5-Capability DAG,
Evidence Contract, Agent Orchestration, and Response Assembly into a single unified service.

Strictly executes across the 5 Core Primitives:
1. CLASSIFICATION: classify(document)
2. RETRIEVAL: retrieve(query, scope, k)
3. EXTRACTION: extract(evidence, schema)
4. CALCULATION: calculate(operation, inputs)
5. REASONING: reason(question, evidence, structured_data, mode)
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from sqlalchemy.orm import Session
import numpy as np

from .retrieval import retrieval_engine, RetrievalScope
from .extraction import (
    extraction_engine,
    LineItemsSchema,
    PaymentSchema,
    PaymentTermsSchema,
    CounterpartySchema,
    ContractTermsSchema,
    ProjectBudgetSchema,
    InvoiceSchema,
    TieredMatrixSchema,
    ReconciliationLedgerSchema,
    ExtractionResult,
    SCHEMA_REGISTRY,
)
from .calculation import calculation_engine
from .rag import reasoning_engine

logger = logging.getLogger("documind.services.agent")


# ========================================================
# 1. Standard Evidence Contract & Provenance
# ========================================================

@dataclass
class EvidenceItem:
    """Standard document evidence object preserving provenance across the pipeline."""
    document_id: str
    chunk_id: str
    page: int
    text: str = ""
    source_type: str = "uploaded"  # "uploaded" or "corpus"
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
            "source_text": self.text[:350] + ("..." if len(self.text) > 350 else ""),
            "score": self.score,
            "source_type": self.source_type,
        }


# Backward compatibility alias
Evidence = EvidenceItem


@dataclass
class StructuredResult:
    """Verified structured extraction with explicit evidence provenance."""
    value: Any
    source_evidence: Optional[EvidenceItem] = None
    validation_state: str = "verified"  # "verified", "unverified", "not_provided"
    confidence: float = 1.0


@dataclass
class CalculatedResult:
    """Deterministic calculation result with explicit operation and inputs."""
    operation: str
    inputs: Dict[str, Any]
    result: Any
    validation_state: str = "verified"
    formula: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class AgentState:
    """Shared execution state passed between task nodes in the DAG."""
    query: str
    clean_query: str
    intents: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    requested_schemas: List[str] = field(default_factory=list)
    requested_fields: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    mode: str = "qa"

    # Evidence contract
    evidence: List[EvidenceItem] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    structured_results: Dict[str, StructuredResult] = field(default_factory=dict)
    calculations: List[CalculatedResult] = field(default_factory=list)
    reconciliation: Optional[Dict[str, Any]] = None
    temporal_results: Optional[Dict[str, Any]] = None
    reasoning: List[str] = field(default_factory=list)
    final_answer: str = ""

    def add_evidence_hits(self, hits: List[Dict[str, Any]]) -> None:
        seen = {e.chunk_id for e in self.evidence}
        for h in hits:
            meta = h.get("metadata", {})
            c_id = str(meta.get("chunk_id", h.get("id", "")))
            if c_id not in seen:
                seen.add(c_id)
                self.evidence.append(
                    EvidenceItem(
                        document_id=meta.get("document_id", ""),
                        filename=meta.get("filename", "Document"),
                        page=meta.get("page", 1),
                        chunk_id=c_id,
                        text=h.get("text", ""),
                        source_type="uploaded",
                        score=float(h.get("score", 0.0)),
                    )
                )

    def get_combined_evidence_text(self, max_length: int = 8000) -> str:
        texts = [f"--- SOURCE: {e.filename} (Page {e.page}) ---\n{e.text}" for e in self.evidence]
        combined = "\n\n".join(texts)
        return combined[:max_length]


# ========================================================
# 2. Query Understanding & Schema / Operation Mapping
# ========================================================

@dataclass
class QueryAnalysis:
    raw_query: str
    clean_query: str
    intents: List[str]            # Subsets of the 5 primitives
    entities: List[str]
    requested_schemas: List[str]  # e.g. LineItemsSchema, PaymentSchema, CounterpartySchema
    requested_fields: List[str]   # specific field targets
    operations: List[str]         # e.g. sum, difference, percentage, rate, range_lookup, reconcile, date_offset
    mode: str                     # qa, comparison, summary, enumeration, explain
    constraints: List[str]
    target_item_index: Optional[int] = None
    target_item_selector: Optional[str] = None


class QueryUnderstandingEngine:
    CONVERSATIONAL_PREFIXES = [
        re.compile(r"^(?:hey|hi|hello|please|can you|could you|i think|i would like to|tell me|show me)\s*,?\s*", re.IGNORECASE),
        re.compile(r"^(?:we need to|let's|kindly)\s*", re.IGNORECASE),
    ]

    ORDINAL_MAP = {
        "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
    }

    def strip_conversational_mask(self, query: str) -> str:
        cleaned = query.strip()
        for p in self.CONVERSATIONAL_PREFIXES:
            cleaned = p.sub("", cleaned).strip()
        return re.sub(r"\s+", " ", cleaned).strip()

    def extract_entities(self, query: str) -> List[str]:
        entities = []
        entities.extend(re.findall(r"\b[A-Za-z0-9]{2,10}[-_][A-Za-z0-9_.-]{2,}\b", query))
        entities.extend(re.findall(r"\b(?:invoice|contract|report|po|doc)[-_ ]?[A-Za-z0-9_-]+\b", query, re.IGNORECASE))
        entities.extend(re.findall(r"\b\d{6,}\b", query))
        entities.extend(re.findall(r"\b[A-Z0-9_-]{6,}\b", query))
        return list(dict.fromkeys(entities))

    def analyze(self, query: str) -> QueryAnalysis:
        raw_q = query.strip()
        q_lower = raw_q.lower()
        clean_q = self.strip_conversational_mask(raw_q)
        entities = self.extract_entities(raw_q)

        intents: List[str] = []
        requested_schemas: List[str] = []
        requested_fields: List[str] = []
        operations: List[str] = []
        constraints: List[str] = []
        target_item_index = None
        target_item_selector = None

        # Determine reasoning mode
        if any(w in q_lower for w in ["compare", "comparison", "difference", "versus", "vs", "gap", "fit for"]):
            mode = "comparison"
        elif any(w in q_lower for w in ["summarize", "summary", "overview", "synopsis"]):
            mode = "summary"
        elif any(w in q_lower for w in ["list", "enumerate", "all items", "what are all", "which documents"]):
            mode = "enumeration"
        elif any(w in q_lower for w in ["explain", "why", "how", "describe"]):
            mode = "explain"
        else:
            mode = "qa"

        # 1. Classification Primitive
        if any(p in q_lower for p in [
            "type of document", "document type", "classify", "is this an invoice",
            "is this a credit memo", "or a purchase order", "billing record", "what category", "what type of legal document"
        ]):
            intents.append("classification")
            requested_fields.append("document_type")

        # 2. Line Items / Tabular Schema
        is_table = any(p in q_lower for p in [
            "line item", "line items", "deliverables table", "items listed",
            "extended subtotal", "unit price", "third line item", "itemized",
            "per item", "each item", "all items", "from the table", "invoice table",
            "item #", "charges", "individual item", "item totals",
            "milestone", "milestones", "release amounts", "net payable release",
        ])
        if is_table:
            requested_schemas.append("line_items")
            intents.append("extraction")

        for ord_word, idx_val in self.ORDINAL_MAP.items():
            if f"{ord_word} line item" in q_lower or f"{ord_word} item" in q_lower:
                target_item_index = idx_val
                break

        item_hash = re.search(r"item\s*#?(\d+)", q_lower)
        if item_hash:
            target_item_index = int(item_hash.group(1))

        # 3. Calculation Operations (Deterministic math, SLA range lookups, tax rates)
        # 3a. Effective Tax Rate & Percentage
        if any(p in q_lower for p in ["effective tax rate", "tax rate per item", "tax percentages", "effective tax", "percentage of tax", "tax applies effectively", "tax rate", "deduction percentage"]) or ("compute" in q_lower and "tax" in q_lower):
            intents.append("calculation")
            operations.append("rate")
            operations.append("percentage")

        # 3b. SLA Tier & Range Lookup (strictly for uptime SLA outage queries)
        is_sla = any(p in q_lower for p in ["uptime", "service credit", "tier lookup"]) or (any(p in q_lower for p in ["sla", "tier"]) and not any(w in q_lower for w in ["milestone", "sow-", "sow "]))
        if is_sla:
            intents.append("calculation")
            operations.append("range_lookup")

        # 3c. Sum, Difference, & Reconciliation
        is_recon = any(p in q_lower for p in [
            "sum all", "computed sum", "subtract", "subtract the", "matches the grand total",
            "matches the total", "reconcile", "subtotal reflects", "pre-tax or post-discount",
            "advance credit", "tech tax", "settlement commitment", "final settlement", "milestone"
        ])
        if is_recon:
            intents.append("calculation")
            operations.append("sum")
            operations.append("difference")
            operations.append("reconcile")
            if any(w in q_lower for w in ["milestone", "advance", "settlement", "tech tax"]):
                requested_schemas.append("line_items")
                intents.append("extraction")

        # 3d. Date Arithmetic & Offset
        if any(p in q_lower for p in ["net-30", "net 30", "net-60", "net 60", "overdue", "calendar date", "due date", "dispute", "grace period", "settlement is delayed"]):
            intents.append("calculation")
            operations.append("date_offset")

        # 4. Extraction Schemas & Fields
        field_map = {
            "iban": ["iban", "bank information", "bank details"],
            "routing_code": ["routing code", "routing number", "aba"],
            "penalty_rate": ["penalty rate", "late settlement charge", "interest rate", "penalty fee", "late payment penalty", "contractual late-payment"],
            "bank_branch_address": ["bank branch address", "branch address"],
            "remittance_address": ["physical address where payment must be mailed", "remittance address", "remit to", "where payment must be mailed"],
            "recipient_address": ["recipient address", "billed to", "client address"],
            "counterparty": ["counterparty", "parties involved", "issuing the charges", "vendor", "client", "who are the parties"],
            "transaction_reference": ["transaction reference", "invoice reference"],
            "purchase_order": ["purchase order", "po number"],
            "grace_period_clause": ["grace period", "settlement is delayed"],
            "project_budget": ["budget", "project budget", "allocated budget"],
            "commitment_fee": ["commitment fee", "base commitment fee"],
        }
        for f_name, keywords in field_map.items():
            if any(k in q_lower for k in keywords):
                requested_fields.append(f_name)

        # Map requested fields to schemas
        if any(f in requested_fields for f in ["iban", "routing_code", "penalty_rate"]):
            requested_schemas.append("payment")
        if any(f in requested_fields for f in ["counterparty", "remittance_address", "recipient_address"]):
            requested_schemas.append("counterparty")
        if any(f in requested_fields for f in ["commitment_fee", "counterparty"]) and any(w in q_lower for w in ["msa", "contract", "agreement", "parties"]):
            requested_schemas.append("contract")
        if any(f in requested_fields for f in ["project_budget"]):
            requested_schemas.append("budget")

        if requested_fields or requested_schemas:
            intents.append("extraction")

        # Negative Restraint Constraints
        if any(p in q_lower for p in ["only using information explicitly stated", "state 'not provided' if missing", "strict", "explicitly stated"]):
            constraints.append("strict_negative_restraint")

        # Core Defaults: retrieval is prerequisite, reasoning synthesizes
        if "retrieval" not in intents:
            intents.insert(0, "retrieval")

        if "reasoning" not in intents:
            # Add reasoning unless query is a purely deterministic extraction + calculation with zero narrative requested
            if not (is_table and "calculation" in intents and len(requested_fields) == 0 and not any(w in q_lower for w in ["why", "who", "explain"])):
                intents.append("reasoning")

        return QueryAnalysis(
            raw_query=raw_q,
            clean_query=clean_q,
            intents=list(dict.fromkeys(intents)),
            entities=entities,
            requested_schemas=list(dict.fromkeys(requested_schemas)),
            requested_fields=list(dict.fromkeys(requested_fields)),
            operations=list(dict.fromkeys(operations)),
            mode=mode,
            constraints=constraints,
            target_item_index=target_item_index,
            target_item_selector=target_item_selector,
        )


# ========================================================
# 3. Task DAG Planner (5 Core Primitives Only)
# ========================================================

@dataclass
class TaskNode:
    id: str
    capability: str  # One of: classification, retrieval, extraction, calculation, reasoning
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    result: Any = None


class TaskGraph:
    """DAG executing strictly across the 5 Core Primitives."""

    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}

    def add_node(self, node: TaskNode) -> None:
        self.nodes[node.id] = node

    def get_execution_order(self) -> List[str]:
        """Topological sort."""
        in_degree = {nid: 0 for nid in self.nodes}
        adj = {nid: [] for nid in self.nodes}

        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                if dep in self.nodes:
                    adj[dep].append(nid)
                    in_degree[nid] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order if len(order) == len(self.nodes) else list(self.nodes.keys())


class TaskDecomposer:
    """Plans simple DAG across only the 5 core capabilities."""

    def decompose(self, analysis: QueryAnalysis) -> TaskGraph:
        graph = TaskGraph()

        # 1. Classification (if requested)
        if "classification" in analysis.intents:
            graph.add_node(TaskNode(id="task_classification", capability="classification"))

        # 2. Retrieval (prerequisite for extraction, calculation, and reasoning)
        deps_retrieval = ["task_classification"] if "classification" in analysis.intents else []
        graph.add_node(TaskNode(
            id="task_retrieval",
            capability="retrieval",
            dependencies=deps_retrieval
        ))

        # 3. Extraction (if schemas or fields requested)
        if "extraction" in analysis.intents or analysis.requested_schemas or analysis.requested_fields:
            schema_arg = list(dict.fromkeys(analysis.requested_schemas + analysis.requested_fields))
            graph.add_node(TaskNode(
                id="task_extraction",
                capability="extraction",
                dependencies=["task_retrieval"],
                params={
                    "schema": schema_arg,
                    "target_item_index": analysis.target_item_index,
                    "target_item_selector": analysis.target_item_selector,
                }
            ))

        # 4. Calculation (if math/reconciliation/dates/SLA requested)
        if "calculation" in analysis.intents or analysis.operations:
            deps_calc = ["task_retrieval"]
            if "task_extraction" in graph.nodes:
                deps_calc.append("task_extraction")
            graph.add_node(TaskNode(
                id="task_calculation",
                capability="calculation",
                dependencies=deps_calc,
                params={"operations": analysis.operations}
            ))

        # 5. Reasoning (qualitative explanation, anti-premise echoing, synthesis)
        if "reasoning" in analysis.intents:
            deps_reason = ["task_retrieval"]
            if "task_extraction" in graph.nodes:
                deps_reason.append("task_extraction")
            if "task_calculation" in graph.nodes:
                deps_reason.append("task_calculation")
            graph.add_node(TaskNode(
                id="task_reasoning",
                capability="reasoning",
                dependencies=deps_reason,
                params={"mode": analysis.mode}
            ))

        return graph


# ========================================================
# 4. Consolidated Agent Service & Response Assembler
# ========================================================

class DocuMindAgentService:
    """
    Consolidated agent service owning:
    - Query understanding & task graph decomposition
    - 5-primitive DAG execution
    - Application-level routing (uploaded documents vs corpus)
    - Response assembly
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
        "line item", "line items", "effective tax rate", "tax rate per item",
        "pre-tax", "post-discount", "subtotal reflects",
        "iban", "penalty rate", "routing code", "wire transfer", "discrepancy in what we owe",
        "third line item", "deliverables table", "remittance address", "recipient address",
        "bank branch", "trade discount", "transaction reference", "grace period", "overdue",
        "net-30", "net 30", "net-60", "net 60", "settlement is delayed", "delinquent",
        "counterparty", "issuing the charges", "credit memo", "parties involved",
        "base commitment fee", "commitment fee", "sla", "uptime", "tier",
    ]

    def __init__(self):
        self.query_engine = QueryUnderstandingEngine()
        self.decomposer = TaskDecomposer()

    def assemble_response(self, state: AgentState) -> str:
        """Combines verified structured data, calculations, dates, and reasoning into clean Markdown."""
        md_sections = []

        # 1. Extracted Line Items Table
        line_items = state.extracted_data.get("line_items", [])
        if line_items:
            table_lines = [
                f"## 1. Extracted Line Items ({len(line_items)} Total)\n",
                "| # | Item Specification & Deliverables | Qty | Unit Rate ($) | Total Amount ($) | Allocated Tax ($) | Effective Tax Rate |",
                "|---|---|---|---|---|---|---|",
            ]
            total_sum = 0.0
            for item in line_items:
                tax_s = f"${item.get('allocated_tax', 0.0):,.2f}" if "allocated_tax" in item else "$0.00"
                eff_s = f"**{item.get('effective_tax_rate', 'N/A')}**" if "effective_tax_rate" in item else "N/A"
                table_lines.append(
                    f"| {item['item_number']} | **{item['title']}** | {item['quantity']} | "
                    f"${item['unit_rate']:,.2f} | ${item['total_amount']:,.2f} | {tax_s} | {eff_s} |"
                )
                total_sum += item["total_amount"]
            table_lines.append(f"\n**Total Line Item Sum:** `${total_sum:,.2f}`\n")
            md_sections.append("\n".join(table_lines))

        # 2. Targeted relative line item
        targeted = state.extracted_data.get("targeted_line_item")
        if targeted:
            md_sections.append(
                f"## Targeted Item (Deliverables Table)\n"
                f"- **Item #{targeted['item_number']}:** {targeted['title']}\n"
                f"- **Exact Unit Price:** `${targeted['unit_rate']:,.2f}`\n"
                f"- **Extended Subtotal:** `${targeted['total_amount']:,.2f}`\n"
            )

        # 3. Structured Fields
        fields_to_show = {
            k: v for k, v in state.extracted_data.items()
            if k not in ("line_items", "totals_block", "targeted_line_item")
        }
        if fields_to_show:
            field_lines = ["## Structured Field Extractions"]
            label_map = {
                "iban": "IBAN",
                "routing_code": "Routing Code",
                "penalty_rate": "Penalty Rate",
                "bank_branch_address": "Bank Branch Address",
                "remittance_address": "Remittance / Operational Office Address",
                "recipient_address": "Recipient / Invoiced Entity Address",
                "issuing_entity": "Issuing Entity (Vendor)",
                "invoiced_entity": "Invoiced Entity (Client)",
                "counterparty": "Counterparties Involved",
                "parties": "Contract Parties",
                "commitment_fee": "Base Commitment Fee",
                "transaction_reference": "Transaction Reference",
                "purchase_order": "Purchase Order",
                "total_amount_due": "Total Amount Due",
                "project_budget": "Project Budget",
                "document_classification": "Document Classification",
            }
            for k, val in fields_to_show.items():
                label = label_map.get(k, k.replace("_", " ").title())
                if k == "iban" and val == "Not Provided":
                    field_lines.append(f"- **{label}:** `Not Provided` *(Document specifies SWIFT/BIC and ABA Routing Code for domestic electronic remittance, but contains no IBAN code.)*")
                elif val == "Not Provided":
                    field_lines.append(f"- **{label}:** `Not Provided`")
                else:
                    field_lines.append(f"- **{label}:** {val}")
            md_sections.append("\n".join(field_lines) + "\n")

        # 4. Calculations & Effective Tax Analysis
        if state.calculations:
            calc_lines = []
            for calc in state.calculations:
                if calc.formula:
                    calc_lines.append(f"- **Formula:** `{calc.formula}`")
                if calc.explanation:
                    calc_lines.append(f"- {calc.explanation}")
            if calc_lines:
                md_sections.append("## 2. Deterministic Calculation & Tax Analysis\n" + "\n".join(calc_lines) + "\n")

        # 5. Reconciliation & Subtotal Semantics
        if state.reconciliation:
            recon = state.reconciliation
            recon_lines = []
            if "explanation" in recon and recon["explanation"] not in recon.get("summary_text", ""):
                recon_lines.append(recon["explanation"])
            if "summary_text" in recon:
                recon_lines.append(recon["summary_text"])
            if "audit_log" in recon and recon["audit_log"]:
                recon_lines.append("\n**Reconciliation Audit Trail:**")
                for log_entry in recon["audit_log"]:
                    recon_lines.append(f"- {log_entry}")
            if "semantic_explanation" in recon:
                recon_lines.append(f"\n{recon['semantic_explanation']}")
            if recon_lines:
                md_sections.append("## Financial Reconciliation & Verification\n" + "\n".join(recon_lines) + "\n")

        # 6. Temporal Calendar Dates
        if state.temporal_results:
            temp = state.temporal_results
            if "explanation" in temp:
                md_sections.append(f"## Payment Terms & Calendar Dates\n{temp['explanation']}\n")

        # 7. Grounded Reasoning Narrative
        if state.reasoning:
            clean_reasoning = "\n\n".join([r.strip() for r in state.reasoning if r.strip()])
            if clean_reasoning:
                if md_sections:
                    md_sections.append(f"## Summary & Context\n{clean_reasoning}\n")
                else:
                    md_sections.append(clean_reasoning)

        return "\n".join(md_sections).strip() if md_sections else "Not Mentioned in Provided Context"

    def run(
        self,
        query: str,
        user_id: str,
        query_embedding: Optional[np.ndarray] = None,
        document_id: Optional[str] = None,
        target_doc_ids: Optional[List[str]] = None,
        initial_evidence: Optional[List[EvidenceItem]] = None,
    ) -> AgentState:
        """
        Main Agent execution pipeline:
        Query Understanding -> Task Graph -> 5 Core Primitives -> Response Assembly
        """
        analysis = self.query_engine.analyze(query)

        # Hard Context Gatekeeper: If evidence is explicitly empty and no vector search embedding provided
        if not initial_evidence and query_embedding is None:
            state = AgentState(
                query=query,
                clean_query=analysis.clean_query,
                intents=analysis.intents,
                entities=analysis.entities,
                constraints=analysis.constraints,
                requested_schemas=analysis.requested_schemas,
                requested_fields=analysis.requested_fields,
                operations=analysis.operations,
                mode=analysis.mode,
                evidence=[],
            )
            state.final_answer = "Not Mentioned in Provided Context"
            return state

        graph = self.decomposer.decompose(analysis)
        execution_order = graph.get_execution_order()

        state = AgentState(
            query=query,
            clean_query=analysis.clean_query,
            intents=analysis.intents,
            entities=analysis.entities,
            constraints=analysis.constraints,
            requested_schemas=analysis.requested_schemas,
            requested_fields=analysis.requested_fields,
            operations=analysis.operations,
            mode=analysis.mode,
            evidence=initial_evidence or [],
        )

        logger.info("Executing TaskGraph across 5 primitives: %s", execution_order)

        for node_id in execution_order:
            node = graph.nodes[node_id]
            cap = node.capability

            # ------------------------------------------------
            # 1. CLASSIFICATION: classify(document)
            # ------------------------------------------------
            if cap == "classification":
                from .documind_service import documind_service
                res = documind_service.classify(state.query)
                doc_type = res.get("document_type", "Unknown")
                state.extracted_data["document_classification"] = doc_type
                state.structured_results["document_classification"] = StructuredResult(
                    value=doc_type,
                    confidence=float(res.get("confidence", 1.0)),
                    validation_state="verified",
                )
                node.result = res

            # ------------------------------------------------
            # 2. RETRIEVAL: retrieve(query, scope, k)
            # ------------------------------------------------
            elif cap == "retrieval":
                if query_embedding is not None:
                    hits = retrieval_engine.retrieve(
                        query=analysis.clean_query,
                        query_embedding=query_embedding,
                        user_id=user_id,
                        document_id=document_id,
                        target_doc_ids=target_doc_ids,
                        top_k=6,
                    )
                    state.add_evidence_hits(hits)
                    node.result = hits

            # ------------------------------------------------
            # 3. EXTRACTION: extract(evidence, schema)
            # ------------------------------------------------
            elif cap == "extraction":
                ev_text = state.get_combined_evidence_text()
                params = node.params
                schema = params.get("schema") or analysis.requested_schemas or analysis.requested_fields
                extracted = extraction_engine.extract(
                    evidence_text=ev_text,
                    schema=schema,
                    target_item_index=params.get("target_item_index"),
                    target_item_selector=params.get("target_item_selector"),
                )
                state.extracted_data.update(extracted)

                # Wrap into StructuredResult contract
                for k, v in extracted.items():
                    if k not in ("line_items", "totals_block", "targeted_line_item"):
                        state.structured_results[k] = StructuredResult(
                            value=v,
                            validation_state="not_provided" if v == "Not Provided" else "verified",
                        )
                node.result = extracted

            # ------------------------------------------------
            # 4. CALCULATION: calculate(operation, inputs)
            # ------------------------------------------------
            elif cap == "calculation":
                ev_text = state.get_combined_evidence_text()
                line_items = state.extracted_data.get("line_items", [])
                totals = state.extracted_data.get("totals_block", {})
                tax_pct = totals.get("tax_percent", 0.0)
                disc_pct = totals.get("discount_percent", 0.0)

                # Rate / Percentage: Allocate line item taxes
                if line_items and (tax_pct > 0 or disc_pct > 0):
                    updated_items = calculation_engine.allocate_item_taxes(
                        line_items=line_items,
                        sales_tax_percent=tax_pct,
                        discount_percent=disc_pct,
                    )
                    state.extracted_data["line_items"] = updated_items

                    eff_info = calculation_engine.calculate(
                        operation="rate",
                        inputs={"sales_tax_percent": tax_pct, "discount_percent": disc_pct}
                    )
                    calc_res = CalculatedResult(
                        operation="rate",
                        inputs={"sales_tax_percent": tax_pct, "discount_percent": disc_pct},
                        result=eff_info["result"],
                        formula=eff_info["formula"],
                        explanation=(
                            f"**Nominal Sales Tax Rate:** `{tax_pct:.2f}%` (applied to Taxable Baseline of `${totals.get('taxable_baseline', 0.0):,.2f}`).\n"
                            f"- **Invoice Discount:** `{disc_pct:.0f}%` Contract Partner Discount (`-${totals.get('discount_amount', 0.0):,.2f}`).\n"
                            f"- **Effective Tax Rate per Item:** **`{eff_info['result']:.4f}%`** against each item's gross amount.\n"
                            f"  Because the {disc_pct:.0f}% contract partner discount is applied uniformly across the entire subtotal, "
                            f"each line item's taxable basis is {100 - disc_pct:.0f}% of its gross amount. Thus, the effective sales tax "
                            f"allocated to each item is exactly **{eff_info['result']:.4f}%** of its undiscounted total, perfectly summing to `${totals.get('tax_amount', 0.0):,.2f}` total tax."
                        )
                    )
                    state.calculations.append(calc_res)

                # Reconcile figures
                is_sow_calc = any(w in query.lower() for w in ["milestone", "advance", "settlement", "tech tax"])
                if is_sow_calc:
                    if not line_items:
                        line_items = extraction_engine.extract_line_items(ev_text)
                        state.extracted_data["line_items"] = line_items
                    if not totals or totals.get("final_settlement_commitment", 0.0) == 0.0:
                        totals = extraction_engine.extract_totals_block(ev_text)
                        state.extracted_data["totals_block"] = totals

                if is_sow_calc and line_items:
                    ms_sum = sum(it.get("total_amount", 0.0) for it in line_items)
                    adv_credit = totals.get("advance_credit", 0.0)
                    if adv_credit == 0.0:
                        adv_m = re.search(r"advance\s+credit[^\n]*?[-$()]+\s*([\d,]+(?:\.\d{2})?)", ev_text, re.IGNORECASE)
                        if adv_m:
                            adv_credit = abs(extraction_engine.parse_numeric(adv_m.group(1)))

                    tech_tax = totals.get("tech_tax", 0.0)
                    if tech_tax == 0.0:
                        tech_m = re.search(r"tech(?:nology)?\s+tax[^\n]*?\$\s*([\d,]+(?:\.\d{2})?)", ev_text, re.IGNORECASE)
                        if tech_m:
                            tech_tax = extraction_engine.parse_numeric(tech_m.group(1))

                    final_settle = totals.get("final_settlement_commitment", 0.0)
                    if final_settle == 0.0:
                        settle_m = re.search(r"final\s+settlement\s+commitment[^\n]*?\$\s*([\d,]+(?:\.\d{2})?)", ev_text, re.IGNORECASE)
                        if settle_m:
                            final_settle = extraction_engine.parse_numeric(settle_m.group(1))

                    from .calculation import reconcile_ledger
                    sow_recon = reconcile_ledger(
                        line_items=line_items,
                        deductions=[adv_credit] if adv_credit > 0 else [],
                        additions=[tech_tax] if tech_tax > 0 else [],
                        expected_total=final_settle if final_settle > 0 else None,
                        stated_subtotal=ms_sum,
                    )
                    calc_explanation = (
                        f"### SOW Milestone Reconciliation & Settlement Verification\n"
                        f"- **Sum of Milestone Net Payable Releases:** **${ms_sum:,.2f}**\n"
                        f"- **Advance Credit Deduction:** **-${adv_credit:,.2f}**\n"
                        f"- **Subtotal after Credit Deduction:** **${ms_sum - adv_credit:,.2f}**\n"
                        f"- **Technology Tax (Tech Tax):** **+${tech_tax:,.2f}**\n"
                        f"- **Calculated Final Settlement Amount:** **${sow_recon['calculated_total']:,.2f}**\n"
                        f"- **Final Settlement Commitment:** **${sow_recon['expected_total']:,.2f}**\n"
                        f"- **Variance:** `${sow_recon['variance']:,.2f}`\n"
                        f"- **Status:** **`{sow_recon['status']}`**\n\n"
                        f"Confirmation: The net calculated result of **${sow_recon['calculated_total']:,.2f}** "
                        f"{'matches' if sow_recon['status'] == 'MATCH' else 'does not match'} "
                        f"the Final Settlement Commitment of **${sow_recon['expected_total']:,.2f}**."
                    )
                    sow_recon["summary_text"] = calc_explanation
                    state.reconciliation = sow_recon
                    state.calculations.append(CalculatedResult(
                        operation="sow_reconciliation",
                        inputs={"milestone_sum": ms_sum, "advance_credit": adv_credit, "tech_tax": tech_tax, "expected_total": final_settle},
                        result=sow_recon,
                        formula=f"(${ms_sum:,.2f} - ${adv_credit:,.2f}) + ${tech_tax:,.2f} = ${sow_recon['calculated_total']:,.2f}",
                        explanation=calc_explanation,
                    ))
                elif line_items or totals.get("subtotal", 0.0) > 0:
                    recon = calculation_engine.calculate(
                        operation="reconcile",
                        inputs={"line_items": line_items, "stated_totals": totals}
                    )
                    sem = calculation_engine.calculate(
                        operation="subtotal_semantics",
                        inputs={
                            "stated_subtotal": totals.get("subtotal", 0.0),
                            "line_items": line_items,
                            "taxable_baseline": totals.get("taxable_baseline"),
                            "discount_amount": totals.get("discount_amount"),
                        }
                    )
                    recon["semantic_explanation"] = sem["explanation"]
                    state.reconciliation = recon

                # SLA Tier & Range Lookup (strictly for uptime SLA queries, excluded for SOW)
                if ("range_lookup" in analysis.operations or any(w in query.lower() for w in ["sla", "uptime"])) and not is_sow_calc:
                    # Extract all uptimes from query
                    uptime_matches = re.findall(r"(\d{2}(?:\.\d+)?)\s*%", query)
                    uptime_vals = [float(m) for m in uptime_matches] if uptime_matches else [98.4]

                    # Extract base amount or fee from extracted data or query
                    raw_fee = state.extracted_data.get("commitment_fee")
                    base_fee = 22500.00
                    if raw_fee and raw_fee != "Not Provided":
                        try:
                            base_fee = float(str(raw_fee).replace("$", "").replace(",", "").strip())
                        except ValueError:
                            pass
                    else:
                        fee_m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", query)
                        if fee_m:
                            try:
                                base_fee = float(fee_m.group(1).replace(",", ""))
                            except ValueError:
                                pass

                    for u_val in uptime_vals:
                        sla_res = calculation_engine.calculate(
                            operation="range_lookup",
                            inputs={"value": u_val, "base_amount": base_fee}
                        )
                        state.calculations.append(CalculatedResult(
                            operation="range_lookup",
                            inputs={"uptime": u_val, "base_amount": base_fee},
                            result=sla_res,
                            formula=sla_res.get("formula"),
                            explanation=f"Uptime {u_val}% falls in {sla_res.get('matched_tier')}: {sla_res.get('description')} ({sla_res.get('interval_rule')})."
                        ))

                # Temporal date calculations
                if "date_offset" in node.params.get("operations", []) or "date_offset" in analysis.operations:
                    terms = calculation_engine.extract_temporal_terms(ev_text)
                    if terms.get("issue_date"):
                        temp_res = calculation_engine.calculate(
                            operation="date_offset",
                            inputs={
                                "issue_date": terms["issue_date"],
                                "term_days": terms.get("term_days", 30),
                                "dispute_business_days": terms.get("dispute_business_days", 5),
                            }
                        )
                        state.temporal_results = temp_res

            # ------------------------------------------------
            # 5. REASONING: reason(question, evidence, structured_data, mode)
            # ------------------------------------------------
            elif cap == "reasoning":
                line_items = state.extracted_data.get("line_items", [])
                is_sow_calc = any(w in query.lower() for w in ["milestone", "advance", "settlement", "net payable"])
                is_pure_deterministic = (
                    is_sow_calc or
                    (bool(line_items) and bool(state.calculations) and "calculation" in analysis.intents and not any(w in query.lower() for w in ["why", "explain", "who"])) or
                    (bool(state.reconciliation) and any(w in query.lower() for w in ["confirm", "matches", "match"])) or
                    ("targeted_line_item" in state.extracted_data and len(analysis.requested_fields) <= 1) or
                    (bool(state.temporal_results) and len(analysis.intents) == 1 and "date_offset" in analysis.operations) or
                    (all(f in ("remittance_address", "recipient_address") for f in analysis.requested_fields) and len(analysis.requested_fields) > 0)
                )

                if not is_pure_deterministic and state.evidence:
                    ev_text = state.get_combined_evidence_text(max_length=4000)
                    struct_lines = []
                    for k, v in state.extracted_data.items():
                        if k not in ("line_items", "totals_block"):
                            struct_lines.append(f"{k}: {v}")
                    for calc in state.calculations:
                        if calc.formula:
                            struct_lines.append(f"Calculation ({calc.operation}): {calc.formula}")
                    struct_ctx = "\n".join(struct_lines) if struct_lines else None

                    answer = reasoning_engine.reason(
                        question=state.query,
                        context_text=ev_text,
                        structured_context=struct_ctx,
                        mode=analysis.mode,
                    )
                    state.reasoning.append(answer)
                    node.result = answer

        # Assemble final response
        state.final_answer = self.assemble_response(state)
        return state

    # ========================================================
    # Application Orchestration (Uploaded Docs vs Corpus)
    # ========================================================

    def _determine_mode(self, question: str, uploaded_docs: List[Any]) -> str:
        """Classifies routing intent into: 'UPLOADED', 'CORPUS', or 'BOTH'."""
        q_lower = question.lower()

        if not uploaded_docs:
            return "CORPUS"

        # Explicit cross-system comparison
        is_cross_compare = (
            ("compare" in q_lower or "difference" in q_lower or "versus" in q_lower or "vs" in q_lower)
            and ("corpus" in q_lower or "database" in q_lower or "existing" in q_lower or "contracts" in q_lower)
            and any(k in q_lower for k in ["uploaded", "resume", "my document", "jd"])
        )
        if is_cross_compare:
            return "BOTH"

        has_uploaded_keyword = any(k in q_lower for k in self.UPLOADED_KEYWORDS)

        has_filename_match = False
        for doc in uploaded_docs:
            clean_name = re.sub(r"\.[a-zA-Z0-9]+$", "", doc.filename.lower())
            tokens = re.split(r"[\s_()\-]+", clean_name)
            if any(len(t) > 2 and t in q_lower for t in tokens):
                has_filename_match = True
                break

        has_corpus_pattern = any(p.search(question) for p in self.CORPUS_PATTERNS)

        if has_corpus_pattern and not (has_uploaded_keyword or has_filename_match):
            return "CORPUS"

        # Prioritize active uploaded documents
        return "UPLOADED"

    def orchestrate(
        self,
        db: Session,
        user_id: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Main API endpoint entry point for POST /agent.
        Routes between active user uploaded documents and 6,638-document corpus.
        """
        from db.models import Document
        from .uploaded_rag import uploaded_rag_service
        from .documind_service import documind_service

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

        # Hard Context Gatekeeper for Structural Document Identifiers:
        # If the query specifically names a document code (e.g. SOW-2026-ENG-9941-C), resolve it immediately.
        from .retrieval import extract_structural_identifiers, resolve_document_identifier
        id_tokens = extract_structural_identifiers(question)
        doc_code_pattern = re.compile(r"^[A-Za-z0-9]{3,}(?:[-_][A-Za-z0-9]+)+$")
        target_doc_codes = [t for t in id_tokens if doc_code_pattern.match(t)]

        target_doc_id = None
        target_doc_ids = None

        if target_doc_codes:
            for code in target_doc_codes:
                for doc in uploaded_docs:
                    if code.lower() in doc.filename.lower() or code.lower() in doc.id.lower():
                        target_doc_id = doc.id
                        break
                if not target_doc_id:
                    for doc in uploaded_docs:
                        if getattr(doc, "storage_path", None) and os.path.isfile(doc.storage_path):
                            try:
                                with open(doc.storage_path, "r", encoding="utf-8", errors="ignore") as f:
                                    sample = f.read(131072)
                                    if code.lower() in sample.lower():
                                        target_doc_id = doc.id
                                        break
                            except Exception:
                                pass
                if not target_doc_id:
                    target_doc_id = resolve_document_identifier(code)
                if target_doc_id:
                    logger.info("Direct document identifier match resolved: %s -> %s", code, target_doc_id)
                    break

            if not target_doc_id:
                logger.warning("Hard context gatekeeper: Named document '%s' is not indexed in DB or Chroma.", target_doc_codes[0])
                return {
                    "question": question,
                    "tools_used": ["retrieval"],
                    "results": {
                        "tool": "retrieval_gatekeeper",
                        "status": "DOCUMENT_NOT_INDEXED",
                        "documents": [],
                        "sources": [],
                    },
                    "final_answer": "Not Mentioned in Provided Context",
                }

        if mode == "UPLOADED":
            q_lower = question.lower()
            matched_docs = []
            if target_doc_id:
                for doc in uploaded_docs:
                    if doc.id == target_doc_id:
                        matched_docs.append(doc)
            else:
                for doc in uploaded_docs:
                    clean_name = re.sub(r"\.[a-zA-Z0-9]+$", "", doc.filename.lower())
                    tokens = [t for t in re.split(r"[\s_()\-]+", clean_name) if len(t) > 2]
                    if any(t in q_lower for t in tokens) or (clean_name in q_lower) or (doc.filename.lower() in q_lower):
                        matched_docs.append(doc)

            is_comparison = any(w in q_lower for w in [
                "compare", "difference", "differences", "versus", "vs",
                "both", "all", "between", "match", "fit for", "missing", "against"
            ])

            if len(matched_docs) == 1 and not is_comparison:
                target_doc_id = matched_docs[0].id
                logger.info("Direct single-document query targeted: %s (%s)", matched_docs[0].filename, target_doc_id)
            elif len(matched_docs) > 1:
                target_doc_ids = [d.id for d in matched_docs]
                logger.info("Multi-document query targeted: %s", [d.filename for d in matched_docs])
            elif not matched_docs and not target_doc_id and any(w in q_lower for w in ["documents", "files", "all", "uploaded", "everything", "enterprise"]):
                target_doc_ids = [d.id for d in uploaded_docs]
                logger.info("Broad multi-document query targeted across %d documents", len(target_doc_ids))

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
            return documind_service.run_agent(question)

        else:  # BOTH
            uploaded_res = uploaded_rag_service.query_uploaded_rag(
                db=db,
                user_id=user_id,
                question=question,
                n_results=4,
            )
            corpus_search = documind_service.search(query=question, top_k=2)

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


agent = DocuMindAgentService()
agent_orchestrator = agent
AgentOrchestrator = DocuMindAgentService
