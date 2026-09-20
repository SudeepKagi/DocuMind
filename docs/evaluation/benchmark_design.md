# DocuMind AI/ML Evaluation Benchmark Design

This document details the architectural design, methodology, query taxonomy, and evaluation rubrics for the quantitative evaluation framework in **DocuMind**.

---

## 1. Objectives and Core Principles

The DocuMind evaluation suite provides a rigorous, reproducible, and transparent assessment of the 6 core ML/AI components in the document intelligence pipeline.

### Design Principles
1. **Zero Contamination / Local Execution:** All evaluations run entirely locally against pre-existing models, checkpoints, and vector stores without modifying weights, pipelines, or cloud services.
2. **Honest & Unfabricated Ground Truth:** Every benchmark item is rooted in verified document text, validated test splits, or manually curated enterprise queries. If ground truth is unavailable for any dimension, it is explicitly reported as `Not Evaluated` rather than synthetic guesswork.
3. **Multi-Faceted Metrics:** Every component is evaluated through multiple complementary angles (e.g., lexical vs semantic, exact match vs soft similarity, token-level vs field-level, retrieval vs generation).
4. **Latency Deconstruction:** RAG latency is disaggregated into retrieval, prompt assembly, and autoregressive generation stages to provide actionable system profiling.

---

## 2. Benchmark Suites

```
DocuMind Pipeline:
Documents -> [1. Classification] -> [2. Metadata Extraction] -> [3. Retrieval] -> [4. Grounded RAG]
                                                                        ^
                                                                        |
                                              [5. Agent Orchestrator / Tool Routing]
                                                                        |
                                              [6. Uploaded Document RAG (ChromaDB)]
```

---

### Benchmark 1: Document Classification
- **Target Model:** Fine-tuned `distilbert-base-uncased` (`ml/models/distilbert_doc_classifier/final`)
- **Baseline Model:** TF-IDF (10,000 features, sublinear TF) + Multinomial Logistic Regression (`ml/models/tfidf_logistic_regression`)
- **Dataset:** `ml/processed/test.csv` (996 held-out enterprise documents)
- **Classes (5):** `Email`, `Contract`, `Invoice`, `Purchase Order`, `Report`
- **Metrics Computed:**
  - Overall Accuracy
  - Macro-averaged Precision, Recall, F1
  - Weighted-averaged Precision, Recall, F1
  - Per-class Precision, Recall, F1, Support
  - $5 \times 5$ Confusion Matrix
- **Key Insight:** Evaluates whether fine-tuned transformer attention delivers measurable advantages over standard n-gram linear models on long-tail visual document text.

---

### Benchmark 2: Invoice Metadata Extraction
- **Target Model:** Token-classification LayoutLM fine-tuned on multi-label invoice entities (`ml/models/layoutlm_multilabel/final`)
- **Dataset:** `ml/processed/invoice_field_quality.csv` (30,823 field instances across DocILE-annotated invoices)
- **Target Fields (7):**
  1. `vendor_name`
  2. `vendor_address`
  3. `customer_billing_name`
  4. `customer_billing_address`
  5. `date_issue`
  6. `amount_total_gross`
  7. `amount_due`
- **Evaluation Levels:**
  - **Document / Field-Level Evaluation:** Compares the final reconstructed field string against ground truth.
  - **Exact Match (EM):** Case-insensitive, whitespace-trimmed string identity.
  - **Character Similarity:** Normalized Levenshtein ratio:
    $$\text{Sim}(s_1, s_2) = \frac{2 \cdot M}{|s_1| + |s_2|}$$
    where $M$ is the number of matching characters.
  - **Success @ Threshold (0.80):** Percentage of extractions with similarity $\ge 0.80$, capturing OCR typos and minor formatting divergences.
  - **Token F1:** Overlap of unigram word tokens between prediction and target.

---

### Benchmark 3: Enterprise Hybrid Retrieval
- **Target Systems:**
  - **Sparse:** BM25S (lexical inverted index, $k_1=1.5, b=0.75$) over 223,234 document chunks
  - **Dense:** BGE-small-en-v1.5 (384-dim dense semantic embeddings) over 223,234 document chunks
  - **Hybrid:** Reciprocal Rank Fusion (RRF, $k=60$) combining BM25S and BGE scores
- **Corpus Files:**
  - `ml/processed/full_text_chunks.parquet`
  - `ml/processed/full_text_embeddings.npy`
  - `ml/processed/full_text_bm25s`
- **Benchmark Dataset:** `evaluation/retrieval/retrieval_benchmark.json` (15 queries)
- **Taxonomy of Queries:**
  - **Exact Keyword:** Specific transaction IDs, exact part codes (`"PO-98214"`, `"INV-2023-0491"`).
  - **Semantic / Paraphrase:** Conceptual queries without lexical overlap (`"penalties for late completion of milestones"`, `"force majeure pandemics"`).
  - **Contract Clauses:** Legal provisions (`"confidentiality duration obligations"`, `"indemnification IP infringement limits"`).
  - **Invoice Queries:** Financial lookups (`"wire transfer routing instructions"`, `"net 30 payment terms discount"`).
  - **Purchase Order Queries:** Procurement items (`"freight shipping FOB destination terms"`).
  - **Discrepancy Queries:** Queries where BM25 and Dense search diverge to highlight hybrid fusion value.
- **Metrics:**
  - **Recall@K ($K \in \{1, 3, 5, 10\}$):** Proportion of expected relevant chunks retrieved in top-$K$.
  - **MRR (Mean Reciprocal Rank):** $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$.
  - **nDCG@10:** Normalized Discounted Cumulative Gain accounting for graded relevance.

---

### Benchmark 4: Grounded RAG & Question Answering
- **Generator Model:** `Qwen/Qwen2.5-1.5B-Instruct` (AutoModelForCausalLM, greedy decoding, max 256 tokens)
- **Retriever:** Enterprise Hybrid Search (top-5 chunks as context)
- **Benchmark Dataset:** `evaluation/rag/rag_benchmark.json` (14 enterprise QA items)
- **Question Categories:**
  1. `factual_extraction`: Single-fact retrieval (dates, amounts, addresses).
  2. `explanation`: Procedural or explanatory questions (termination steps, dispute procedures).
  3. `calculation`: Quantitative questions (sums, quantities, unit prices).
  4. `definition`: Legal definitions ("Termination Event", "Confidential Information").
  5. `multi_hop`: Inferences requiring multiple chunks or connected statements.
  6. `enumeration`: Multi-item lists (warranties, deliverables, covered expenses).
- **Evaluation Dimensions:**
  - **Token F1:** Unigram precision, recall, and harmonic mean between candidate and reference answer.
  - **Exact Match:** Strict identity check for exact numerical/name responses.
  - **Retrieval Hit Rate:** Fraction of queries where at least one ground-truth chunk/document was in the retrieved context.
  - **Groundedness / Completeness (Rubric):** Fraction of expected core facts and keywords present in generated answer.
  - **Hallucination Rate:** Frequency of unsupported claims conflicting with retrieved chunks.
  - **Latency Profiling:** Retrieval latency, LLM generation latency, tokens per second, and end-to-end latency.

---

### Benchmark 5: Agent Tool Routing & Orchestration
- **Target Orchestrator:** Rule-based Agent Planner (`agents/orchestrator.py`)
- **Available Tools:**
  - `document_classifier`
  - `metadata_extractor`
  - `corpus_search`
  - `qa_rag`
- **Benchmark Dataset:** `evaluation/agent/agent_benchmark.json` (20 representative enterprise queries)
- **Query Categories:**
  1. `classification_only`: Requests to classify document type ("What type of document is this?").
  2. `metadata_only`: Structured extraction requests ("Extract vendor name and total due").
  3. `search_only`: Pure information retrieval ("Find all documents mentioning Force Majeure").
  4. `rag_qa`: In-depth question answering ("Explain the warranty terms in contract 42").
  5. `multi_tool`: Chained workflows ("Classify this document, then extract its metadata").
  6. `comparison`: Cross-document analysis ("Compare payment terms in invoice A vs invoice B").
- **Metrics:**
  - **Overall Routing Accuracy:** Percentage of queries where exact expected tool sequence was planned.
  - **Single-Tool Accuracy:** Precision on atomic 1-step queries.
  - **Multi-Tool Routing Accuracy:** Correct ordering and completeness on compound queries.
  - **Invalid Tool Call Rate:** Rate of executing unnecessary or hallucinated tools.

---

### Benchmark 6: Uploaded Document RAG (ChromaDB)
- **Vector Store:** ChromaDB (`data/chromadb`) using `bge-small-en-v1.5` embeddings
- **Formats Tested:** `.pdf`, `.docx`, `.txt`, `.eml`
- **Benchmark Dataset:** `evaluation/uploaded_rag/uploaded_rag_benchmark.json` (8 questions)
- **Query Scenarios:**
  - Single-document factual extraction across each file format.
  - Multi-document queries spanning multiple files or cross-file syntheses.
- **Metrics:**
  - Retrieval Recall@K ($K=1, 3, 5$)
  - Source Document Attribution Accuracy (correct document metadata returned)
  - Answer Quality (Token F1, Keyword Coverage)
  - End-to-end Latency
