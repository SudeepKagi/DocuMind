# DocuMind | Enterprise Document Intelligence

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2019-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Bundler-Vite%208-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%2016-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-FF6F00?style=flat)](https://www.trychroma.com)
[![PyTorch](https://img.shields.io/badge/ML%20Core-PyTorch%20%7C%20CUDA-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Models-DistilBERT%20%7C%20LayoutLM%20%7C%20BGE%20%7C%20Qwen2.5-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)

DocuMind is a full-stack enterprise document intelligence platform designed around modular, reusable pipelines:

$$\text{Document Ingestion} \longrightarrow \text{Document Classification} \longrightarrow \text{Metadata Extraction} \longrightarrow \text{Hybrid Retrieval} \longrightarrow \text{Grounded RAG} \longrightarrow \text{Agentic Orchestration} \longrightarrow \text{Uploaded-Document Intelligence}$$

Rather than being tied to a single fixed document or static dataset, DocuMind provides an extensible, service-oriented architecture capable of processing both historical institutional corpora and dynamic, multi-tenant user uploads through unified agentic orchestration.

---

## Technology Stack

- **Frontend:** React 19, Vite 8, Vanilla CSS Design System with dark SaaS tokens
- **Backend:** FastAPI, Python 3.11+, SQLAlchemy 2.0, Alembic, PostgreSQL 16, ChromaDB
- **ML / AI Components:**
  - **Document Classification:** Fine-tuned `distilbert-base-uncased` (5 classes)
  - **Structured Extraction:** Fine-tuned `LayoutLM` multi-label token classifier + deterministic spatial extractors
  - **Dense Semantic Retrieval:** `BAAI/bge-small-en-v1.5` (384-dimensional embeddings)
  - **Sparse Lexical Retrieval:** `BM25S` inverted index with token-level BM25 scoring
  - **Generative Language Model:** `Qwen/Qwen2.5-1.5B-Instruct` for context-grounded synthesis
- **Infrastructure & Tooling:** PyTorch (CUDA / CPU), Hugging Face Transformers, Docker & Docker Compose

---

## System Architecture

DocuMind cleanly decouples user interaction, API routing, agent orchestration, machine learning inference, and storage:

```mermaid
flowchart TD
    User([Enterprise User]) --> UI[React 19 + Vite Frontend]
    UI --> API[FastAPI Application Backend]
    
    API --> AgentLayer[Agent & Orchestration Layer]
    AgentLayer --> Classify[Document Classifier<br/>DistilBERT]
    AgentLayer --> Extract[Metadata Extractor<br/>LayoutLM + Heuristics]
    AgentLayer --> Hybrid[Hybrid Search Engine<br/>BM25S + BGE Embeddings]
    AgentLayer --> GroundedRAG[Grounded Enterprise RAG<br/>Qwen2.5-1.5B-Instruct]
    AgentLayer --> UploadRAG[Uploaded Document RAG<br/>Parser + BGE + ChromaDB]
    
    subgraph DataLayer["Data & Index Persistence Layer"]
        PG[(PostgreSQL 16<br/>Users, Document Records, File Metadata)]
        ChromaStore[(ChromaDB Vector Store<br/>Multi-Tenant Upload Embeddings)]
        CorpusIndex[(Corpus Index<br/>223k Passages in BM25S + BGE)]
    end
    
    API --> PG
    Hybrid --> CorpusIndex
    GroundedRAG --> CorpusIndex
    UploadRAG --> ChromaStore
```

### Data Layer Roles
- **PostgreSQL 16:** System of record for user accounts, document metadata, file lifecycle statuses, chunk indices, and multi-tenant permissions.
- **ChromaDB:** Isolated vector storage for user-uploaded document chunks, indexed with cosine distance and filtered by `user_id`.
- **Local BM25 / BGE Indexes:** Dual lexical and dense representation covering the pre-indexed enterprise knowledge corpus (223,234 passages).

---

## Supported Document Types & Formats

The architecture explicitly differentiates between document classification categories and file ingestion formats:

- **Classification Categories (Fine-Tuned DistilBERT):**
  - `Contract` (commercial agreements, NDAs, master service agreements)
  - `Invoice` (standard billing, vendor invoices, credit memos)
  - `Purchase Order` (procurement authorizations, line-item orders)
  - `Email` (internal corporate communications, notices, correspondence)
  - `Report` (operational reviews, SEC filings, audit reports)

- **Uploaded Document Ingestion Formats (Multi-Format Parsers):**
  - **PDF (`.pdf`):** Digital text stream extraction with exact page-level mapping via `pypdf`.
  - **Word (`.docx`):** Structural paragraph and table traversal via `python-docx`.
  - **Plain Text (`.txt`):** Structured line and block tokenization with UTF-8 / Latin-1 encoding detection.
  - **Email (`.eml`):** Multipart message parsing with header isolation (`From`, `To`, `Subject`, `Date`) and body decoding.

---

## Key Features

1. **AI Document Classification:** Identifies document types using fine-tuned DistilBERT with sliding-window multi-chunk pooling.
2. **Structured Metadata Extraction:** Reconstructs financial amounts, billing entities, dates, and order numbers using token-level spatial LayoutLM and deterministic rules.
3. **Hybrid Semantic + Lexical Retrieval:** Combines sparse BM25S lexical precision with dense BGE semantic matching using Reciprocal Rank Fusion (RRF, $k=60$).
4. **Grounded Question Answering:** Generates context-bounded answers using Qwen2.5-1.5B-Instruct conditioned strictly on retrieved evidence.
5. **Multi-Tool Agent Orchestration:** Automatically identifies user intent (`CORPUS`, `UPLOADED`, or `BOTH`), selecting and chaining classification, extraction, search, and RAG tools.
6. **Multi-Format Document Ingestion:** Parses PDFs, Word documents, text files, and email files with page-level structural preservation.
7. **ChromaDB-Backed Uploaded RAG:** Provides isolated, on-the-fly semantic search and synthesis over dynamic user uploads.
8. **PostgreSQL Relational Management:** Manages document lifecycles, user ownership, chunk counts, and file references with relational audit integrity.
9. **Automated Evaluation Framework:** Comprehensive, reproducible evaluation suites measuring accuracy, F1, retrieval recall, and latency across all components.

---

## Evaluation & Benchmarks

DocuMind includes a fully automated, reproducible evaluation framework that executes against local model checkpoints, processed datasets, and vector indexes without cloud API dependencies.

For complete methodologies, per-class breakdowns, confusion matrices, and latency graphs:
- **[View Consolidated Evaluation Matrix](docs/evaluation/evaluation_matrix.md)**
- **[View Technical Evaluation Report](docs/evaluation/evaluation_report.md)**

The evaluation framework reports standard quantitative metrics across 6 distinct suites:
- **Document Classification:** Overall Accuracy, Macro F1, Weighted F1, Per-class Precision/Recall, $5 \times 5$ Confusion Matrix.
- **Metadata Extraction:** Strict Exact Match (EM), Mean Character Similarity (Levenshtein), Success Rate @ 0.80, Token-level F1.
- **Hybrid Retrieval:** Recall@K ($K \in \{1, 3, 5, 10\}$), Mean Reciprocal Rank (MRR), nDCG@10.
- **Grounded Enterprise RAG:** Retrieval Hit Rate, Hallucination Rate, Mean Token F1, End-to-End Latency.
- **Agent Tool Routing:** Tool Selection Accuracy, Single-Tool vs Multi-Tool routing, Unnecessary Call Rate, Routing Latency.
- **Uploaded Document RAG:** Recall@5, Source Attribution Accuracy, Keyword Grounding Recall.

### Selected Empirical Results

The following table summarizes empirical measurements obtained from local benchmark execution:

| Component | Target Model / System | Evaluated Benchmark Split | Primary Metric | Measured Score | Scope Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Document Classification** | DistilBERT (`distilbert-base-uncased`) | `ml/processed/test.csv` (996 samples) | Overall Accuracy | **99.90%** (0.9990) | Held-out 5-class test split; Macro F1: **99.93%** (1 error / 996 docs) |
| **Document Classification** | TF-IDF + Logistic Regression | `ml/processed/test.csv` (996 samples) | Overall Accuracy | **99.70%** (0.9970) | 30k n-gram baseline; DistilBERT achieves **+0.20%** accuracy delta |
| **Metadata Extraction** | LayoutLM Multi-Label + Heuristics | 3,850 DocILE invoices (30,823 fields) | Success Rate @ 0.80 | **97.01%** (0.9701) | Normalized character similarity: **97.69%**; Token Macro F1: **87.46%** |
| **Hybrid Retrieval** | Hybrid RRF ($k=60$, BM25S + BGE) | 223,234 corpus chunks (15 queries) | Recall@10 | **73.33%** (0.7333) | Outperforms BM25S alone (66.67%) and dense BGE alone (66.67%) |
| **Grounded RAG** | Qwen2.5-1.5B-Instruct | 14 multi-category enterprise queries | Retrieval Hit Rate | **100.00%** (1.0000) | Supporting passage present in top-5; Hallucination Rate: **0.00%** |
| **Agent Tool Routing** | Rule-Based Agent Orchestrator | 20 natural enterprise queries | Tool Selection Acc | **70.00%** (0.7000) | Single-tool: **75.00%**, Multi-tool: **50.00%**, Mean Latency: **0.019 ms** |
| **Uploaded Document RAG** | ChromaDB + Qwen2.5-1.5B | 8 queries across PDF, DOCX, TXT, EML | Recall@5 / Source Acc | **100.00%** / **100.00%** | 100% correct file attribution across all 4 supported upload formats |

> **Evaluation Scope & Note on Generalization:** Current benchmark results are measured on curated held-out test splits, annotated benchmark datasets (DocILE, CUAD, Enron), and structured local evaluation suites. While the modular pipeline is designed to be extensible to additional domains, additional out-of-distribution and cross-domain evaluations would be required to quantify performance on arbitrary, unseen enterprise corpora.

---

## Extensibility

DocuMind is designed as an open, modular document intelligence architecture that can be extended across several dimensions without re-architecting the core application:

- **Additional Document Classes:** New document categories can be introduced by fine-tuning the classification head or registering few-shot prompt classifiers in `services/documind_service.py`.
- **Custom Metadata Schemas:** The extraction layer separates LayoutLM token classification from field-level parsing heuristics, allowing developers to define new extraction schemas (e.g., medical records, shipping manifests, tax forms).
- **Alternative Retrieval Backends:** The hybrid search engine decouples candidate generation from rank fusion; BM25S can be replaced with Elasticsearch/OpenSearch, and BGE can be swapped for larger embedding models or specialized legal/financial encoders.
- **Pluggable Vector Databases:** ChromaDB is encapsulated behind `chroma_service.py`, enabling straightforward migration to Milvus, Qdrant, Pinecone, or pgvector.
- **Scalable Language Models:** The RAG pipeline communicates with local or hosted autoregressive models via standardized prompt assembly, allowing drop-in upgrades to larger models (e.g., Qwen2.5-7B, Llama-3.3-8B) or hosted endpoints.
- **New Document Formats:** Format-specific parsers in `document_ingestion.py` output a uniform chunk data contract (`text`, `page_number`, `chunk_index`), making it easy to add support for HTML, Markdown, RTF, or image-based OCR formats.

---

## User Workflow & Usage

DocuMind separates document lifecycle management from conversational reasoning into two dedicated interfaces:

1. **Start Backend & Database:** Start PostgreSQL via Docker and run the FastAPI server on port 8000.
2. **Start Frontend:** Launch Vite on port 5173 and navigate to `http://localhost:5173`.
3. **Document Management (`Documents` View):**
   - Click **Documents** in the sidebar to upload files (`.pdf`, `.docx`, `.txt`, `.eml`).
   - Monitor real-time extraction status, view page/chunk counts, and inspect parsed chunks.
4. **Conversational Intelligence (`Agent` View):**
   - Click **Agent** in the sidebar (the primary query interface).
   - Enter questions in natural language. The orchestrator automatically identifies intent and executes the required pipeline:
     - *Classification:* `"What type of document is invoice_0292?"`
     - *Metadata Extraction:* `"Extract the total gross amount and date for invoice_0292."`
     - *Corpus Search & QA:* `"Explain the early termination fee provisions in contract_0086."`
     - *Uploaded-Document QA:* `"What qualifications are required in my uploaded internship JD?"`
     - *Cross-System Comparison:* `"Compare the notice period in contract_0112 with my uploaded agreement."`
5. **Inspect Traces & Grounding:**
   - Examine intermediate tool calls, class confidence scores, and verbatim retrieved source chunks supporting each answer.

---

## Repository Structure

```
DocuMind/
├── backend/                  # FastAPI Application, DB Models, Services, REST Routes
│   ├── alembic/              # Database migration scripts
│   ├── db/                   # SQLAlchemy models and session management
│   ├── routes/               # /agent and /documents REST API endpoints
│   ├── schemas/              # Pydantic v2 request/response schemas
│   ├── services/             # Orchestrator, ChromaDB, ingestion, and ML services
│   └── main.py               # Application entry point and CORS configuration
├── frontend/                 # React 19 + Vite 8 Desktop SaaS UI
│   ├── src/                  # Components, views, API clients, and stylesheets
│   ├── package.json          # Node dependencies and build scripts
│   └── vite.config.js        # Vite bundler configuration
├── ml/                       # Machine Learning Engineering Assets
│   ├── notebooks/            # Exploratory analysis, training, and validation notebooks
│   └── src/                  # Core ML agent, hybrid search, and inference tools
├── agents/                   # Agent orchestration logic and multi-tool routing
├── evaluation/               # Comprehensive Automated Evaluation Framework
│   ├── agent/                # Agent intent and tool-routing benchmarks
│   ├── classification/       # DistilBERT vs TF-IDF classification benchmarks
│   ├── common/               # Metric computation utilities and result serializers
│   ├── metadata/             # DocILE invoice metadata extraction benchmarks
│   ├── rag/                  # Grounded Qwen enterprise QA benchmarks
│   ├── results/              # Consolidated evaluation JSON results
│   ├── retrieval/            # BM25S, BGE, and Hybrid RRF benchmarks
│   ├── uploaded_rag/         # Multi-format ChromaDB RAG benchmarks
│   └── run_all_evaluations.py# Master test suite runner
├── docs/                     # Technical Documentation
│   └── evaluation/           # Evaluation Matrix, Report, and Benchmark Design
├── data/                     # Dataset references and schema definitions (git-ignored data)
├── docker-compose.yml        # PostgreSQL container configuration
├── .gitignore                # Exclusion rules for secrets, virtualenvs, models, and caches
└── README.md                 # Project documentation and architecture guide
```

---

## Local Setup & Quickstart

### Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** 3.11 or 3.12 (with virtual environment support)
- **Node.js:** Node 18+ and npm 9+
- **Docker:** Docker Desktop or Docker Engine (for PostgreSQL)
- **Hardware (Optional):** NVIDIA GPU with CUDA 12+ for accelerated local LLM inference; CPU fallback supported.

### 1. Database Setup
Start PostgreSQL using Docker Compose:
```bash
docker-compose up -d
```

### 2. Backend Setup
Activate your Python virtual environment and install dependencies:
```bash
# Windows
ml\.venv\Scripts\Activate.ps1

# Linux / macOS
source ml/.venv/bin/activate

cd backend
pip install -r requirements.txt
```

Apply database migrations:
```bash
alembic upgrade head
```

Launch the FastAPI backend server:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
API interactive documentation will be available at `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

### 4. Running the Automated Evaluation Suite
To execute the entire 6-component evaluation suite and regenerate all benchmark summaries:
```bash
python evaluation/run_all_evaluations.py
```

---

## Technical Limitations

- **Image-Only Scanned PDFs:** The current document parser relies on digital text stream extraction (`pypdf`). Image-only PDF scans without an embedded OCR layer require external optical character recognition preprocessing.
- **Embedded ChromaDB Scope:** The default vector store operates in embedded persistent mode (`chromadb.PersistentClient`). Enterprise multi-node high availability would benefit from an external Chroma server or distributed cluster.
- **Local Generation Throughput:** Local autoregressive inference with `Qwen2.5-1.5B-Instruct` executes sequentially on single-GPU setups; enterprise production deployments can point to batched vLLM or Triton inference servers.

---

## License

This project is licensed under the MIT License.
