# DocuMind | Enterprise Document Intelligence Platform

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2019-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Bundler-Vite%208-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%2016-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-FF6F00?style=flat)](https://www.trychroma.com)
[![Google GenAI](https://img.shields.io/badge/Agent-Google%20GenAI%20Gemini-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![PyTorch](https://img.shields.io/badge/ML%20Core-PyTorch%20%7C%20CUDA-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Models-DistilBERT%20%7C%20LayoutLM%20%7C%20BGE-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)

DocuMind is an enterprise-grade document intelligence platform combining high-accuracy fine-tuned ML models with frontier cloud agent orchestration:

$$\text{Document Ingestion} \longrightarrow \text{DistilBERT Classification} \longrightarrow \text{LayoutLM / Generic Extraction} \longrightarrow \text{Hybrid BM25+BGE Retrieval} \longrightarrow \text{Hosted Gemini Reasoning}$$

DocuMind provides a unified, service-oriented architecture capable of processing institutional document corpora (223,234 passages) alongside dynamic, multi-tenant user uploads (PDF, DOCX, TXT, EML) through model-driven autonomous orchestration.

---

## Key Highlights

- **Model-Driven Autonomous Agent:** Powered by the official Google GenAI Python SDK (`google-genai`) and Gemini models (`gemini-3.5-flash`), with autonomous tool calling, multi-model failover, in-context mathematical reasoning, and local ChromaDB evidence fallback.
- **Unified Studio Canvas UI:** A single-page workspace inspired by modern editorial design (Godly.design) featuring a responsive layout integrating the Agent Workspace and the Document Management Repository.
- **Strict Multi-Tenant Retrieval Isolation:** Scoped ChromaDB vector search enforcing tenant and document boundaries (`user_id`, `document_id`) with zero fallback to global queries on empty results.
- **Hybrid Retrieval with RRF:** Lexical search (`BM25S`) and dense semantic search (`BAAI/bge-small-en-v1.5`) fused via Reciprocal Rank Fusion ($k=60$) with structural business identifier boosting (`INV-...`, `SOW-...`, `MSKU-...`, `SWIFT`).
- **Fine-Tuned Document Classification:** Fine-tuned `distilbert-base-uncased` achieving 99.9% accuracy across 5 core enterprise document classes (`Contract`, `Email`, `Invoice`, `Purchase Order`, `Report`).
- **Runtime LayoutLM Invoice Extraction:** Specialized token-classification inference via fine-tuned `MultiLabelLayoutLM` utilizing true token bounding boxes from PDF layouts (`PyMuPDF`) for 7 target invoice fields, paired with a generic schema-driven extraction pipeline for non-invoice fields.

---

## Technology Stack

- **Frontend:** React 19, Vite 8, Vanilla CSS Design System with dark enterprise tokens and JetBrains Mono typography.
- **Backend:** FastAPI, Python 3.11+, SQLAlchemy 2.0 (`Mapped` / `mapped_column`), Alembic, PostgreSQL 16, ChromaDB (HNSW cosine space).
- **Agent & ML Stack:**
  - **Single Frontier Generative LLM:** Google GenAI Python SDK (`google-genai`), Gemini 3.5 Flash (`gemini-3.5-flash`), performing autonomous tool selection, synthesis, calculations, and grounded document QA.
  - **Document Classification:** Fine-tuned `distilbert-base-uncased` (5 classes, exported checkpoint in `ml/models/distilbert_doc_classifier/final`).
  - **Invoice Layout Extraction:** Fine-tuned `MultiLabelLayoutLM` (`microsoft/layoutlm-base-uncased` with 7-field token classifier in `ml/models/layoutlm_multilabel/final`), evaluated with true spatial bounding boxes.
  - **Dense Semantic Retrieval:** `BAAI/bge-small-en-v1.5` (384-dimensional embeddings, normalized cosine distance).
  - **Sparse Lexical Retrieval:** Rust-accelerated `BM25S` inverted index over 223K corpus passages.
- **Database & Storage:** PostgreSQL 16 (system of record for users and document metadata), ChromaDB (isolated chunk embeddings), local disk storage for binary files and extracted caches.

---

## System Architecture

```mermaid
flowchart TD
    User([Enterprise User]) --> UI[React 19 Studio Canvas<br/>60/40 Layout: Agent & Documents]
    UI --> API[FastAPI Application Server<br/>Port 8000]
    
    subgraph AgentEngine["Model-Driven Agent Layer (services/agent.py)"]
        Agent[Hosted Gemini Agent Orchestrator<br/>google-genai SDK: gemini-3.5-flash]
        Tool1[classify_document<br/>DistilBERT 5-class]
        Tool2[retrieve_documents<br/>Hybrid BM25S + BGE-small + RRF]
        Tool3[extract_document<br/>Specialized LayoutLM + Generic Schema Parser]
        Fallback[Local ChromaDB Grounded Fallback]
        
        Agent --> Tool1
        Agent --> Tool2
        Agent --> Tool3
        Agent -. API Unavailable .-> Fallback
    end
    
    subgraph PersistenceLayer["Data & Vector Persistence"]
        PG[(PostgreSQL 16<br/>Users, Document Metadata, Lifecycle)]
        ChromaStore[(ChromaDB Vector Store<br/>Multi-Tenant Upload Embeddings)]
        CorpusIndex[(Corpus Index<br/>223,234 Passages in BM25S + BGE)]
    end
    
    API --> AgentEngine
    API --> PG
    Tool2 --> CorpusIndex
    Tool2 --> ChromaStore
    Fallback --> ChromaStore
```

### Core Autonomous Tools

1. **`classify_document(document_id, text)`**: Invokes the fine-tuned DistilBERT sequence classifier to categorize documents into Contract, Email, Invoice, Purchase Order, or Report.
2. **`retrieve_documents(query, scope, top_k)`**: Executes hybrid sparse-dense retrieval across both the historical 223K passage corpus and user-uploaded ChromaDB embeddings, enforcing strict tenant scoping.
3. **`extract_document(evidence, schema)`**: Dispatches to fine-tuned `MultiLabelLayoutLM` for supported invoice fields using true spatial bounding boxes from PDF layouts (`PyMuPDF`), with schema-driven extraction for non-invoice and table fields.

---

## Supported Document Types & Formats

### Classification Categories (DistilBERT)
- **Contract:** Master services agreements, statement of work (SOW) documents, non-disclosure agreements, and licensing terms.
- **Email:** Corporate correspondence, procurement negotiation threads, dispatch notifications, and approval chains.
- **Invoice:** Billing statements, vendor tax invoices, itemized delivery bills, and remittance notices.
- **Purchase Order:** Procurement requisitions, line-item purchase orders, and supplier confirmations.
- **Report:** Corporate annual reviews, 10-K regulatory filings, executive summaries, and multi-KPI balance sheets.

### Ingestion File Formats
- **PDF (`.pdf`):** Multi-page extraction via PyPDF with per-page tracking and sliding-window chunking.
- **Word (`.docx`):** Paragraph and embedded table extraction via `python-docx`.
- **Plain Text (`.txt`):** Structured text reading with whitespace normalization.
- **Email (`.eml`):** Native RFC 822 parsing extracting Subject, From, To, Date, and body components.

---

## Repository Structure

```text
DocuMind/
├── backend/                            # FastAPI Application Server & Services
│   ├── alembic/                        # SQLAlchemy database migrations
│   ├── db/                             # Models (User, Document) & DB engine
│   ├── routes/                         # REST controllers (/agent, /documents)
│   ├── schemas/                        # Pydantic validation and serialization models
│   ├── services/                       # Core business logic services:
│   │   ├── agent.py                    # Model-driven Gemini agent with tool calling
│   │   ├── chroma_service.py           # Persistent ChromaDB vector client
│   │   ├── document_ingestion.py       # Multi-format parser (PDF, DOCX, TXT, EML)
│   │   ├── documind_service.py         # Adapter linking backend to local ML models
│   │   ├── extraction.py               # LayoutLM & Schema-Driven Extraction Engine
│   │   ├── retrieval.py                # Hybrid BM25S + BGE search with scope isolation
│   │   └── uploaded_rag.py             # Upload lifecycle & isolated document QA
│   ├── storage/                        # Persistent local document uploads & ChromaDB
│   ├── alembic.ini                     # Migration configuration
│   ├── main.py                         # Application factory, lifespan, CORS, and routing
│   └── requirements.txt                # Python backend dependencies
├── frontend/                           # React 19 + Vite 8 Single-Page Application
│   ├── src/
│   │   ├── api/                        # API client (documind.js)
│   │   ├── components/                 # UI components (Header, QuerySection, etc.)
│   │   ├── App.jsx                     # Root Studio Canvas layout
│   │   └── index.css                   # Minimal editorial styling tokens
│   ├── package.json                    # Node dependencies and build scripts
│   └── vite.config.js                  # Vite configuration
├── ml/                                 # Machine Learning Engineering Assets
│   ├── models/                         # Model weights and checkpoint directories
│   │   ├── distilbert_doc_classifier/  # Fine-tuned DistilBERT 5-class sequence classifier
│   │   └── layoutlm_multilabel/        # Fine-tuned LayoutLM 7-field invoice extractor
│   ├── notebooks/                      # Exploratory data analysis & training notebooks
│   ├── processed/                      # Pre-built search indices & dataset parquets
│   └── src/                            # Local ML inference scripts (documind_agent.py)
├── evaluation/                         # Automated Evaluation Framework
│   ├── agent/                          # Agent routing benchmarks
│   ├── classification/                 # DistilBERT evaluation scripts
│   ├── metadata/                       # Invoice extraction benchmarks
│   ├── rag/                            # Corpus QA groundedness evaluation
│   ├── retrieval/                      # Hybrid retrieval benchmarks
│   ├── stress_tests/                   # Edge cases and negative restraint benchmarks
│   ├── uploaded_rag/                   # User document QA evaluation
│   ├── gemini_eval.py                  # Evaluation runner for hosted Gemini agent
│   └── run_all_evaluations.py          # Master evaluation orchestrator
├── docs/                               # System documentation and evaluation specifications
├── docker-compose.yml                  # PostgreSQL 16 container definition
├── pyrightconfig.json                  # Static analysis path configuration
├── run_backend.py                      # Dedicated uvicorn launcher
└── README.md                           # Project guide and architectural documentation
```

---

## Local Setup & Quickstart

### Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** 3.11 or 3.12 (with virtual environment support)
- **Node.js:** Node 18+ and npm 9+
- **Docker:** Docker Desktop or Docker Engine (for PostgreSQL)
- **Gemini API Key:** Obtain an API key from [Google AI Studio](https://aistudio.google.com/)

### 1. Start PostgreSQL
```bash
docker compose up -d
```
This starts PostgreSQL 16 on port `5432` with database `documind`.

### 2. Configure Backend Environment
Create `backend/.env` (or copy from `backend/.env.example`):
```ini
DATABASE_URL=postgresql+psycopg://postgres:password@127.0.0.1:5432/documind
DEV_USER_ID=dev_user_001
DEV_USER_EMAIL=developer@documind.local
CHROMA_PATH=./storage/chroma

# Google GenAI Gemini Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

### 3. Install Dependencies & Run Database Migrations
Activate your Python environment:
```bash
# Apply database migrations
cd backend
python -m alembic upgrade head
cd ..
```

### 4. Start Backend Server
```bash
python run_backend.py
```
The FastAPI backend starts at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 5. Start Frontend UI
```bash
cd frontend
npm install
npm run dev
```
The React frontend starts at `http://localhost:5173`.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/agent` | Execute autonomous multi-tool Gemini agent across documents |
| `POST` | `/classify` | Classify document into 5 categories using DistilBERT |
| `POST` | `/extract` | Extract structured schema fields from document text |
| `POST` | `/search` | Hybrid BM25S + BGE retrieval across corpus chunks |
| `POST` | `/qa` | Grounded question answering on corpus documents |
| `POST` | `/documents/upload` | Upload PDF, DOCX, TXT, or EML document for chunking & indexing |
| `GET` | `/documents` | List uploaded documents and extraction statuses |
| `GET` | `/documents/{id}` | Get document metadata and extracted chunks |
| `DELETE` | `/documents/{id}` | Cascade delete uploaded document from DB and ChromaDB |
| `POST` | `/documents/{id}/qa` | Ask a question strictly isolated to an uploaded document |
| `GET` | `/health` | Health check probe |

---

## License & Attribution
DocuMind is an open-source project developed for enterprise document intelligence research and technical benchmarking. Built with FastAPI, React, PyTorch, Hugging Face Transformers, ChromaDB, and Google GenAI.
