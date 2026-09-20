# DocuMind AI/ML Evaluation Matrix

This matrix provides a comprehensive, quantitative summary of the evaluations conducted across all six core machine learning and artificial intelligence components in DocuMind. 

All metrics listed as **Measured Result** were obtained by executing the automated benchmark suite against local checkpoints, processed datasets, and vector indexes. Historical validation benchmarks from model training are included as **Target / Reference** for direct comparison.

---

## Consolidated Evaluation Matrix

| Component | Model / System | Dataset / Split | Metric | Measured Result | Target / Reference | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Document Classification** | DistilBERT (`distilbert-base-uncased`) | `ml/processed/test.csv` (996 samples) | Overall Accuracy | **99.90%** (0.9990) | ~99.90% | Evaluated on 5 balanced enterprise document classes (Production Agent pipeline) |
| | DistilBERT | `ml/processed/test.csv` (996 samples) | Macro F1 | **99.93%** (0.9993) | ~99.93% | Unweighted mean across all 5 classes |
| | DistilBERT | `ml/processed/test.csv` (996 samples) | Weighted F1 | **99.90%** (0.9990) | ~99.90% | Support-weighted mean across all 5 classes |
| | DistilBERT | `ml/processed/test.csv` (996 samples) | Macro Precision | **99.93%** (0.9993) | — | Unweighted mean precision |
| | DistilBERT | `ml/processed/test.csv` (996 samples) | Macro Recall | **99.93%** (0.9993) | — | Unweighted mean recall |
| | DistilBERT | `ml/processed/test.csv` (77 samples) | Contract F1 | **100.00%** (1.0000) | 100.00% | Precision: 1.0000, Recall: 1.0000 |
| | DistilBERT | `ml/processed/test.csv` (300 samples) | Email F1 | **99.83%** (0.9983) | 99.00%+ | Precision: 1.0000, Recall: 0.9967 (1 misclassified as Report) |
| | DistilBERT | `ml/processed/test.csv` (300 samples) | Invoice F1 | **100.00%** (1.0000) | 100.00% | Precision: 1.0000, Recall: 1.0000 |
| | DistilBERT | `ml/processed/test.csv` (19 samples) | Purchase Order F1 | **100.00%** (1.0000) | 100.00% | Precision: 1.0000, Recall: 1.0000 |
| | DistilBERT | `ml/processed/test.csv` (300 samples) | Report F1 | **99.83%** (0.9983) | 99.00%+ | Precision: 0.9967, Recall: 1.0000 |
| | TF-IDF + Logistic Regression | `ml/processed/test.csv` (996 samples) | Overall Accuracy | **99.70%** (0.9970) | Baseline | 30k n-gram TF-IDF baseline model |
| | TF-IDF + Logistic Regression | `ml/processed/test.csv` (996 samples) | Macro F1 | **99.20%** (0.9920) | Baseline | DistilBERT outperforms baseline Macro F1 (+0.73%) |
| | DistilBERT vs Baseline | `ml/processed/test.csv` (996 samples) | Accuracy Delta | **+0.0020** (+0.20%) | Baseline | Fine-tuned DistilBERT outperforms TF-IDF baseline |
| | DistilBERT (Naïve Truncation Ablation) | `ml/processed/test.csv` (996 samples) | Overall Accuracy | **98.80%** (0.9880) | — | Uncleaned raw text truncated at 512 tokens without sliding window |
| | DistilBERT (Naïve Truncation Ablation) | `ml/processed/test.csv` (996 samples) | Macro F1 | **99.20%** (0.9920) | — | 11 Reports misclassified due to raw OCR header artifacts |
| **2. Metadata Extraction** | LayoutLM Multi-Label | DocILE Annotations (`invoice_field_quality.csv`) | Token-level Macro F1 | **87.46%** (0.8746) | 87.46% | Measured during model checkpoint validation |
| | LayoutLM Multi-Label | 3,850 docs / 30,823 field instances | Field Exact Match (EM) | **76.91%** (0.7691) | — | Document/field-level exact string match |
| | LayoutLM Multi-Label | 3,850 docs / 30,823 field instances | Mean Character Sim | **97.69%** (0.9769) | — | Normalized Levenshtein ratio across all fields |
| | LayoutLM Multi-Label | 3,850 docs / 30,823 field instances | Success Rate @ 0.80 | **97.01%** (0.9701) | — | Extractions with $\ge 0.80$ similarity to ground truth |
| | LayoutLM Multi-Label | 3,850 docs / 30,823 field instances | Macro Token F1 | **91.40%** (0.9140) | — | Mean token-level unigram F1 across all fields |
| | LayoutLM Multi-Label | 5,140 instances | `vendor_name` EM / Sim | **80.43%** / **98.62%** | — | Token F1: 91.75%, Success @ 0.80: 98.62% |
| | LayoutLM Multi-Label | 5,285 instances | `vendor_address` EM / Sim | **49.76%** / **96.92%** | — | Multi-line OCR wrapping causes lower EM |
| | LayoutLM Multi-Label | 4,081 instances | `customer_billing_name` EM / Sim | **83.17%** / **98.94%** | — | Token F1: 93.53%, Success @ 0.80: 98.90% |
| | LayoutLM Multi-Label | 3,706 instances | `customer_billing_address` EM / Sim | **58.90%** / **98.30%** | — | High similarity confirms near-perfect semantic match |
| | LayoutLM Multi-Label | 4,431 instances | `date_issue` EM / Sim | **90.68%** / **98.49%** | — | Token F1: 92.91%, Success @ 0.80: 97.97% |
| | LayoutLM Multi-Label | 3,910 instances | `amount_total_gross` EM / Sim | **89.18%** / **95.86%** | — | Token F1: 89.26%, Success @ 0.80: 94.48% |
| | LayoutLM Multi-Label | 4,270 instances | `amount_due` EM / Sim | **90.37%** / **96.70%** | — | Token F1: 90.52%, Success @ 0.80: 95.57% |
| **3. Hybrid Retrieval** | BM25S (Lexical) | 223,234 enterprise chunks (15 queries) | Recall@1 | **26.67%** (0.2667) | — | Exact term matching |
| | BM25S | 223,234 enterprise chunks (15 queries) | Recall@5 | **66.67%** (0.6667) | — | Top-5 lexical candidate recall |
| | BM25S | 223,234 enterprise chunks (15 queries) | Recall@10 | **66.67%** (0.6667) | — | Top-10 lexical candidate recall |
| | BM25S | 223,234 enterprise chunks (15 queries) | MRR | **0.4356** | — | Mean Reciprocal Rank |
| | BM25S | 223,234 enterprise chunks (15 queries) | nDCG@10 | **0.4783** | — | Normalized Discounted Cumulative Gain |
| | BGE-small-en-v1.5 (Dense) | 223,234 enterprise chunks (15 queries) | Recall@1 | **33.33%** (0.3333) | — | Dense semantic search outperforms BM25 on top-1 |
| | BGE-small-en-v1.5 | 223,234 enterprise chunks (15 queries) | Recall@5 | **66.67%** (0.6667) | — | Top-5 dense candidate recall |
| | BGE-small-en-v1.5 | 223,234 enterprise chunks (15 queries) | Recall@10 | **66.67%** (0.6667) | — | Top-10 dense candidate recall |
| | BGE-small-en-v1.5 | 223,234 enterprise chunks (15 queries) | MRR | **0.4835** | — | Dense achieves higher MRR than BM25S alone |
| | BGE-small-en-v1.5 | 223,234 enterprise chunks (15 queries) | nDCG@10 | **0.5030** | — | Highest ranking quality among single retrievers |
| | Hybrid RRF (BM25S + BGE) | 223,234 enterprise chunks (15 queries) | Recall@1 | **20.00%** (0.2000) | — | Reciprocal Rank Fusion ($k=60$) |
| | Hybrid RRF | 223,234 enterprise chunks (15 queries) | Recall@5 | **60.00%** (0.6000) | — | Top-5 candidate recall |
| | Hybrid RRF | 223,234 enterprise chunks (15 queries) | Recall@10 | **73.33%** (0.7333) | > 66.67% | **Highest overall recall at K=10** (+6.66% over single retrievers) |
| | Hybrid RRF | 223,234 enterprise chunks (15 queries) | MRR | **0.3956** | — | Balanced reciprocal rank |
| | Hybrid RRF | 223,234 enterprise chunks (15 queries) | nDCG@10 | **0.4742** | — | Graded relevance score |
| **4. Grounded RAG** | Qwen2.5-1.5B-Instruct | 14 enterprise QA items (6 categories) | Retrieval Hit Rate | **100.00%** (1.0000) | 100.00% | Supporting context present in retrieved top-5 chunks |
| | Qwen2.5-1.5B-Instruct | 14 enterprise QA items (6 categories) | Hallucination Rate | **0.00%** (0.0000) | < 5.0% | Zero unsupported claims generated contrary to context |
| | Qwen2.5-1.5B-Instruct | 14 enterprise QA items (6 categories) | Mean Token F1 | **28.55%** (0.2855) | ~30.0% | Generative verbosity difference vs concise references |
| | Qwen2.5-1.5B-Instruct | 14 enterprise QA items (6 categories) | Exact Match Rate | **0.00%** (0.0000) | — | Autoregressive model provides full conversational explanations |
| | Qwen2.5-1.5B-Instruct | 14 enterprise QA items (6 categories) | Mean Total Latency | **6,702 ms** | < 10,000 ms | Local consumer GPU/CPU inference |
| | Qwen2.5-1.5B-Instruct | 14 enterprise QA items (6 categories) | Retrieval Latency | **1,005 ms** | < 2,000 ms | Hybrid search across 223k chunks |
| | Qwen2.5-1.5B-Instruct | 14 enterprise QA items (6 categories) | Generation Latency | **5,697 ms** | < 8,000 ms | Autoregressive decoding (mean ~150 tokens) |
| **5. Agent Tool Routing** | Rule-Based Agent Planner | 20 diverse user queries (6 categories) | Tool Selection Accuracy | **70.00%** (0.7000) | 100.0% (on 6-query training set) | Generalization across complex enterprise intents |
| | Rule-Based Agent Planner | 16 single-tool queries | Single-Tool Accuracy | **75.00%** (0.7500) | — | Evaluated on atomic search, classification, extraction |
| | Rule-Based Agent Planner | 4 multi-tool queries | Multi-Tool Accuracy | **50.00%** (0.5000) | — | Compound workflows (classify + extract, search + compare) |
| | Rule-Based Agent Planner | 20 diverse user queries | Unnecessary Call Rate | **20.00%** (0.2000) | < 25.0% | Rate of planning surplus or misdirected tool calls |
| | Rule-Based Agent Planner | 20 diverse user queries | Mean Planning Latency | **0.014 ms** | < 1.0 ms | Sub-millisecond deterministic routing |
| **6. Uploaded Document RAG** | ChromaDB + Qwen2.5-1.5B | 8 multi-format queries (PDF, DOCX, TXT, EML) | Retrieval Recall@5 | **100.00%** (1.0000) | 100.00% | 100% of target documents retrieved in top-5 |
| | ChromaDB + Qwen2.5-1.5B | 8 multi-format queries | Source Attribution Acc | **100.00%** (1.0000) | 100.00% | 100% correct file attribution in responses |
| | ChromaDB + Qwen2.5-1.5B | 8 multi-format queries | Keyword Grounding | **87.50%** (0.8750) | > 80.0% | Core domain facts captured in synthesized answer |
| | ChromaDB + Qwen2.5-1.5B | 8 multi-format queries | Mean Token F1 | **30.01%** (0.3001) | ~30.0% | Balanced token overlap across 4 file formats |
| | ChromaDB + Qwen2.5-1.5B | 8 multi-format queries | Mean Total Latency | **5,868 ms** | < 8,000 ms | ChromaDB query + Qwen2.5 autoregressive synthesis |

---

## Metric Key and Methodology Notes

1. **Document Classification:**
   - Evaluated on `ml/processed/test.csv` (996 held-out enterprise documents across 5 classes).
   - Checkpoint: Fine-tuned `distilbert-base-uncased` (`ml/models/distilbert_doc_classifier/final`).
   - In production agent inference (`ml/src/documind_agent.py`) and historical training benchmarks (`ml/notebooks/02_document_classification.ipynb`), documents are preprocessed with `clean_model_text` (stripping HTML tags, markdown image syntax, and OCR page-split markers) and tokenized using sliding-window chunking (`max_length=512`, `stride=128`) with mean logit pooling. Under this native pipeline, DistilBERT achieves **99.90% Accuracy** (995/996) and **99.93% Macro F1**, outperforming the 30k n-gram TF-IDF baseline (99.70% Accuracy, 99.20% Macro F1) with only a single document misclassified (`email_0946` as Report).
   - An independent ablation evaluating raw uncleaned text with single-chunk 512-token truncation reproduces **98.80% Accuracy** and **99.20% Macro F1**; here, uncleaned image and page-split headers in 11 multi-page Report documents pushed body content beyond the 512-token window, misclassifying them as Email. Both evaluation modes are generated and verified from the model.
2. **Metadata Extraction:**
   - Evaluated on 30,823 ground-truth field instances across DocILE-annotated invoices.
   - Exact match is string-level identical. Character similarity ($\text{Sim} \ge 0.80$ at **97.01%**) demonstrates that non-exact matches are almost exclusively due to OCR newline wrapping, trailing commas, or phone number hyphenation rather than entity hallucination.
3. **Hybrid Retrieval:**
   - Hybrid RRF achieved **73.33% Recall@10**, outperforming both pure BM25S (66.67%) and pure BGE dense retrieval (66.67%) across a large 223,234-chunk corpus.
4. **Grounded RAG:**
   - 100% of queries successfully retrieved supporting context in top-5 chunks.
   - Zero hallucinations were detected against ground truth.
   - Explicit citation formatting was omitted by default Qwen system prompting, highlighting a prompt engineering optimization path.
5. **Agent Tool Routing:**
   - While the planner scored 100% on its original 6 synthetic training cases, evaluating it against 20 natural queries revealed an overall accuracy of **70.00%**, highlighting edge cases in multi-hop chaining that can be resolved with an LLM-assisted or few-shot classifier fallback.
6. **Uploaded Document RAG:**
   - Demonstrates 100% Recall@5 and 100% Source Attribution Accuracy across all 4 production file formats (PDF, DOCX, TXT, EML) in ChromaDB.
