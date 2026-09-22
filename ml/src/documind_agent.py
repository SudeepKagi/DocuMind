import re
import numpy as np
import pandas as pd
import torch
import bm25s

from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# LOAD DATA
# ============================================================

full_chunks_df = pd.read_parquet(
    PROCESSED_DIR / "full_text_chunks.parquet"
)

full_corpus_df = pd.read_parquet(
    PROCESSED_DIR / "full_corpus.parquet"
)

full_embeddings = np.load(
    PROCESSED_DIR / "full_text_embeddings.npy",
    mmap_mode="r"
)

full_bm25s = bm25s.BM25.load(
    str(PROCESSED_DIR / "full_text_bm25s"),
    load_corpus=False
)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("DocuMind agent loading...")
print("Device:", device)
print("Chunks:", len(full_chunks_df))

# ============================================================
# LOAD MODELS
# ============================================================

# ------------------------------------------------------------
# BGE semantic embedding model
# ------------------------------------------------------------

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device=device
)


# ------------------------------------------------------------
# DistilBERT document classifier
# ------------------------------------------------------------

CLASSIFIER_PATH = MODELS_DIR / "distilbert_doc_classifier" / "final"

classifier_tokenizer = AutoTokenizer.from_pretrained(
    str(CLASSIFIER_PATH)
)

classifier_model = AutoModelForSequenceClassification.from_pretrained(
    str(CLASSIFIER_PATH),
    torch_dtype=torch.float16
).to(device)

classifier_model.eval()


print("Models loaded successfully")
print("BGE device:", embedding_model.device)
print("Classifier device:", next(classifier_model.parameters()).device)
print("Classes:", classifier_model.config.id2label)

# ============================================================
# DOCUMENT ID EXTRACTION
# ============================================================

def extract_document_id(question):
    match = re.search(
        r'\b(invoice|contract|email|report|purchase[_ ]?order)[_-]?(\d{3,5})\b',
        question,
        re.IGNORECASE
    )

    if not match:
        return None

    doc_type = match.group(1).lower()
    number = match.group(2)

    prefix_map = {
        "invoice": "invoice",
        "contract": "contract",
        "email": "email",
        "report": "report",
        "purchase order": "purchase_order",
        "purchase_order": "purchase_order"
    }

    return f"{prefix_map[doc_type]}_{number}"


# ============================================================
# CLASSIFICATION TOOL
# ============================================================

def classify_document(question="", document_id=None, text=None):

    target_text = text
    doc_id = document_id or extract_document_id(question)

    if not target_text and doc_id:
        matches = full_corpus_df[
            full_corpus_df["document_id"].astype(str).str.lower()
            == str(doc_id).lower()
        ]
        if len(matches) > 0:
            target_text = str(matches.iloc[0]["text"])

    if not target_text and question:
        target_text = question

    if not target_text:
        return {
            "status": "error",
            "message": "No text or document found to classify."
        }

    encoded = classifier_tokenizer(
        target_text,
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt"
    )


    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():

        outputs = classifier_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        mean_logits = outputs.logits.mean(dim=0)

        probabilities = torch.softmax(
            mean_logits,
            dim=-1
        )

        predicted_id = int(torch.argmax(
            probabilities
        ).item())

        confidence = float(probabilities[
            predicted_id
        ].item())

    predicted_class = classifier_model.config.id2label[
        predicted_id
    ]

    probability_dict = {}

    for i in range(len(probabilities)):
        label = classifier_model.config.id2label[i]

        probability_dict[label] = round(
            probabilities[i].item(),
            4
        )

    return {
        "status": "success",
        "document_id": document_id,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "num_chunks": int(input_ids.shape[0]),
        "probabilities": probability_dict,
        "method": "distilbert_document_classifier"
    }


# ============================================================
# METADATA EXTRACTION
# ============================================================

def extract_metadata(question):

    document_id = extract_document_id(question)

    if document_id is None:
        return {
            "status": "error",
            "message": "Could not identify a document ID."
        }

    if not document_id.startswith("invoice_"):
        return {
            "status": "error",
            "message": "Metadata extraction currently supports invoices."
        }

    matches = full_corpus_df[
        full_corpus_df["document_id"].astype(str).str.lower()
        == document_id.lower()
    ]

    if len(matches) == 0:
        return {
            "status": "error",
            "message": f"Document {document_id} not found."
        }

    text = str(matches.iloc[0]["text"])

    q = question.lower()

    if any(
        phrase in q
        for phrase in [
            "total amount",
            "gross amount",
            "total",
            "amount_total_gross"
        ]
    ):
        field = "amount_total_gross"

    elif any(
        phrase in q
        for phrase in [
            "amount due",
            "due amount",
            "balance due"
        ]
    ):
        field = "amount_due"

    else:
        return {
            "status": "error",
            "message": "Could not determine the requested metadata field."
        }

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    start = None

    for i in range(len(lines)):
        if lines[i].lower() == "invoice totals":
            start = i
            break

    if start is not None:

        section = lines[start:start + 20]

        currency_pattern = re.compile(
            r'^\(?\$[\d,]+(?:\.\d{2})?\)?$'
        )

        values = []

        for line in section:
            if currency_pattern.match(line):
                values.append(line)

        if field == "amount_total_gross" and len(values) >= 1:
            return {
                "status": "success",
                "document_id": document_id,
                "field": field,
                "value": values[0],
                "method": "invoice_totals_table"
            }

        if field == "amount_due" and len(values) >= 2:
            return {
                "status": "success",
                "document_id": document_id,
                "field": field,
                "value": values[-1],
                "method": "invoice_totals_table"
            }

    return {
        "status": "error",
        "message": f"Could not extract {field}."
    }


# ============================================================
# HYBRID SEARCH
# ============================================================

def search_documents(
    question,
    top_k=5,
    candidate_k=30,
    scope=None,
):

    q = question.lower().strip()

    # Generic hybrid retrieval across full BM25S and BGE embeddings
    # Optional document_type filter without keyword regex matching
    target_type = None
    if isinstance(scope, dict) and "document_type" in scope:
        target_type = scope["document_type"]
    elif isinstance(top_k, dict) and "document_type" in top_k:
        target_type = top_k["document_type"]
    elif isinstance(scope, str):
        target_type = scope

    limit = top_k if isinstance(top_k, int) else 5

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    query_tokens = bm25s.tokenize([question])

    bm25_results, bm25_scores = full_bm25s.retrieve(
        query_tokens,
        k=candidate_k
    )

    bm25_indices = bm25_results[0]
    bm25_scores = bm25_scores[0]

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    semantic_scores = np.dot(
        full_embeddings,
        query_embedding
    )

    semantic_indices = np.argsort(
        semantic_scores
    )[::-1][:candidate_k]

    # --------------------------------------------------------
    # Combine candidates
    # --------------------------------------------------------

    candidate_indices = set(
        [int(x) for x in bm25_indices]
        + [int(x) for x in semantic_indices]
    )

    results = []

    for idx in candidate_indices:

        row = full_chunks_df.iloc[idx]

        if target_type:
            row_label = str(row.get("label", "")).lower()
            if target_type.lower() not in row_label:
                continue

        semantic_score = float(
            semantic_scores[idx]
        )

        bm25_score = 0.0

        for j in range(len(bm25_indices)):
            if int(bm25_indices[j]) == idx:
                bm25_score = float(bm25_scores[j])
                break

        normalized_bm25 = (
            bm25_score /
            (1.0 + abs(bm25_score))
        )

        combined_score = (
            0.6 * semantic_score
            + 0.4 * normalized_bm25
        )

        results.append({
            "document_id": row["document_id"],
            "label": row["label"],
            "source": row["source"],
            "chunk_index": int(idx),
            "text": str(row["text"]),
            "semantic_score": semantic_score,
            "bm25_score": bm25_score,
            "score": combined_score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return {
        "status": "success",
        "query": question,
        "count": len(results[:limit]),
        "results": results[:limit],
        "method": "hybrid_search"
    }



print("Hybrid search service created")

# ============================================================
# RAG / QUESTION ANSWERING
# ============================================================

def answer_question(question, top_k=3):

    document_id = extract_document_id(question)

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    if document_id is not None:

        doc_rows = full_chunks_df[
            full_chunks_df["document_id"].astype(str).str.lower()
            == document_id.lower()
        ]

        if len(doc_rows) == 0:
            return {
                "status": "error",
                "message": f"Document {document_id} not found."
            }

        query_embedding = embedding_model.encode(
            question,
            normalize_embeddings=True
        )

        row_positions = doc_rows.index.to_numpy()

        doc_embeddings = full_embeddings[row_positions]

        scores = np.dot(
            doc_embeddings,
            query_embedding
        )

        best_positions = np.argsort(scores)[::-1][:top_k]

        contexts = []

        for i in range(len(best_positions)):

            local_position = int(best_positions[i])
            global_position = int(
                row_positions[local_position]
            )

            row = full_chunks_df.iloc[global_position]

            contexts.append({
                "text": str(row["text"]),
                "score": float(
                    scores[local_position]
                ),
                "source_chunk": (
                    f"{document_id}_{local_position}"
                )
            })

    else:

        search_result = search_documents(
            question,
            top_k=top_k
        )

        if search_result["status"] != "success":
            return search_result

        contexts = []

        for i in range(
            len(search_result["results"])
        ):

            result = search_result["results"][i]

            contexts.append({
                "text": result["text"],
                "score": result["score"],
                "source_chunk": result[
                    "document_id"
                ]
            })

    if len(contexts) == 0:
        return {
            "status": "error",
            "message": "No relevant evidence found."
        }

    # --------------------------------------------------------
    # Build evidence
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Grounded Gemini Generation
    # --------------------------------------------------------
    try:
        from services.agent import agent_service
        answer = agent_service.answer_document_qa(
            question=question,
            context_chunks=contexts,
            document_id=document_id,
        )
    except Exception as e:
        answer = "Not Mentioned in Provided Context"

    return {
        "status": "success",
        "document_id": document_id,
        "answer": answer,
        "source_chunk": contexts[0]["source_chunk"],
        "retrieval_score": round(
            contexts[0]["score"],
            4
        ),
        "method": "gemini_grounded_rag"
    }


print("RAG service created")


# ============================================================
# RUN AGENT DELEGATION
# ============================================================

def run_agent(question):
    """
    Delegates to the model-driven Gemini agent orchestrator.
    Eliminates rule-driven plan_tools() and phrase routing.
    """
    try:
        from services.agent import agent_service
        return agent_service.run_agent(question=question)
    except Exception as e:
        logger_msg = str(e)
        return {
            "question": question,
            "tools_used": ["retrieve_documents"],
            "results": [],
            "final_answer": "Not Mentioned in Provided Context"
        }