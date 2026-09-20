/**
 * DocuMind Frontend API Client
 * Interfaces with the FastAPI backend running at http://127.0.0.1:8000
 */

const API_BASE_URL = "http://127.0.0.1:8000";

async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = { ...options.headers };

  // Only set Content-Type: application/json if a body is present and not FormData
  if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const config = {
    ...options,
    headers,
  };

  try {
    const response = await fetch(url, config);

    if (!response.ok) {
      let errorDetail = `Request failed with status ${response.status}`;
      try {
        const errorJson = await response.json();
        if (errorJson?.detail) {
          errorDetail = typeof errorJson.detail === "string" 
            ? errorJson.detail 
            : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Response wasn't JSON
      }
      throw new Error(errorDetail);
    }

    return await response.json();
  } catch (error) {
    if (error.name === "TypeError" && error.message.includes("fetch")) {
      throw new Error("Unable to connect to DocuMind backend. Ensure FastAPI is running on port 8000.");
    }
    throw error;
  }
}

/**
 * Run multi-tool DocuMind Agent
 * @param {string} question 
 */
export async function askAgent(question) {
  return request("/agent", {
    method: "POST",
    body: JSON.stringify({ question: question.trim() }),
  });
}

/**
 * Classify document using DistilBERT classifier
 * @param {string} question 
 */
export async function classifyDocument(question) {
  return request("/classify", {
    method: "POST",
    body: JSON.stringify({ question: question.trim() }),
  });
}

/**
 * Extract financial/table metadata from invoice
 * @param {string} question 
 */
export async function extractMetadata(question) {
  return request("/extract", {
    method: "POST",
    body: JSON.stringify({ question: question.trim() }),
  });
}

/**
 * Perform hybrid BM25 + BGE semantic search
 * @param {string} question 
 * @param {number} topK 
 */
export async function searchDocuments(question, topK = 5) {
  return request("/search", {
    method: "POST",
    body: JSON.stringify({ question: question.trim(), top_k: topK }),
  });
}

/**
 * Grounded QA using BGE retrieval & Qwen2.5 on corpus
 * @param {string} question 
 */
export async function askQuestion(question) {
  return request("/qa", {
    method: "POST",
    body: JSON.stringify({ question: question.trim() }),
  });
}

/**
 * Check backend health
 */
export async function checkHealth() {
  return request("/health", {
    method: "GET",
  });
}

// ============================================================
// Uploaded Documents Endpoints
// ============================================================

/**
 * Upload a document (PDF, DOCX, TXT, EML)
 * @param {File} file 
 */
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  return request("/documents/upload", {
    method: "POST",
    body: formData,
  });
}

/**
 * List all uploaded documents
 */
export async function getDocuments() {
  return request("/documents", {
    method: "GET",
  });
}

/**
 * Get document details and chunks
 * @param {string} documentId 
 */
export async function getDocument(documentId) {
  return request(`/documents/${documentId}`, {
    method: "GET",
  });
}

/**
 * Delete an uploaded document
 * @param {string} documentId 
 */
export async function deleteDocument(documentId) {
  return request(`/documents/${documentId}`, {
    method: "DELETE",
  });
}

/**
 * Ask a question specifically about an uploaded document (isolated QA)
 * @param {string} documentId 
 * @param {string} question 
 */
export async function askUploadedDocument(documentId, question) {
  return request(`/documents/${documentId}/qa`, {
    method: "POST",
    body: JSON.stringify({ question: question.trim() }),
  });
}

export default {
  askAgent,
  classifyDocument,
  extractMetadata,
  searchDocuments,
  askQuestion,
  checkHealth,
  uploadDocument,
  getDocuments,
  getDocument,
  deleteDocument,
  askUploadedDocument,
};
