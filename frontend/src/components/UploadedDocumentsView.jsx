import React, { useState, useEffect, useRef } from "react";
import {
  uploadDocument,
  getDocuments,
  getDocument,
  deleteDocument,
} from "../api/documind";

export default function UploadedDocumentsView() {
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");
  const [dragOver, setDragOver] = useState(false);

  // Document chunk viewer modal state
  const [viewModalDoc, setViewModalDoc] = useState(null);
  const [viewModalChunks, setViewModalChunks] = useState([]);
  const [loadingChunks, setLoadingChunks] = useState(false);

  // In-app Delete confirmation modal state
  const [docToDelete, setDocToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const fileInputRef = useRef(null);

  const fetchDocuments = async () => {
    setLoadingDocs(true);
    try {
      const data = await getDocuments();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error("Error fetching documents:", err);
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      await processUpload(file);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await processUpload(file);
    }
  };

  const processUpload = async (file) => {
    setUploading(true);
    setUploadError("");
    setUploadSuccess("");

    try {
      const summary = await uploadDocument(file);
      setUploadSuccess(`Successfully uploaded and indexed "${summary.filename}" into ChromaDB (${summary.chunk_count} chunks).`);
      await fetchDocuments();
    } catch (err) {
      setUploadError(err.message || "Failed to upload document.");
    } finally {
      setUploading(false);
    }
  };

  const handleViewChunks = async (doc) => {
    setViewModalDoc(doc);
    setLoadingChunks(true);
    setViewModalChunks([]);

    try {
      const details = await getDocument(doc.document_id);
      setViewModalChunks(details.chunks || []);
    } catch (err) {
      console.error("Error loading document chunks:", err);
    } finally {
      setLoadingChunks(false);
    }
  };

  const confirmDelete = async () => {
    if (!docToDelete) return;
    setDeleting(true);
    setUploadError("");
    setUploadSuccess("");

    try {
      await deleteDocument(docToDelete.document_id);
      setUploadSuccess(`Successfully deleted "${docToDelete.filename}" and removed its ChromaDB vectors.`);
      if (viewModalDoc?.document_id === docToDelete.document_id) {
        setViewModalDoc(null);
      }
      setDocToDelete(null);
      await fetchDocuments();
    } catch (err) {
      setUploadError(`Failed to delete document: ${err.message}`);
      setDocToDelete(null);
    } finally {
      setDeleting(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const formatDate = (isoStr) => {
    if (!isoStr) return "Just now";
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return isoStr;
    }
  };

  return (
    <div className="uploaded-docs-container">
      {/* Header section */}
      <div className="docs-page-header">
        <div>
          <p className="eyebrow">DOCUMENT REPOSITORY & VECTOR INDEX</p>
          <h1>Documents Management</h1>
          <p className="docs-page-description">
            Upload PDF, DOCX, TXT, or EML files. DocuMind extracts full text, chunks content,
            and indexes embeddings into <strong>ChromaDB</strong>. To ask questions, compare files, or summarize, switch to the <strong>Agent</strong> tab.
          </p>
        </div>

        <button
          className="btn-upload-primary"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          <span>{uploading ? "Uploading & Indexing..." : "Upload Document"}</span>
        </button>
      </div>

      {/* Cross-link banner directing to Agent */}
      <div className="agent-cta-banner">
        <div className="cta-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
          </svg>
        </div>
        <div className="cta-text">
          <strong>Ask questions directly in the Agent</strong>
          <span>The Agent automatically searches your uploaded documents and cross-compares multiple files (e.g., "Compare my resume with the internship JD").</span>
        </div>
      </div>

      {/* Upload Drop Zone */}
      <div
        className={`upload-dropzone ${dragOver ? "drag-over" : ""} ${uploading ? "uploading" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.eml"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />

        <div className="dropzone-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>

        {uploading ? (
          <div className="upload-progress-box">
            <div className="spinner"></div>
            <p className="upload-main-text">Processing text & generating ChromaDB vectors...</p>
            <p className="upload-sub-text">Extracting structure, chunking text, and embedding with BGE</p>
          </div>
        ) : (
          <>
            <p className="upload-main-text">
              Click or drag files here to upload
            </p>
            <p className="upload-sub-text">
              Supports <strong>PDF</strong> (with page tracking), <strong>DOCX</strong> (tables & runs), <strong>TXT</strong>, and <strong>EML</strong>
            </p>
            <div className="format-badges">
              <span className="fmt-badge">.PDF</span>
              <span className="fmt-badge">.DOCX</span>
              <span className="fmt-badge">.TXT</span>
              <span className="fmt-badge">.EML</span>
            </div>
          </>
        )}
      </div>

      {uploadError && (
        <div className="upload-feedback error">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{uploadError}</span>
        </div>
      )}

      {uploadSuccess && (
        <div className="upload-feedback success">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
          <span>{uploadSuccess}</span>
        </div>
      )}

      {/* Uploaded Documents List */}
      <div className="docs-list-section">
        <div className="docs-list-header">
          <h2>Uploaded Documents ({documents.length})</h2>
          <button className="btn-refresh" onClick={fetchDocuments} title="Refresh document list">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M23 4v6h-6" />
              <path d="M1 20v-6h6" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
            <span>Refresh</span>
          </button>
        </div>

        {loadingDocs ? (
          <div className="docs-loading-state">
            <div className="spinner"></div>
            <span>Loading document catalog...</span>
          </div>
        ) : documents.length === 0 ? (
          <div className="docs-empty-state">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="12" y1="18" x2="12" y2="12" />
              <line x1="9" y1="15" x2="15" y2="15" />
            </svg>
            <h3>No uploaded documents</h3>
            <p>Upload a PDF, DOCX, TXT, or EML above to start querying them with the Agent.</p>
          </div>
        ) : (
          <div className="docs-grid">
            {documents.map((doc) => (
              <div key={doc.document_id} className="doc-card">
                <div className="doc-card-top">
                  <span className={`doc-type-badge ${doc.file_type.toLowerCase()}`}>
                    {doc.file_type}
                  </span>
                  <span className="doc-date">{formatDate(doc.upload_time)}</span>
                </div>

                <div className="doc-card-main">
                  <div className="doc-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
                    </svg>
                  </div>
                  <div className="doc-info">
                    <h4 className="doc-filename" title={doc.filename}>{doc.filename}</h4>
                    <div className="doc-stats">
                      <span>{doc.chunk_count} chunks</span>
                      {doc.page_count ? <span>&bull; {doc.page_count} {doc.page_count === 1 ? "page" : "pages"}</span> : null}
                      <span>&bull; {formatFileSize(doc.file_size)}</span>
                    </div>
                  </div>
                </div>

                <div className="doc-card-bottom">
                  <span className={`status-tag ${doc.extraction_status}`}>
                    <span className="status-tag-dot"></span>
                    {doc.extraction_status.toUpperCase()}
                  </span>

                  <div className="doc-actions">
                    <button
                      className="btn-action view"
                      onClick={() => handleViewChunks(doc)}
                      title="View extracted chunks and pages"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                      <span>View</span>
                    </button>

                    <button
                      className="btn-action delete"
                      onClick={() => setDocToDelete(doc)}
                      title="Delete document and remove ChromaDB vectors"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                      <span>Delete</span>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {docToDelete && (
        <div className="modal-backdrop" onClick={() => !deleting && setDocToDelete(null)}>
          <div className="modal-content modal-confirm-delete" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <div className="delete-modal-icon">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    <line x1="10" y1="11" x2="10" y2="17" />
                    <line x1="14" y1="11" x2="14" y2="17" />
                  </svg>
                </div>
                <div>
                  <h3 style={{ margin: 0, fontSize: "17px", color: "var(--text-main)" }}>Delete Document</h3>
                  <p className="modal-subtitle">Permanent removal from DocuMind</p>
                </div>
              </div>
              <button
                className="btn-modal-close"
                onClick={() => !deleting && setDocToDelete(null)}
                disabled={deleting}
              >
                &times;
              </button>
            </div>

            <div className="modal-body" style={{ padding: "20px 24px" }}>
              <p style={{ margin: "0 0 14px 0", color: "var(--text-main)", fontSize: "14px", lineHeight: "1.6" }}>
                Are you sure you want to delete <strong>{docToDelete.filename}</strong>?
              </p>
              <div className="delete-impact-box">
                <p style={{ margin: "0 0 6px 0", fontSize: "12.5px", fontWeight: 700, color: "#b91c1c" }}>
                  This will permanently remove:
                </p>
                <ul style={{ margin: 0, paddingLeft: "18px", fontSize: "12.5px", color: "var(--text-muted)", lineHeight: "1.6" }}>
                  <li>All ChromaDB vector embeddings ({docToDelete.chunk_count} chunks)</li>
                  <li>PostgreSQL ownership & document records</li>
                  <li>Raw uploaded storage file from disk</li>
                </ul>
              </div>
            </div>

            <div className="modal-footer" style={{ gap: "10px" }}>
              <button
                className="btn-modal-cancel"
                onClick={() => setDocToDelete(null)}
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                className="btn-modal-confirm-delete"
                onClick={confirmDelete}
                disabled={deleting}
              >
                {deleting ? (
                  <>
                    <span className="spinner-sm"></span>
                    <span>Deleting from ChromaDB...</span>
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                    <span>Delete Permanently</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Document Chunks View Modal */}
      {viewModalDoc && (
        <div className="modal-backdrop" onClick={() => setViewModalDoc(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <span className="modal-badge">{viewModalDoc.file_type}</span>
                <h3>{viewModalDoc.filename}</h3>
                <p className="modal-subtitle">
                  {viewModalDoc.chunk_count} chunks indexed in ChromaDB &bull; {viewModalDoc.page_count ? `${viewModalDoc.page_count} pages` : ""} &bull; {formatFileSize(viewModalDoc.file_size)}
                </p>
              </div>
              <button className="btn-modal-close" onClick={() => setViewModalDoc(null)}>
                &times;
              </button>
            </div>

            <div className="modal-body">
              {loadingChunks ? (
                <div className="modal-loading">
                  <div className="spinner"></div>
                  <span>Loading chunks...</span>
                </div>
              ) : viewModalChunks.length === 0 ? (
                <p className="empty-chunks-msg">No chunks available for this document.</p>
              ) : (
                <div className="chunks-list">
                  {viewModalChunks.map((chunk, idx) => (
                    <div key={idx} className="chunk-card">
                      <div className="chunk-header">
                        <span className="chunk-num">Chunk #{idx + 1}</span>
                        {chunk.page && <span className="chunk-page">Page {chunk.page}</span>}
                        {chunk.word_count && <span className="chunk-words">{chunk.word_count} words</span>}
                      </div>
                      <p className="chunk-text">{chunk.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button className="btn-modal-done" onClick={() => setViewModalDoc(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
