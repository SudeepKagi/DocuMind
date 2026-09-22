import React, { useState, useEffect, useRef, useMemo } from "react";
import { marked } from "marked";
import {
  uploadDocument,
  getDocuments,
  getDocument,
  deleteDocument,
  askUploadedDocument,
} from "../api/documind";

marked.setOptions({
  gfm: true,
  breaks: false,
});

export default function UploadedDocumentsView({ onDocCountChange, onAskInAgent }) {
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");
  const [dragOver, setDragOver] = useState(false);

  // Document chunk viewer modal
  const [viewModalDoc, setViewModalDoc] = useState(null);
  const [viewModalChunks, setViewModalChunks] = useState([]);
  const [loadingChunks, setLoadingChunks] = useState(false);

  // Delete confirmation modal
  const [docToDelete, setDocToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  // Document QA modal
  const [qaModalDoc, setQaModalDoc] = useState(null);
  const [qaQuestion, setQaQuestion] = useState("");
  const [qaLoading, setQaLoading] = useState(false);
  const [qaResult, setQaResult] = useState(null);
  const [qaError, setQaError] = useState("");
  const [copiedAnswer, setCopiedAnswer] = useState(false);

  const fileInputRef = useRef(null);

  const fetchDocuments = async () => {
    setLoadingDocs(true);
    try {
      const data = await getDocuments();
      const docs = data.documents || [];
      setDocuments(docs);
      if (onDocCountChange) onDocCountChange(docs.length);
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
      setUploadSuccess(`Indexed "${summary.filename}" into ChromaDB (${summary.chunk_count} chunks).`);
      await fetchDocuments();
    } catch (err) {
      setUploadError(err.message || "Failed to upload and index document.");
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

  const handleOpenQA = (doc) => {
    setQaModalDoc(doc);
    setQaQuestion("");
    setQaResult(null);
    setQaError("");
    setCopiedAnswer(false);
  };

  const handleAskDocument = async (e) => {
    e?.preventDefault();
    if (!qaQuestion.trim() || !qaModalDoc || qaLoading) return;

    setQaLoading(true);
    setQaError("");
    setQaResult(null);

    try {
      const resp = await askUploadedDocument(qaModalDoc.document_id, qaQuestion.trim());
      setQaResult(resp);
    } catch (err) {
      setQaError(err.message || "Failed to retrieve answer for this document.");
    } finally {
      setQaLoading(false);
    }
  };

  const confirmDelete = async () => {
    if (!docToDelete) return;
    setDeleting(true);
    setUploadError("");
    setUploadSuccess("");

    try {
      await deleteDocument(docToDelete.document_id);
      setUploadSuccess(`Deleted "${docToDelete.filename}" and removed its ChromaDB vectors.`);
      if (viewModalDoc?.document_id === docToDelete.document_id) {
        setViewModalDoc(null);
      }
      if (qaModalDoc?.document_id === docToDelete.document_id) {
        setQaModalDoc(null);
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
      });
    } catch {
      return isoStr;
    }
  };

  const parsedQaAnswer = useMemo(() => {
    if (!qaResult?.answer) return "";
    try {
      return marked.parse(qaResult.answer);
    } catch {
      return qaResult.answer;
    }
  }, [qaResult?.answer]);

  return (
    <div className="documents-panel-wrapper">
      {/* Panel Header */}
      <div className="panel-header">
        <div className="panel-title-group">
          <div className="panel-icon" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
              <path d="M8 7h8" />
              <path d="M8 11h8" />
            </svg>
          </div>
          <div>
            <h2 className="panel-title">Document Repository</h2>
            <p className="panel-subtitle">ChromaDB Vector Index</p>
          </div>
        </div>

        <div className="panel-header-actions">
          <button
            type="button"
            className="btn-refresh-sm"
            onClick={fetchDocuments}
            title="Refresh documents"
            aria-label="Refresh document list"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
            </svg>
          </button>
        </div>
      </div>

      {/* Upload Dropzone */}
      <div
        className={`compact-dropzone ${dragOver ? "drag-over" : ""} ${uploading ? "uploading" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload document dropzone"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.eml"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />

        {uploading ? (
          <div className="dropzone-uploading-view">
            <div className="upload-spinner-sm" />
            <span className="dropzone-uploading-text">Chunking & embedding with BGE...</span>
          </div>
        ) : (
          <div className="dropzone-idle-view">
            <div className="dropzone-icon-circle" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <div className="dropzone-copy">
              <span className="dropzone-title">Drop files or <span className="underline">browse</span></span>
              <span className="dropzone-formats">PDF, DOCX, TXT, EML</span>
            </div>
          </div>
        )}
      </div>

      {/* Feedback Banners */}
      {uploadError && (
        <div className="feedback-banner-sm error" role="alert">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{uploadError}</span>
        </div>
      )}

      {uploadSuccess && (
        <div className="feedback-banner-sm success" role="status">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
          <span>{uploadSuccess}</span>
        </div>
      )}

      {/* Document Items List */}
      <div className="panel-document-list" aria-label="Indexed Documents">
        <div className="list-meta-bar">
          <span className="list-heading">Indexed Files</span>
          <span className="list-count-badge">{documents.length}</span>
        </div>

        {loadingDocs ? (
          <div className="list-loading-state">
            <div className="catalog-spinner" />
            <span>Loading documents...</span>
          </div>
        ) : documents.length === 0 ? (
          <div className="list-empty-state">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
            <p className="empty-text-bold">No documents indexed</p>
            <p className="empty-text-sub">Upload a file above to query it with the Agent.</p>
          </div>
        ) : (
          <div className="document-rows">
            {documents.map((doc) => {
              const fileType = (doc.file_type || "TXT").toUpperCase();

              return (
                <div key={doc.document_id} className="document-row-card">
                  <div className="row-main">
                    <div className="row-icon-col">
                      <span className="type-badge-mini">{fileType}</span>
                    </div>

                    <div className="row-info-col">
                      <h4 className="row-filename" title={doc.filename}>
                        {doc.filename}
                      </h4>
                      <div className="row-meta">
                        <span className="meta-chunk-count">{doc.chunk_count} chunks</span>
                        <span className="meta-sep">&bull;</span>
                        <span>{formatFileSize(doc.file_size)}</span>
                        <span className="meta-sep">&bull;</span>
                        <span>{formatDate(doc.upload_time)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="row-actions">
                    <button
                      type="button"
                      className="row-btn view"
                      onClick={() => handleViewChunks(doc)}
                      title="Inspect extracted chunks"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                      <span>View</span>
                    </button>

                    <button
                      type="button"
                      className="row-btn ask"
                      onClick={() => {
                        if (onAskInAgent) {
                          onAskInAgent(doc);
                        } else {
                          handleOpenQA(doc);
                        }
                      }}
                      title="Ask a question about this document"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                        <line x1="12" y1="17" x2="12.01" y2="17" />
                      </svg>
                      <span>Ask</span>
                    </button>

                    <button
                      type="button"
                      className="row-btn delete"
                      onClick={() => setDocToDelete(doc)}
                      title="Delete document"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 6h18m-2 0v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6m3 0V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                      </svg>
                      <span>Delete</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Document QA Modal (Isolated QA Workflow) */}
      {qaModalDoc && (
        <div className="modal-backdrop" onClick={() => !qaLoading && setQaModalDoc(null)}>
          <div className="modal-content modal-qa-dialog" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="qa-modal-title">
            <header className="modal-header">
              <div className="modal-header-info">
                <span className="modal-badge">Isolated Document QA</span>
                <h3 id="qa-modal-title" className="modal-title" title={qaModalDoc.filename}>
                  {qaModalDoc.filename}
                </h3>
                <p className="modal-subtitle">
                  Queries are strictly scoped to this document's {qaModalDoc.chunk_count} indexed chunks.
                </p>
              </div>

              <button
                type="button"
                className="btn-modal-close"
                onClick={() => setQaModalDoc(null)}
                disabled={qaLoading}
                aria-label="Close dialog"
              >
                &times;
              </button>
            </header>

            <div className="modal-body qa-modal-body">
              <form onSubmit={handleAskDocument} className="qa-input-form">
                <div className="qa-input-wrapper">
                  <input
                    type="text"
                    value={qaQuestion}
                    onChange={(e) => setQaQuestion(e.target.value)}
                    placeholder={`Ask a question about ${qaModalDoc.filename}...`}
                    disabled={qaLoading}
                    autoFocus
                  />
                  <button
                    type="submit"
                    className="btn-qa-ask"
                    disabled={qaLoading || !qaQuestion.trim()}
                  >
                    {qaLoading ? (
                      <>
                        <span className="btn-spinner" aria-hidden="true" />
                        <span>Searching...</span>
                      </>
                    ) : (
                      <>
                        <span>Ask</span>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
                          <line x1="5" y1="12" x2="19" y2="12" />
                          <polyline points="12 5 19 12 12 19" />
                        </svg>
                      </>
                    )}
                  </button>
                </div>
              </form>

              {qaError && (
                <div className="feedback-banner-sm error" role="alert">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                  <span>{qaError}</span>
                </div>
              )}

              {qaLoading && (
                <div className="qa-loading-block">
                  <div className="catalog-spinner" />
                  <p className="qa-loading-text">Retrieving matching chunks and synthesizing grounded answer...</p>
                </div>
              )}

              {qaResult && !qaLoading && (
                <div className="qa-result-card">
                  <div className="qa-result-header">
                    <span className="qa-result-label">Grounded Answer</span>
                    <button
                      type="button"
                      className="btn-copy-sm"
                      onClick={() => {
                        if (qaResult.answer) {
                          navigator.clipboard.writeText(qaResult.answer);
                          setCopiedAnswer(true);
                          setTimeout(() => setCopiedAnswer(false), 2000);
                        }
                      }}
                    >
                      {copiedAnswer ? "Copied" : "Copy"}
                    </button>
                  </div>

                  <div
                    className="qa-answer-markdown markdown-content"
                    dangerouslySetInnerHTML={{ __html: parsedQaAnswer }}
                  />

                  {qaResult.sources && qaResult.sources.length > 0 && (
                    <div className="qa-sources-block">
                      <span className="qa-sources-title">Retrieved Evidence ({qaResult.sources.length} chunks)</span>
                      <div className="qa-sources-list">
                        {qaResult.sources.map((src, i) => (
                          <div key={i} className="qa-source-chunk">
                            <div className="qa-chunk-meta">
                              <span className="qa-chunk-page">Page {src.page || 1}</span>
                              {src.score !== undefined && (
                                <span className="qa-chunk-score">{(src.score * 100).toFixed(0)}% match</span>
                              )}
                            </div>
                            <p className="qa-chunk-snippet">{src.text}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <footer className="modal-footer">
              <button
                type="button"
                className="btn-modal-done"
                onClick={() => setQaModalDoc(null)}
              >
                Close
              </button>
            </footer>
          </div>
        </div>
      )}

      {/* Document Chunks View Modal */}
      {viewModalDoc && (
        <div className="modal-backdrop" onClick={() => setViewModalDoc(null)}>
          <div className="modal-content modal-chunks-dialog" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="chunks-modal-title">
            <header className="modal-header">
              <div className="modal-header-info">
                <span className="modal-badge">{viewModalDoc.file_type}</span>
                <h3 id="chunks-modal-title" className="modal-title" title={viewModalDoc.filename}>
                  {viewModalDoc.filename}
                </h3>
                <p className="modal-subtitle">
                  {viewModalDoc.chunk_count} chunks indexed in ChromaDB &bull; {viewModalDoc.page_count ? `${viewModalDoc.page_count} pages` : ""} &bull; {formatFileSize(viewModalDoc.file_size)}
                </p>
              </div>

              <button
                type="button"
                className="btn-modal-close"
                onClick={() => setViewModalDoc(null)}
                aria-label="Close dialog"
              >
                &times;
              </button>
            </header>

            <div className="modal-body chunks-modal-body">
              {loadingChunks ? (
                <div className="catalog-loading">
                  <div className="catalog-spinner" />
                  <span>Loading chunks from storage...</span>
                </div>
              ) : viewModalChunks.length === 0 ? (
                <p className="empty-chunks-msg">No chunks available for this document.</p>
              ) : (
                <div className="chunks-list">
                  {viewModalChunks.map((chunk, idx) => (
                    <div key={idx} className="chunk-card">
                      <div className="chunk-card-header">
                        <span className="chunk-id-tag">Chunk #{idx + 1}</span>
                        <div className="chunk-badges">
                          {chunk.page && <span className="chunk-badge">Page {chunk.page}</span>}
                          {chunk.word_count && <span className="chunk-badge">{chunk.word_count} words</span>}
                        </div>
                      </div>
                      <p className="chunk-text">{chunk.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <footer className="modal-footer">
              <button
                type="button"
                className="btn-modal-done"
                onClick={() => setViewModalDoc(null)}
              >
                Done
              </button>
            </footer>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {docToDelete && (
        <div className="modal-backdrop" onClick={() => !deleting && setDocToDelete(null)}>
          <div className="modal-content modal-delete-dialog" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="delete-modal-title">
            <header className="modal-header">
              <div className="delete-header-icon" aria-hidden="true">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 6h18m-2 0v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6m3 0V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                </svg>
              </div>
              <div className="modal-header-info">
                <h3 id="delete-modal-title" className="modal-title">Delete Document</h3>
                <p className="modal-subtitle">Permanent removal from DocuMind</p>
              </div>
              <button
                type="button"
                className="btn-modal-close"
                onClick={() => !deleting && setDocToDelete(null)}
                disabled={deleting}
                aria-label="Close dialog"
              >
                &times;
              </button>
            </header>

            <div className="modal-body delete-modal-body">
              <p className="delete-confirm-text">
                Are you sure you want to delete <strong>{docToDelete.filename}</strong>?
              </p>
              <div className="delete-warning-box">
                <p className="warning-heading">This action cannot be undone and will delete:</p>
                <ul className="warning-list">
                  <li>ChromaDB vector collection entries ({docToDelete.chunk_count} chunks)</li>
                  <li>PostgreSQL ownership records and chunks metadata</li>
                  <li>Local storage file on disk</li>
                </ul>
              </div>
            </div>

            <footer className="modal-footer">
              <button
                type="button"
                className="btn-modal-cancel"
                onClick={() => setDocToDelete(null)}
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-modal-delete-confirm"
                onClick={confirmDelete}
                disabled={deleting}
              >
                {deleting ? (
                  <>
                    <span className="btn-spinner" aria-hidden="true" />
                    <span>Deleting...</span>
                  </>
                ) : (
                  <span>Delete Permanently</span>
                )}
              </button>
            </footer>
          </div>
        </div>
      )}
    </div>
  );
}
