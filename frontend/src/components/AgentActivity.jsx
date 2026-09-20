import React from "react";

export default function AgentActivity({ results }) {
  if (!results) {
    return null;
  }

  const normalizedResults = Array.isArray(results) ? results : [results];
  if (normalizedResults.length === 0) {
    return null;
  }

  const formatConfidence = (conf) => {
    if (typeof conf === "number") {
      return `${(conf * 100).toFixed(2)}%`;
    }
    return conf || "N/A";
  };

  const getToolDisplayName = (tool) => {
    switch (tool) {
      case "uploaded_rag":
        return "Uploaded Documents Vector Retrieval (ChromaDB)";
      case "multi_source_synthesis":
        return "Multi-Source Synthesis (Corpus + Uploaded)";
      case "classification":
        return "Document Classification";
      case "metadata":
        return "Metadata Extraction";
      case "search":
        return "Hybrid Search";
      case "rag_qa":
        return "Grounded RAG QA";
      default:
        return tool || "Agent Tool";
    }
  };

  return (
    <div className="tool-section">
      <div className="section-heading small">
        <div>
          <p className="eyebrow">AGENT EXECUTION TRACE</p>
          <h2>Agent activity</h2>
        </div>
        <span className="activity-subtitle">
          Sequential multi-tool planner invocation
        </span>
      </div>

      <div className="tool-grid">
        {normalizedResults.map((toolResult, index) => {
          const stepNum = String(index + 1).padStart(2, "0");
          const isSuccess = toolResult.status === "success";

          return (
            <div
              className={`tool-card ${toolResult.tool}`}
              key={`${toolResult.tool}-${index}`}
            >
              <div className="tool-top">
                <div className="tool-step-badge">
                  <span className="step-num">{stepNum}</span>
                  <span className="tool-tag">{toolResult.tool}</span>
                </div>

                <span className={`status-pill ${isSuccess ? "success" : "failed"}`}>
                  <span className="status-indicator-dot"></span>
                  {toolResult.status || "completed"}
                </span>
              </div>

              <h3 className="tool-title">{getToolDisplayName(toolResult.tool)}</h3>

              {/* Uploaded RAG / Multi-Source */}
              {(toolResult.tool === "uploaded_rag" || toolResult.tool === "multi_source_synthesis") && (
                <div className="tool-body">
                  {toolResult.documents && toolResult.documents.length > 0 && (
                    <div className="metadata-row">
                      <span className="meta-label">Matched Documents</span>
                      <div className="doc-id-chip-list">
                        {toolResult.documents.map((docName, i) => (
                          <span key={i} className="doc-chip uploaded-doc-chip">
                            {docName}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {toolResult.mode && (
                    <div className="metadata-row">
                      <span className="meta-label">Retrieval Scope</span>
                      <span className="meta-value method-tag">{toolResult.mode}</span>
                    </div>
                  )}

                  {toolResult.sources && toolResult.sources.length > 0 && (
                    <div className="metadata-row">
                      <span className="meta-label">Evidence Chunks</span>
                      <span className="meta-value score-val">{toolResult.sources.length} retrieved</span>
                    </div>
                  )}
                </div>
              )}

              {/* Classification */}
              {toolResult.tool === "classification" && (
                <div className="tool-body">
                  <div className="value-highlight class-highlight">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
                    </svg>
                    <span>{toolResult.predicted_class || "Unknown"}</span>
                  </div>

                  <div className="metadata-row">
                    <span className="meta-label">Confidence</span>
                    <span className="meta-value confidence-value">
                      {formatConfidence(toolResult.confidence)}
                    </span>
                  </div>

                  {toolResult.document_id && (
                    <div className="metadata-row">
                      <span className="meta-label">Document</span>
                      <span className="meta-value doc-id-tag">{toolResult.document_id}</span>
                    </div>
                  )}

                  {toolResult.probabilities && (
                    <div className="probability-breakdown">
                      <span className="breakdown-title">Class Distribution</span>
                      <div className="probability-bars">
                        {Object.entries(toolResult.probabilities).map(([cls, prob]) => (
                          <div key={cls} className="prob-item">
                            <span className="prob-name">{cls}</span>
                            <div className="prob-bar-track">
                              <div
                                className="prob-bar-fill"
                                style={{ width: `${Math.min(100, Math.max(2, prob * 100))}%` }}
                              ></div>
                            </div>
                            <span className="prob-num">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Metadata */}
              {toolResult.tool === "metadata" && (
                <div className="tool-body">
                  <div className="value-highlight amount-highlight">
                    <span className="currency-val">{toolResult.value || "Not found"}</span>
                  </div>

                  <div className="metadata-row">
                    <span className="meta-label">Field</span>
                    <span className="meta-value field-tag">{toolResult.field}</span>
                  </div>

                  {toolResult.document_id && (
                    <div className="metadata-row">
                      <span className="meta-label">Document</span>
                      <span className="meta-value doc-id-tag">{toolResult.document_id}</span>
                    </div>
                  )}

                  {toolResult.method && (
                    <div className="metadata-row">
                      <span className="meta-label">Method</span>
                      <span className="meta-value method-tag">{toolResult.method}</span>
                    </div>
                  )}
                </div>
              )}

              {/* Search */}
              {toolResult.tool === "search" && (
                <div className="tool-body">
                  <div className="value-highlight search-highlight">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                    <span>{toolResult.count ?? (toolResult.results?.length || 0)} documents</span>
                  </div>

                  {toolResult.search_term && (
                    <div className="metadata-row">
                      <span className="meta-label">Search Term</span>
                      <span className="meta-value term-tag">"{toolResult.search_term}"</span>
                    </div>
                  )}

                  {toolResult.results && toolResult.results.length > 0 && (
                    <div className="doc-id-list-wrapper">
                      <span className="meta-label">Matching Documents</span>
                      <div className="doc-id-chip-list">
                        {toolResult.results.map((res, i) => (
                          <span key={i} className="doc-chip" title={res.text || res.document_id}>
                            {res.document_id}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {toolResult.method && (
                    <div className="metadata-row">
                      <span className="meta-label">Method</span>
                      <span className="meta-value method-tag">{toolResult.method}</span>
                    </div>
                  )}
                </div>
              )}

              {/* RAG QA */}
              {toolResult.tool === "rag_qa" && (
                <div className="tool-body">
                  <div className="value-highlight rag-highlight">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                    </svg>
                    <span>Grounded QA</span>
                  </div>

                  <div className="metadata-row">
                    <span className="meta-label">Source</span>
                    <span className="meta-value source-chunk-tag">
                      {toolResult.source_chunk || "retrieved_chunk"}
                    </span>
                  </div>

                  {toolResult.retrieval_score !== undefined && (
                    <div className="metadata-row">
                      <span className="meta-label">Retrieval Score</span>
                      <span className="meta-value score-val">
                        {toolResult.retrieval_score}
                      </span>
                    </div>
                  )}

                  {toolResult.method && (
                    <div className="metadata-row">
                      <span className="meta-label">Engine</span>
                      <span className="meta-value method-tag">{toolResult.method}</span>
                    </div>
                  )}
                </div>
              )}

              {/* Error or unsupported fallback */}
              {toolResult.message && !isSuccess && (
                <div className="tool-body error-body">
                  <p className="tool-error-msg">{toolResult.message}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
