import React, { useState } from "react";

export default function AgentActivity({ results }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!results) {
    return null;
  }

  const normalizedResults = Array.isArray(results) ? results : [results];
  if (normalizedResults.length === 0) {
    return null;
  }

  const formatConfidence = (conf) => {
    if (typeof conf === "number") {
      return `${(conf * 100).toFixed(1)}%`;
    }
    return conf || "N/A";
  };

  const getToolDisplayName = (tool) => {
    switch (tool) {
      case "classify_document":
      case "classification":
        return "Document Classification";
      case "retrieve_documents":
      case "search":
        return "Corpus & Vector Retrieval";
      case "extract_document":
      case "metadata":
        return "Structured Field Extraction";
      case "uploaded_rag":
        return "Uploaded Document Retrieval (ChromaDB)";
      case "multi_source_synthesis":
        return "Multi-Source Synthesis";
      case "rag_qa":
        return "Grounded RAG QA";
      case "gemini_reasoning":
        return "Direct Model Reasoning";
      default:
        return tool ? tool.replace(/_/g, " ") : "Agent Operation";
    }
  };

  return (
    <section className="agent-activity-container" aria-label="Agent Execution Trace">
      <button
        type="button"
        className="activity-toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <div className="activity-toggle-left">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <span className="activity-toggle-title">Agent Execution Trace</span>
          <span className="activity-count-badge">
            {normalizedResults.length} {normalizedResults.length === 1 ? "step" : "steps"}
          </span>
        </div>

        <div className="activity-toggle-right">
          <span className="activity-toggle-hint">{isOpen ? "Hide details" : "View details"}</span>
          <svg
            className={`chevron-icon ${isOpen ? "rotated" : ""}`}
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </button>

      {isOpen && (
        <div className="activity-timeline">
          {normalizedResults.map((toolResult, index) => {
            const stepNum = index + 1;
            const isSuccess = toolResult.status === "success" || !toolResult.status;

            return (
              <div className="activity-step-card" key={index}>
                <div className="step-header">
                  <div className="step-id">
                    <span className="step-index">0{stepNum}</span>
                    <span className="step-tool-name">{getToolDisplayName(toolResult.tool)}</span>
                  </div>

                  <span className={`step-status-pill ${isSuccess ? "success" : "failed"}`}>
                    <span className="step-status-dot" />
                    {toolResult.status || "completed"}
                  </span>
                </div>

                <div className="step-details">
                  {/* Classification details */}
                  {(toolResult.predicted_class || toolResult.tool === "classify_document" || toolResult.tool === "classification") && (
                    <div className="step-meta-row">
                      {toolResult.predicted_class && (
                        <div className="meta-pair">
                          <span className="meta-key">Predicted Class</span>
                          <span className="meta-val highlight">{toolResult.predicted_class}</span>
                        </div>
                      )}
                      {toolResult.confidence !== undefined && (
                        <div className="meta-pair">
                          <span className="meta-key">Confidence</span>
                          <span className="meta-val mono">{formatConfidence(toolResult.confidence)}</span>
                        </div>
                      )}
                      {toolResult.document_id && (
                        <div className="meta-pair">
                          <span className="meta-key">Target Doc</span>
                          <span className="meta-val mono">{toolResult.document_id}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Retrieval details */}
                  {(toolResult.count !== undefined || toolResult.tool === "retrieve_documents" || toolResult.tool === "search") && (
                    <div className="step-meta-row">
                      {toolResult.query && (
                        <div className="meta-pair">
                          <span className="meta-key">Query</span>
                          <span className="meta-val">"{toolResult.query}"</span>
                        </div>
                      )}
                      <div className="meta-pair">
                        <span className="meta-key">Evidence Retrieved</span>
                        <span className="meta-val mono">{toolResult.count ?? toolResult.results?.length ?? 0} chunks</span>
                      </div>
                    </div>
                  )}

                  {/* Extraction details */}
                  {(toolResult.extracted_data || toolResult.tool === "extract_document" || toolResult.tool === "metadata") && (
                    <div className="step-meta-row">
                      {toolResult.extracted_data && typeof toolResult.extracted_data === "object" && (
                        <div className="meta-pair full-width">
                          <span className="meta-key">Extracted Fields</span>
                          <div className="extracted-fields-list">
                            {Object.entries(toolResult.extracted_data).map(([k, v]) => (
                              <span key={k} className="field-tag">
                                <span className="field-k">{k}:</span>{" "}
                                <span className="field-v">{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Uploaded Documents scope */}
                  {toolResult.documents && toolResult.documents.length > 0 && (
                    <div className="step-meta-row">
                      <div className="meta-pair full-width">
                        <span className="meta-key">Scoped Documents</span>
                        <div className="extracted-fields-list">
                          {toolResult.documents.map((d, i) => (
                            <span key={i} className="field-tag mono">{d}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
