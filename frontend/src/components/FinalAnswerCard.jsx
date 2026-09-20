import React, { useState, useMemo } from "react";
import { marked } from "marked";

// Configure marked with GitHub Flavored Markdown and line breaks
marked.setOptions({
  gfm: true,
  breaks: true,
});

export default function FinalAnswerCard({ finalAnswer, toolsUsedCount, sources = [] }) {
  const [copied, setCopied] = useState(false);
  const [expandedSource, setExpandedSource] = useState(null);

  const htmlContent = useMemo(() => {
    if (!finalAnswer) return "";
    try {
      return marked.parse(finalAnswer);
    } catch {
      return finalAnswer;
    }
  }, [finalAnswer]);

  const handleCopy = () => {
    if (finalAnswer) {
      navigator.clipboard.writeText(finalAnswer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const toggleSource = (idx) => {
    setExpandedSource(expandedSource === idx ? null : idx);
  };

  return (
    <div className="answer-card-wrapper">
      <div className="section-heading">
        <div>
          <p className="eyebrow">SYNTHESIZED INTELLIGENCE</p>
          <h2>Grounded response</h2>
        </div>

        {toolsUsedCount !== undefined && (
          <div className="tool-count-badge">
            <span className="count-indicator"></span>
            {toolsUsedCount} {toolsUsedCount === 1 ? "source / tool" : "sources / tools"} executed
          </div>
        )}
      </div>

      <div className="answer-card">
        <div className="answer-top">
          <div className="answer-label">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
            <span>FINAL ANSWER</span>
          </div>

          <button
            type="button"
            className="copy-button"
            onClick={handleCopy}
            title="Copy answer to clipboard"
          >
            {copied ? (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                <span>Copied</span>
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                </svg>
                <span>Copy</span>
              </>
            )}
          </button>
        </div>

        <div
          className="answer-text markdown-body"
          dangerouslySetInnerHTML={{ __html: htmlContent }}
        />

        {sources && sources.length > 0 && (
          <div className="sources-section">
            <div className="sources-header">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <span>Cited Sources & Evidence ({sources.length})</span>
            </div>

            <div className="sources-grid">
              {sources.map((src, idx) => (
                <div
                  key={idx}
                  className={`source-chip-card ${expandedSource === idx ? "expanded" : ""}`}
                  onClick={() => toggleSource(idx)}
                >
                  <div className="source-chip-header">
                    <span className="source-file-name">{src.filename || src.document_id || `Source ${idx + 1}`}</span>
                    <div className="source-meta-badges">
                      {src.page && <span className="source-badge page-badge">Page {src.page}</span>}
                      {src.score !== undefined && (
                        <span className="source-badge score-badge">
                          {(src.score * 100).toFixed(0)}% match
                        </span>
                      )}
                    </div>
                  </div>

                  {src.text && (
                    <p className="source-snippet">
                      {expandedSource === idx ? src.text : `${src.text.slice(0, 140)}...`}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
