import React, { useState, useMemo } from "react";
import { marked } from "marked";

marked.setOptions({
  gfm: true,
  breaks: false,
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
    <article className="final-answer-container" aria-label="Grounded Answer">
      <div className="answer-card">
        <header className="answer-header">
          <div className="answer-tag">
            <span className="answer-tag-dot" />
            <span>Grounded Answer</span>
          </div>

          <div className="answer-header-actions">
            {toolsUsedCount !== undefined && (
              <span className="tool-count-pill">
                {toolsUsedCount} {toolsUsedCount === 1 ? "tool" : "tools"} executed
              </span>
            )}

            <button
              type="button"
              className="btn-copy"
              onClick={handleCopy}
              aria-label="Copy answer to clipboard"
            >
              {copied ? (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                    <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                  </svg>
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        </header>

        <div
          className="answer-body markdown-content"
          dangerouslySetInnerHTML={{ __html: htmlContent }}
        />

        {sources && sources.length > 0 && (
          <footer className="answer-sources">
            <div className="sources-header">
              <span className="sources-title">Cited Evidence ({sources.length})</span>
              <span className="sources-note">Grounded in verified document context</span>
            </div>

            <div className="sources-list">
              {sources.map((src, idx) => {
                const isExpanded = expandedSource === idx;
                const docLabel = src.filename || src.document_id || `Source ${idx + 1}`;

                return (
                  <div
                    key={idx}
                    className={`source-item ${isExpanded ? "expanded" : ""}`}
                    onClick={() => toggleSource(idx)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        toggleSource(idx);
                      }
                    }}
                    aria-expanded={isExpanded}
                  >
                    <div className="source-item-header">
                      <span className="source-doc-name" title={docLabel}>
                        {docLabel}
                      </span>
                      <div className="source-badges">
                        {src.page && <span className="source-meta-tag">Page {src.page}</span>}
                        {src.score !== undefined && (
                          <span className="source-meta-tag score">
                            {(src.score * 100).toFixed(0)}% match
                          </span>
                        )}
                      </div>
                    </div>

                    {src.text && (
                      <p className="source-snippet">
                        {isExpanded ? src.text : `${src.text.slice(0, 160).trim()}...`}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </footer>
        )}
      </div>
    </article>
  );
}
