import React, { useRef } from "react";

export default function QuerySection({
  question,
  setQuestion,
  onSubmit,
  loading,
}) {
  const textareaRef = useRef(null);

  const exampleQuestions = [
    "Compare my resume with the internship JD.",
    "Summarize my uploaded resume.",
    "What skills are required in the internship JD?",
    "What type of document is invoice_0292?",
    "Which contracts mention termination fees?",
    "What is the total amount of invoice_0292?",
  ];

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading && question.trim()) {
        onSubmit();
      }
    }
  };

  const handleSelectExample = (example) => {
    setQuestion(example);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  return (
    <section className="query-section" aria-label="Query Workspace">
      <div className="agent-page-header">
        <h1 className="page-title">Ask your documents</h1>
        <p className="page-subtitle">
          Query contracts, invoices, and your uploaded files with grounded, model-driven reasoning.
        </p>
      </div>

      <div className={`query-container ${question.trim() ? "has-content" : ""}`}>
        <textarea
          ref={textareaRef}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents..."
          rows={3}
          aria-label="Ask a question about your documents"
          disabled={loading}
        />

        <div className="query-footer">
          <div className="shortcut-hint" aria-hidden="true">
            <kbd>Enter</kbd> to ask <span className="separator">&bull;</span> <kbd>Shift + Enter</kbd> for newline
          </div>

          <div className="query-actions">
            {question.trim() && !loading && (
              <button
                type="button"
                className="btn-clear"
                onClick={() => setQuestion("")}
                title="Clear question input"
              >
                Clear
              </button>
            )}

            <button
              type="button"
              className="btn-ask"
              onClick={onSubmit}
              disabled={loading || !question.trim()}
              aria-label="Submit question to DocuMind Agent"
            >
              {loading ? (
                <>
                  <span className="btn-spinner" aria-hidden="true" />
                  <span>Synthesizing...</span>
                </>
              ) : (
                <>
                  <span>Ask</span>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12" />
                    <polyline points="12 5 19 12 12 19" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="suggestions-block">
        <span className="suggestions-label">Suggested queries</span>
        <div className="suggestions-chips">
          {exampleQuestions.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => handleSelectExample(item)}
              className="suggestion-chip"
              disabled={loading}
            >
              {item}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
