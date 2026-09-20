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
    <section className="query-section">
      <div className="query-box">
        <textarea
          ref={textareaRef}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask questions across your uploaded files (resume, JD) or the enterprise corpus (contracts, invoices)..."
          rows={3}
          aria-label="Agent question input"
        />

        <div className="query-footer">
          <div className="shortcut-hint">
            <kbd>Enter</kbd> to ask &bull; <kbd>Shift + Enter</kbd> for newline
          </div>

          <div className="query-actions">
            {question.trim() && !loading && (
              <button
                type="button"
                className="btn-clear"
                onClick={() => setQuestion("")}
                title="Clear question"
              >
                Clear
              </button>
            )}

            <button
              type="button"
              className="btn-submit"
              onClick={onSubmit}
              disabled={loading || !question.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner-inline"></span>
                  <span>Synthesizing...</span>
                </>
              ) : (
                <>
                  <span>Ask Agent</span>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <line x1="5" y1="12" x2="19" y2="12" />
                    <polyline points="12 5 19 12 12 19" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="examples">
        <div className="examples-header">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
          </svg>
          <span>Suggested agent queries</span>
        </div>

        <div className="example-list">
          {exampleQuestions.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => handleSelectExample(item)}
              className="example-button"
            >
              {item}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
