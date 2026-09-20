import React from "react";

export default function ErrorState({ message, onRetry }) {
  return (
    <div className="error-card" role="alert">
      <div className="error-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      </div>

      <div className="error-content">
        <h4>Connection Notice</h4>
        <p>{message || "Unable to connect to DocuMind backend."}</p>
        <span className="error-help">
          Verify that FastAPI is running on <code>http://127.0.0.1:8000</code> with CORS enabled.
        </span>
      </div>

      {onRetry && (
        <button type="button" className="btn-retry" onClick={onRetry}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
          <span>Retry</span>
        </button>
      )}
    </div>
  );
}
