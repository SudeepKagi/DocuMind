import React from "react";

export default function ErrorState({ message, onRetry }) {
  const isConnectionError =
    message &&
    (message.toLowerCase().includes("unable to connect") ||
      message.toLowerCase().includes("failed to fetch") ||
      message.toLowerCase().includes("network"));

  return (
    <div className="error-card" role="alert">
      <div className="error-icon" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      </div>

      <div className="error-details">
        <h4 className="error-title">
          {isConnectionError ? "Backend Connection Issue" : "Unable to Complete Request"}
        </h4>
        <p className="error-message">{message || "An unexpected error occurred while communicating with the backend."}</p>
        {isConnectionError && (
          <span className="error-hint">
            Ensure FastAPI backend is running on <code>http://127.0.0.1:8000</code>.
          </span>
        )}
      </div>

      {onRetry && (
        <button type="button" className="btn-retry" onClick={onRetry}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
          <span>Retry</span>
        </button>
      )}
    </div>
  );
}
