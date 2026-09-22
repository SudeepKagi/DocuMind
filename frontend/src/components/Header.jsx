import React from "react";

export default function Header({ backendOnline, docCount = 0 }) {
  return (
    <header className="site-header">
      <div className="header-container">
        <div className="header-brand">
          <div className="brand-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
              <rect width="18" height="18" x="3" y="3" rx="4" />
              <path d="m9 12 2 2 4-4" />
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-name">DocuMind</span>
            <span className="brand-badge">Enterprise AI</span>
          </div>
        </div>

        <div className="header-meta">
          {docCount > 0 && (
            <div className="meta-stat-pill" title={`${docCount} documents indexed in ChromaDB`}>
              <span className="stat-num">{docCount}</span>
              <span className="stat-label">{docCount === 1 ? "document" : "documents"}</span>
            </div>
          )}

          <div className={`status-pill-header ${backendOnline ? "online" : "offline"}`}>
            <span className="status-dot-pulse" />
            <span className="status-pill-text">
              {backendOnline ? "AI system online" : "Connecting to API..."}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
