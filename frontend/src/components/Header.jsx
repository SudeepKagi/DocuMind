import React from "react";

export default function Header({ onToggleMobile }) {
  return (
    <header className="topbar">
      <div className="topbar-row">
        <button
          className="mobile-menu-btn"
          onClick={onToggleMobile}
          aria-label="Toggle navigation menu"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>

        <div>
          <p className="eyebrow">ENTERPRISE DOCUMENT INTELLIGENCE</p>
          <h1>Ask your documents.</h1>
          <p className="subtitle">
            Autonomous multi-tool intelligence providing document classification,
            financial metadata extraction, hybrid lexical/semantic search, and grounded AI answers.
          </p>
        </div>
      </div>

      <div className="capability-pills">
        <div className="cap-pill">
          <span className="pill-dot"></span>
          <span>Classification (DistilBERT)</span>
        </div>
        <div className="cap-pill">
          <span className="pill-dot"></span>
          <span>Metadata Extraction</span>
        </div>
        <div className="cap-pill">
          <span className="pill-dot"></span>
          <span>Hybrid Search (BM25 + BGE)</span>
        </div>
        <div className="cap-pill">
          <span className="pill-dot"></span>
          <span>Grounded QA (Qwen2.5)</span>
        </div>
      </div>
    </header>
  );
}
