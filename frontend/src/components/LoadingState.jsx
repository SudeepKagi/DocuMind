import React from "react";

export default function LoadingState() {
  return (
    <div className="loading-container">
      <div className="loading-card">
        <div className="loading-radar">
          <div className="radar-circle circle-1"></div>
          <div className="radar-circle circle-2"></div>
          <div className="radar-circle circle-3"></div>
          <div className="radar-center">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          </div>
        </div>

        <div className="loading-text-group">
          <h3 className="loading-title">DocuMind is analyzing your question...</h3>
          <p className="loading-subtitle">
            Evaluating intent &bull; Planning tool sequence &bull; Retrieving context
          </p>
        </div>

        <div className="loading-stages">
          <div className="stage-pill active">
            <span className="stage-dot pulse"></span>
            <span>Intent Planning</span>
          </div>
          <div className="stage-pill">
            <span className="stage-dot"></span>
            <span>Tool Execution</span>
          </div>
          <div className="stage-pill">
            <span className="stage-dot"></span>
            <span>Answer Synthesis</span>
          </div>
        </div>
      </div>
    </div>
  );
}
