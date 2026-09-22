import React from "react";

export default function LoadingState() {
  return (
    <div className="loading-card" role="status" aria-live="polite">
      <div className="loading-indicator-track">
        <div className="loading-indicator-bar" />
      </div>

      <div className="loading-content">
        <div className="loading-spinner-circle" aria-hidden="true" />
        <div className="loading-text">
          <p className="loading-title">Synthesizing grounded answer</p>
          <p className="loading-desc">
            Evaluating query intent <span className="separator">&bull;</span> Retrieving document evidence <span className="separator">&bull;</span> Performing reasoning
          </p>
        </div>
      </div>
    </div>
  );
}
