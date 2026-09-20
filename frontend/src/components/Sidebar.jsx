import React from "react";

export default function Sidebar({
  activeNav,
  setActiveNav,
  backendOnline,
  mobileOpen,
  setMobileOpen,
}) {
  const navItems = [
    { id: "agent", label: "Agent", icon: "agent" },
    { id: "documents", label: "Documents", icon: "documents" },
  ];

  return (
    <>
      {mobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
          </div>
          <div>
            <h2>DocuMind</h2>
            <span>Document Intelligence</span>
          </div>
        </div>

        <nav>
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setActiveNav(item.id);
                setMobileOpen(false);
              }}
              className={`nav-item ${activeNav === item.id ? "active" : ""}`}
            >
              {item.id === "agent" && (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                </svg>
              )}
              {item.id === "documents" && (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                </svg>
              )}
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className={`status-dot ${backendOnline ? "online" : "offline"}`}></span>
            <span className="status-text">
              {backendOnline ? "AI system online" : "Backend connecting..."}
            </span>
          </div>

          <div className="model-badges">
            <span className="badge">ChromaDB</span>
            <span className="badge">BGE</span>
            <span className="badge">Qwen 2.5</span>
          </div>
        </div>
      </aside>
    </>
  );
}
