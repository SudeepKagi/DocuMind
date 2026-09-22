import React from "react";

export default function Sidebar({
  activeNav,
  setActiveNav,
  backendOnline,
  mobileOpen,
  setMobileOpen,
}) {
  const navItems = [
    {
      id: "agent",
      label: "Agent",
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3" />
          <path d="M12 2v3m0 14v3M2 12h3m14 0h3" />
          <path d="m4.93 4.93 2.12 2.12m9.9 9.9 2.12 2.12M4.93 19.07l2.12-2.12m9.9-9.9 2.12-2.12" />
        </svg>
      ),
    },
    {
      id: "documents",
      label: "Documents",
      icon: (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
          <path d="M8 7h8" />
          <path d="M8 11h8" />
          <path d="M8 15h5" />
        </svg>
      ),
    },
  ];

  return (
    <>
      {mobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="sidebar-header">
          <div className="brand">
            <div className="brand-mark">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect width="18" height="18" x="3" y="3" rx="4" />
                <path d="m9 12 2 2 4-4" />
              </svg>
            </div>
            <div className="brand-meta">
              <span className="brand-title">DocuMind</span>
              <span className="brand-tag">v1.0</span>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Main Navigation">
          <div className="nav-group-label">Intelligence</div>
          {navItems.map((item) => {
            const isActive = activeNav === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setActiveNav(item.id);
                  setMobileOpen(false);
                }}
                className={`nav-item ${isActive ? "active" : ""}`}
                aria-current={isActive ? "page" : undefined}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
                {isActive && <span className="nav-active-pip" />}
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="system-pill" title={backendOnline ? "Operational" : "Disconnected"}>
            <span className={`status-dot ${backendOnline ? "online" : "offline"}`} />
            <span className="system-status-text">
              {backendOnline ? "Hosted Agent Ready" : "FastAPI Offline"}
            </span>
          </div>
        </div>
      </aside>
    </>
  );
}
