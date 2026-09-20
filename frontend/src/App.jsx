import React, { useState, useEffect, useRef } from "react";
import "./index.css";
import { askAgent, checkHealth } from "./api/documind";

import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import QuerySection from "./components/QuerySection";
import FinalAnswerCard from "./components/FinalAnswerCard";
import AgentActivity from "./components/AgentActivity";
import LoadingState from "./components/LoadingState";
import ErrorState from "./components/ErrorState";
import UploadedDocumentsView from "./components/UploadedDocumentsView";

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [backendOnline, setBackendOnline] = useState(true);
  const [activeNav, setActiveNav] = useState("agent");
  const [mobileOpen, setMobileOpen] = useState(false);

  const resultsEndRef = useRef(null);

  // Periodically verify backend health
  useEffect(() => {
    let isMounted = true;
    const verifyStatus = async () => {
      try {
        await checkHealth();
        if (isMounted) setBackendOnline(true);
      } catch {
        if (isMounted) setBackendOnline(false);
      }
    };

    verifyStatus();
    const interval = setInterval(verifyStatus, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await askAgent(question.trim());
      setResult(data);
      setBackendOnline(true);

      setTimeout(() => {
        resultsEndRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      setError(
        err.message || "Unable to connect to DocuMind backend. Ensure FastAPI is running on port 8000."
      );
      setBackendOnline(false);
    } finally {
      setLoading(false);
    }
  };

  // Helper to extract sources from result whether in results.sources or top level
  const extractSources = (res) => {
    if (!res) return [];
    if (res.results && Array.isArray(res.results.sources)) {
      return res.results.sources;
    }
    if (Array.isArray(res.sources)) {
      return res.sources;
    }
    return [];
  };

  return (
    <div className="app">
      <Sidebar
        activeNav={activeNav}
        setActiveNav={setActiveNav}
        backendOnline={backendOnline}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
      />

      <main className="main">
        <Header onToggleMobile={() => setMobileOpen(!mobileOpen)} />

        {activeNav === "documents" ? (
          <UploadedDocumentsView />
        ) : (
          <div className="agent-workspace-view">
            <QuerySection
              question={question}
              setQuestion={setQuestion}
              onSubmit={handleAsk}
              loading={loading}
            />

            {error && (
              <ErrorState
                message={error}
                onRetry={question.trim() ? handleAsk : null}
              />
            )}

            {loading && <LoadingState />}

            {result && !loading && (
              <section className="results" ref={resultsEndRef}>
                <FinalAnswerCard
                  finalAnswer={result.final_answer}
                  toolsUsedCount={result.tools_used?.length || 1}
                  sources={extractSources(result)}
                />

                <AgentActivity results={result.results} />
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  );
}