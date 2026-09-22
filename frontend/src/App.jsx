import React, { useState, useEffect, useRef } from "react";
import "./index.css";
import { askAgent, checkHealth } from "./api/documind";

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
  const [docCount, setDocCount] = useState(0);

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
      setBackendOnline(true);

      if (data && data.status === "error") {
        setError(
          data.final_answer ||
            "The document intelligence service is temporarily experiencing high traffic. Please retry in a moment."
        );
        setResult(null);
      } else {
        setResult(data);
        setError("");
        setTimeout(() => {
          resultsEndRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 100);
      }
    } catch (err) {
      setError(
        err.message || "Unable to connect to DocuMind backend. Ensure FastAPI is running on port 8000."
      );
      setBackendOnline(false);
    } finally {
      setLoading(false);
    }
  };

  const handleAskDocumentFromPanel = (doc) => {
    setQuestion(`Summarize key obligations and terms in ${doc.filename}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
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
    <div className="app-shell">
      <Header backendOnline={backendOnline} docCount={docCount} />

      <main className="studio-canvas">
        <div className="studio-grid">
          {/* Left Column: Focused Agent Intelligence Workspace */}
          <section className="agent-column" aria-label="Agent Workspace">
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
              <div className="results-wrapper" ref={resultsEndRef}>
                <FinalAnswerCard
                  finalAnswer={result.final_answer}
                  toolsUsedCount={result.tools_used?.length || 1}
                  sources={extractSources(result)}
                />

                <AgentActivity results={result.results} />
              </div>
            )}
          </section>

          {/* Right Column: Integrated Document Repository Panel */}
          <aside className="documents-column" aria-label="Document Management">
            <UploadedDocumentsView
              onDocCountChange={setDocCount}
              onAskInAgent={handleAskDocumentFromPanel}
            />
          </aside>
        </div>
      </main>
    </div>
  );
}