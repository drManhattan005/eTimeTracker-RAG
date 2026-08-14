/**
 * app/page.tsx
 * ─────────────
 * Veloitt RAG chat interface.
 *
 * Features:
 *   - Dark/Light mode theme toggle in the header (top right).
 *   - Sleek floating pill input box centered at the bottom.
 *   - Clean conversational interface with live streaming assistant responses.
 *   - Automatic redirection for off-topic queries handled by backend.
 */

"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useChatStream, ChatMessage } from "@/hooks/useChatStream";

// ── Sub-components ────────────────────────────────────────────────────────────

function MessageBubble({
  message,
  isStreamingThis,
}: {
  message: ChatMessage;
  isStreamingThis: boolean;
}) {
  const isUser = message.role === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      <div className={`bubble ${isUser ? "user-bubble" : "assistant-bubble"}`}>
        <span className="bubble-role">{isUser ? "You" : "Veloitt"}</span>
        <p className="bubble-text">
          {message.content}
          {isStreamingThis && <span className="cursor" aria-hidden="true" />}
        </p>
      </div>
    </div>
  );
}

// ── Main Page Component ───────────────────────────────────────────────────────

export default function ChatPage() {
  const { messages, isStreaming, error, sendMessage, reset } = useChatStream();
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  const inputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Toggle Theme
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  // Auto-scroll to bottom when new content arrives
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const input = inputRef.current;
    if (!input || !input.value.trim() || isStreaming) return;
    sendMessage(input.value);
    input.value = "";
  }

  const streamingId =
    isStreaming && messages.length > 0
      ? messages[messages.length - 1].id
      : null;

  return (
    <div className="page-root">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <div className="brand-dot" aria-hidden="true" />
            <span className="brand-name">Veloitt RAG</span>
          </div>

          <div className="header-actions">
            {messages.length > 0 && (
              <button
                className="btn-ghost"
                onClick={reset}
                aria-label="Clear conversation"
              >
                Clear
              </button>
            )}
            <button
              className="theme-toggle-btn"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              {theme === "dark" ? <SunIcon /> : <MoonIcon />}
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Layout ────────────────────────────────────────────── */}
      <div className="body-layout">
        <main className="chat-panel">
          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="empty-state">
                <p className="empty-heading">What would you like to know about eTimeTracker?</p>
                <p className="empty-sub">
                  Ask about workforce management, attendance tracking, leave policies, and shift scheduling.
                </p>
                <div className="suggestions-grid">
                  <button
                    className="suggestion-chip"
                    onClick={() => sendMessage("How does eTimeTracker handle field sales attendance?")}
                  >
                    Field sales attendance tracking
                  </button>
                  <button
                    className="suggestion-chip"
                    onClick={() => sendMessage("What leave approval workflows are supported?")}
                  >
                    Leave approval workflows
                  </button>
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  isStreamingThis={msg.id === streamingId}
                />
              ))
            )}

            {error && (
              <div className="error-banner" role="alert">
                {error}
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* ── Floating Pill Input Bar ──────────────────────────────── */}
          <div className="pill-input-container">
            <form className="pill-input-form" onSubmit={handleSubmit}>
              <input
                ref={inputRef}
                id="question-input"
                className="pill-input"
                type="text"
                placeholder="Ask eTimeTracker assistant…"
                disabled={isStreaming}
                autoComplete="off"
                aria-label="Question input"
              />
              <button
                id="send-btn"
                className="pill-send-btn"
                type="submit"
                disabled={isStreaming}
                aria-label="Send question"
              >
                {isStreaming ? (
                  <span className="spinner" aria-label="Streaming…" />
                ) : (
                  <SendIcon />
                )}
              </button>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
}

// ── Icons ────────────────────────────────────────────────────────────────────

function SendIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}
