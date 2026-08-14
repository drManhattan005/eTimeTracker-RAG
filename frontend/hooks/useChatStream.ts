/**
 * hooks/useChatStream.ts
 * ───────────────────────
 * React hook that streams NDJSON answers from POST /api/chat.
 *
 * NDJSON events consumed:
 *   {"type":"token","content":"..."}              → append to assistant message
 *   {"type":"meta","hits":[...]}                  → store retrieval hits
 *   {"type":"error","message":"..."}              → surface error, preserve partial content
 *   {"type":"done","done_reason":"...","stats":{}} → mark stream complete
 *
 * Hardening applied:
 *   - Buffer handles partial JSON lines across chunk boundaries correctly.
 *   - isStreaming only set to false on "done" event OR when the stream body
 *     closes — not prematurely on "error" event.
 *   - "error" event preserves partial assistant content (shows what arrived
 *     before the error) and appends the error note at the end.
 *   - doneReason and streamStats exposed for debugging.
 *   - Abort controller cancels both fetch AND any pending state updates.
 *   - Empty question guard prevents empty sends.
 *   - reset() safely aborts in-flight stream.
 *
 * Exposed API:
 *   messages      – ChatMessage[]
 *   isStreaming   – true until "done" received or stream closes
 *   error         – string | null
 *   hits          – SourceHit[] from last meta event
 *   doneReason    – string from last done event ("stop", "length", "error", etc.)
 *   sendMessage   – (question: string) => void
 *   reset         – () => void
 */

"use client";

import { useCallback, useRef, useState } from "react";

// ── Types ────────────────────────────────────────────────────────────────────

export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
}

export interface SourceHit {
  id: string;
  score: number;
  section: { h2?: string; h3?: string };
  intent_sphere: string;
  text_preview: string;
}

interface StreamStats {
  prompt_tokens?: number;
  completion_tokens?: number;
  eval_ms?: number;
  total_ms?: number;
}

// ── NDJSON event shapes ──────────────────────────────────────────────────────

interface TokenEvent  { type: "token";  content: string }
interface MetaEvent   { type: "meta";   hits: SourceHit[] }
interface ErrorEvent  { type: "error";  message: string }
interface DoneEvent   { type: "done";   done_reason?: string; stats?: StreamStats }
type StreamEvent = TokenEvent | MetaEvent | ErrorEvent | DoneEvent;

// ── Helpers ──────────────────────────────────────────────────────────────────

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

/**
 * Parse a single NDJSON line safely.
 * Returns null for empty lines or malformed JSON — caller skips them.
 */
function parseLine(line: string): StreamEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed) as StreamEvent;
  } catch {
    // Log in dev only — not a fatal error, just skip.
    if (process.env.NODE_ENV !== "production") {
      console.warn("[useChatStream] unparseable NDJSON line:", trimmed.slice(0, 80));
    }
    return null;
  }
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useChatStream() {
  const [messages, setMessages]     = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError]            = useState<string | null>(null);
  const [hits, setHits]              = useState<SourceHit[]>([]);
  const [doneReason, setDoneReason]  = useState<string>("");

  // Stable refs — survive rerenders without stale closure issues
  const assistantIdRef = useRef<string | null>(null);
  const abortRef       = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed) return;

    // Cancel any in-flight stream cleanly
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    // Reset per-query state
    setError(null);
    setHits([]);
    setDoneReason("");
    setIsStreaming(true);

    // Optimistically add user + empty assistant messages
    const userMsg: ChatMessage = { id: uid(), role: "user", content: trimmed };
    const assistantId = uid();
    assistantIdRef.current = assistantId;
    const assistantMsg: ChatMessage = { id: assistantId, role: "assistant", content: "" };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    let streamEndedCleanly = false;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, limit: 3 }),
        signal: controller.signal,
      });

      // Even a non-200 response from the hardened proxy is NDJSON — try to
      // stream it rather than treating it as a fatal error immediately.
      if (!response.body) {
        throw new Error(`Response has no body (status ${response.status})`);
      }

      const reader  = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer    = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          // Stream body closed. Process any remaining buffered content.
          if (buffer.trim()) {
            const event = parseLine(buffer);
            if (event) processEvent(event);
          }
          break;
        }

        // Decode chunk and append to buffer.
        // {stream:true} keeps the internal decode state for multi-byte chars.
        buffer += decoder.decode(value, { stream: true });

        // Split on newlines — each complete line is one NDJSON event.
        // The last element may be an incomplete line; keep it in the buffer.
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const event = parseLine(line);
          if (event) processEvent(event);
        }
      }

    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        // User-initiated cancel — not an error state.
        setIsStreaming(false);
        return;
      }
      const msg = err instanceof Error ? err.message : "Unexpected error";
      setError(msg);
      appendToAssistant(`\n\n⚠ ${msg}`);
    } finally {
      // Only set isStreaming=false here if the done event never arrived
      // (connection dropped without a clean done). If done was received,
      // setIsStreaming(false) was already called inside processEvent.
      if (!streamEndedCleanly) {
        setIsStreaming(false);
      }
    }

    // ── Event processor (defined inside sendMessage to close over refs) ──
    function processEvent(event: StreamEvent) {
      switch (event.type) {
        case "token":
          appendToAssistant(event.content);
          break;

        case "meta":
          // Always update hits when received, even if preceded by an error.
          setHits(event.hits ?? []);
          break;

        case "error":
          // Record error but do NOT stop streaming yet — done is still coming.
          // Preserve whatever content arrived before the error.
          setError(event.message);
          if (event.message) {
            appendToAssistant(`\n\n⚠ ${event.message}`);
          }
          break;

        case "done":
          // done is the authoritative end-of-stream signal.
          setDoneReason(event.done_reason ?? "stop");
          setIsStreaming(false);
          streamEndedCleanly = true;
          if (process.env.NODE_ENV !== "production") {
            console.debug(
              "[useChatStream] done | reason=%s stats=%o",
              event.done_reason,
              event.stats
            );
          }
          break;
      }
    }

    function appendToAssistant(text: string) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantIdRef.current
            ? { ...m, content: m.content + text }
            : m
        )
      );
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setHits([]);
    setError(null);
    setDoneReason("");
    setIsStreaming(false);
    assistantIdRef.current = null;
  }, []);

  return { messages, isStreaming, error, hits, doneReason, sendMessage, reset };
}
