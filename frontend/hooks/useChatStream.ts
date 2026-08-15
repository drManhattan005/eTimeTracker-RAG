/**
 * hooks/useChatStream.ts
 * ───────────────────────
 * React hook that streams NDJSON answers from POST /api/chat.
 */

"use client";

import { useCallback, useRef, useState } from "react";

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

interface TokenEvent {
  type: "token";
  token?: string;
  content?: string;
}

interface MetaEvent {
  type: "meta";
  hits?: SourceHit[];
  session_id?: string;
  effective_question?: string;
}

interface ErrorEvent {
  type: "error";
  message: string;
}

interface DoneEvent {
  type: "done";
  done_reason?: string;
  response_type?: string;
  stats?: StreamStats;
}

type StreamEvent = TokenEvent | MetaEvent | ErrorEvent | DoneEvent;

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

function makeSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `session_${Date.now()}_${uid()}`;
}

function parseLine(line: string): StreamEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed) as StreamEvent;
  } catch {
    if (process.env.NODE_ENV !== "production") {
      console.warn("[useChatStream] unparseable NDJSON line:", trimmed.slice(0, 120));
    }
    return null;
  }
}

export function useChatStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hits, setHits] = useState<SourceHit[]>([]);
  const [doneReason, setDoneReason] = useState<string>("");

  const assistantIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const sessionIdRef = useRef<string>(makeSessionId());

  const sendMessage = useCallback(async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError(null);
    setHits([]);
    setDoneReason("");
    setIsStreaming(true);

    const userMsg: ChatMessage = { id: uid(), role: "user", content: trimmed };
    const assistantId = uid();
    assistantIdRef.current = assistantId;
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    let streamEndedCleanly = false;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          session_id: sessionIdRef.current,
          limit: 3,
        }),
        signal: controller.signal,
      });

      if (!response.body) {
        throw new Error("Response has no body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const event = parseLine(line);
          if (!event) continue;

          if (event.type === "meta") {
            setHits(event.hits ?? []);
            if (event.session_id && event.session_id.trim()) {
              sessionIdRef.current = event.session_id.trim();
            }
            continue;
          }

          if (event.type === "token") {
            const piece = event.token ?? event.content ?? "";
            if (!piece) continue;

            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, content: msg.content + piece }
                  : msg
              )
            );
            continue;
          }

          if (event.type === "error") {
            setError(event.message || "Something went wrong.");

            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? {
                    ...msg,
                    content: msg.content || event.message || "Something went wrong.",
                  }
                  : msg
              )
            );
            continue;
          }

          if (event.type === "done") {
            streamEndedCleanly = true;
            setDoneReason(event.done_reason ?? event.response_type ?? "done");
            setIsStreaming(false);
          }
        }
      }

      if (buffer.trim()) {
        const event = parseLine(buffer);
        if (event?.type === "token") {
          const piece = event.token ?? event.content ?? "";
          if (piece) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, content: msg.content + piece }
                  : msg
              )
            );
          }
        } else if (event?.type === "meta") {
          setHits(event.hits ?? []);
          if (event.session_id && event.session_id.trim()) {
            sessionIdRef.current = event.session_id.trim();
          }
        } else if (event?.type === "error") {
          setError(event.message || "Something went wrong.");
        } else if (event?.type === "done") {
          streamEndedCleanly = true;
          setDoneReason(event.done_reason ?? event.response_type ?? "done");
          setIsStreaming(false);
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        setDoneReason("aborted");
      } else {
        const message =
          err instanceof Error ? err.message : "Something went wrong.";
        setError(message);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                ...msg,
                content: msg.content || message,
              }
              : msg
          )
        );
      }
    } finally {
      if (!streamEndedCleanly) {
        setIsStreaming(false);
      }
      abortRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    assistantIdRef.current = null;
    sessionIdRef.current = makeSessionId();
    setMessages([]);
    setIsStreaming(false);
    setError(null);
    setHits([]);
    setDoneReason("");
  }, []);

  return {
    messages,
    isStreaming,
    error,
    hits,
    doneReason,
    sendMessage,
    reset,
    sessionId: sessionIdRef.current,
  };
}
