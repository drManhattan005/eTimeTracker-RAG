import { useCallback, useMemo, useRef, useState } from "react";

export type ChatRole = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  tokenSnapshot?: SessionTokenBudget;
};

export type RetrievedHit = {
  id: string;
  chunk_id: string;
  score: number;
  fused_score?: number;
  dense_rank?: number | null;
  bm25_rank?: number | null;
  section?: string | Record<string, unknown>;
  intent_sphere?: string;
  chunk_type?: string;
  plan_tier?: string;
  text_preview?: string;
};

export type StreamMeta = {
  session_id: string;
  question: string;
  effective_question: string;
  retrieval_route: string;
  model: string;
  hits: RetrievedHit[];
  session_tokens_used?: number;
  session_tokens_total?: number;
  session_tokens_remaining?: number;
  session_blocked?: boolean;
};

export type SessionTokenBudget = {
  session_tokens_used: number;
  session_tokens_total: number;
  session_tokens_remaining: number;
  session_blocked: boolean;
};

const HISTORY_TURNS = 3;
const DEFAULT_API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function createId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function trimMessagesForUi(messages: ChatMessage[]) {
  return messages;
}

function trimMessagesForServer(messages: ChatMessage[]) {
  const maxMessages = HISTORY_TURNS * 2;
  return messages.slice(-maxMessages);
}

function toSessionTokenBudget(event: Partial<SessionTokenBudget>) {
  if (
    typeof event.session_tokens_used !== "number" ||
    typeof event.session_tokens_total !== "number" ||
    typeof event.session_tokens_remaining !== "number"
  ) {
    return null;
  }

  return {
    session_tokens_used: event.session_tokens_used,
    session_tokens_total: event.session_tokens_total,
    session_tokens_remaining: event.session_tokens_remaining,
    session_blocked: Boolean(event.session_blocked),
  };
}

export function useChatStream() {
  const apiBaseUrl = DEFAULT_API_BASE_URL;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [meta, setMeta] = useState<StreamMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionTokenState, setSessionTokenState] =
    useState<SessionTokenBudget | null>(null);
  const [sessionBlocked, setSessionBlocked] = useState(false);
  const [sessionId, setSessionId] = useState(() => createId());

  const sessionIdRef = useRef<string>(sessionId);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (content: string) => {
      const question = content.trim();
      if (!question || isStreaming || sessionBlocked) return;

      setError(null);

      const userMessage: ChatMessage = {
        id: createId(),
        role: "user",
        content: question,
        tokenSnapshot: sessionTokenState ?? undefined,
      };

      const assistantMessageId = createId();
      const historyForServer = trimMessagesForServer(messages);

      setMessages((prev) => [
        ...trimMessagesForUi([...prev, userMessage]),
        { id: assistantMessageId, role: "assistant", content: "" },
      ]);
      setIsStreaming(true);
      setMeta(null);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await fetch(`${apiBaseUrl}/query/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          signal: controller.signal,
          body: JSON.stringify({
            question,
            session_id: sessionIdRef.current,
            limit: 12,
            history: historyForServer,
          }),
        });

        if (!response.ok || !response.body) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finalAnswer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line) continue;

            const event = JSON.parse(line) as
              | {
                type: "meta";
                hits: RetrievedHit[];
                session_id: string;
                question: string;
                effective_question: string;
                retrieval_route: string;
                model: string;
                session_tokens_used?: number;
                session_tokens_total?: number;
                session_tokens_remaining?: number;
                session_blocked?: boolean;
              }
              | { type: "token"; text?: string; token?: string }
              | {
                type: "done";
                answer: string;
                session_tokens_used?: number;
                session_tokens_total?: number;
                session_tokens_remaining?: number;
                session_blocked?: boolean;
              }
              | {
                type: "error";
                message?: string;
                session_tokens_used?: number;
                session_tokens_total?: number;
                session_tokens_remaining?: number;
                session_blocked?: boolean;
              };

            if (event.type === "meta") {
              setMeta({
                session_id: event.session_id,
                question: event.question,
                effective_question: event.effective_question,
                retrieval_route: event.retrieval_route,
                model: event.model,
                hits: event.hits ?? [],
                session_tokens_used: event.session_tokens_used,
                session_tokens_total: event.session_tokens_total,
                session_tokens_remaining: event.session_tokens_remaining,
                session_blocked: event.session_blocked,
              });
              continue;
            }

            if (event.type === "token") {
              finalAnswer += event.text ?? event.token ?? "";
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, content: finalAnswer }
                    : msg
                )
              );
              continue;
            }

            if (event.type === "error") {
              const message = event.message ?? "Something went wrong";
              finalAnswer = message;
              setError(message);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, content: finalAnswer }
                    : msg
                )
              );
              continue;
            }

            if (event.type === "done") {
              finalAnswer = event.answer;
              const completedBudget = toSessionTokenBudget(event);
              if (completedBudget) {
                setSessionTokenState(completedBudget);
                setSessionBlocked(completedBudget.session_blocked);
              }
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, content: finalAnswer }
                    : msg.id === userMessage.id && completedBudget
                      ? { ...msg, tokenSnapshot: completedBudget }
                    : msg
                )
              );
            }
          }
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Something went wrong";
        setError(message);
        setMessages((prev) =>
          prev.filter((msg) => msg.id !== assistantMessageId)
        );
      } finally {
        abortRef.current = null;
        setIsStreaming(false);
      }
    },
    [apiBaseUrl, isStreaming, messages, sessionBlocked, sessionTokenState]
  );

  const reset = useCallback(async () => {
    abortRef.current?.abort();
    abortRef.current = null;

    const oldSessionId = sessionIdRef.current;
    const nextSessionId = createId();
    sessionIdRef.current = nextSessionId;
    setSessionId(nextSessionId);

    setMessages([]);
    setMeta(null);
    setError(null);
    setIsStreaming(false);
    setSessionTokenState(null);
    setSessionBlocked(false);

    try {
      await fetch(`${apiBaseUrl}/session/reset`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: oldSessionId,
        }),
      });
    } catch {
      // noop
    }
  }, [apiBaseUrl]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  }, []);

  return useMemo(
    () => ({
      messages,
      isStreaming,
      meta,
      error,
      sessionTokenState,
      sessionBlocked,
      sessionId,
      sendMessage,
      reset,
      stop,
    }),
    [
      messages,
      isStreaming,
      meta,
      error,
      sessionTokenState,
      sessionBlocked,
      sessionId,
      sendMessage,
      reset,
      stop,
    ]
  );
}
