/**
 * app/api/chat/route.ts
 * ─────────────────────
 * Next.js App Router proxy — forwards POST body to FastAPI /query/stream
 * and pipes the NDJSON stream back to the browser without buffering.
 */

import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const FASTAPI_BASE_URL =
  process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

function ndjsonErrorStream(message: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      controller.enqueue(
        encoder.encode(JSON.stringify({ type: "error", message }) + "\n")
      );
      controller.enqueue(
        encoder.encode(
          JSON.stringify({
            type: "done",
            done_reason: "proxy_error",
            stats: {},
          }) + "\n"
        )
      );
      controller.close();
    },
  });
}

export async function POST(req: NextRequest): Promise<Response> {
  let body: unknown;

  try {
    body = await req.json();
  } catch {
    return new Response(ndjsonErrorStream("Invalid JSON body"), {
      status: 400,
      headers: { "Content-Type": "application/x-ndjson" },
    });
  }

  const { question, session_id, limit = 3 } = (body ?? {}) as {
    question?: string;
    session_id?: string;
    limit?: number;
  };

  if (!question || typeof question !== "string" || question.trim() === "") {
    return new Response(
      ndjsonErrorStream("question is required and must be a non-empty string"),
      {
        status: 422,
        headers: { "Content-Type": "application/x-ndjson" },
      }
    );
  }

  if (
    !session_id ||
    typeof session_id !== "string" ||
    session_id.trim() === ""
  ) {
    return new Response(
      ndjsonErrorStream("session_id is required and must be a non-empty string"),
      {
        status: 422,
        headers: { "Content-Type": "application/x-ndjson" },
      }
    );
  }

  let fastapiResponse: Response;

  try {
    fastapiResponse = await fetch(`${FASTAPI_BASE_URL}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question.trim(),
        session_id: session_id.trim(),
        limit,
      }),
      signal: req.signal,
      // @ts-expect-error – required for streaming in Node runtime
      duplex: "half",
    });
  } catch (err: unknown) {
    const isAbort = err instanceof Error && err.name === "AbortError";
    const message = isAbort
      ? "Request cancelled"
      : "Backend unavailable. Is the FastAPI server running?";
    console.error("[chat/route] FastAPI fetch failed:", err);
    return new Response(ndjsonErrorStream(message), {
      status: isAbort ? 499 : 502,
      headers: { "Content-Type": "application/x-ndjson" },
    });
  }

  if (!fastapiResponse.ok) {
    const text = await fastapiResponse.text().catch(() => "");
    console.error(
      "[chat/route] FastAPI non-2xx:",
      fastapiResponse.status,
      text.slice(0, 200)
    );
    return new Response(
      ndjsonErrorStream(`Backend error ${fastapiResponse.status}`),
      {
        status: fastapiResponse.status,
        headers: { "Content-Type": "application/x-ndjson" },
      }
    );
  }

  if (!fastapiResponse.body) {
    return new Response(ndjsonErrorStream("Backend returned an empty body"), {
      status: 502,
      headers: { "Content-Type": "application/x-ndjson" },
    });
  }

  const reqId = fastapiResponse.headers.get("x-request-id") ?? "";

  return new Response(fastapiResponse.body, {
    status: 200,
    headers: {
      "Content-Type": "application/x-ndjson",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
      ...(reqId ? { "X-Request-Id": reqId } : {}),
    },
  });
}
