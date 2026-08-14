/**
 * app/api/chat/route.ts
 * ─────────────────────
 * Next.js App Router proxy — forwards POST body to FastAPI /query/stream
 * and pipes the NDJSON stream back to the browser without buffering.
 *
 * Hardening applied:
 *  - dynamic = "force-dynamic" prevents Next.js from ever caching this route.
 *  - AbortSignal forwarded from the browser request to FastAPI so that if the
 *    user cancels (e.g. page nav), FastAPI also stops streaming.
 *  - Backend connection errors produce a structured NDJSON error+done payload
 *    instead of a raw HTTP 502, so the frontend hook parses it cleanly.
 *  - Request-ID forwarded from FastAPI response header for log tracing.
 */

import { NextRequest, NextResponse } from "next/server";

// Force dynamic prevents Next.js from caching this streaming route.
export const dynamic = "force-dynamic";

const FASTAPI_BASE_URL =
  process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

/** Emit a minimal NDJSON error+done pair for client-parseable failures. */
function ndjsonErrorStream(message: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      controller.enqueue(
        encoder.encode(
          JSON.stringify({ type: "error", message }) + "\n"
        )
      );
      controller.enqueue(
        encoder.encode(
          JSON.stringify({ type: "done", done_reason: "proxy_error", stats: {} }) + "\n"
        )
      );
      controller.close();
    },
  });
}

export async function POST(req: NextRequest): Promise<Response> {
  // ── Parse and validate input ───────────────────────────────────────────
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return new Response(
      ndjsonErrorStream("Invalid JSON body"),
      {
        status: 400,
        headers: { "Content-Type": "application/x-ndjson" },
      }
    );
  }

  const { question, limit = 3 } = (body ?? {}) as {
    question?: string;
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

  // ── Forward to FastAPI ─────────────────────────────────────────────────
  let fastapiResponse: Response;

  try {
    fastapiResponse = await fetch(`${FASTAPI_BASE_URL}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.trim(), limit }),
      // Forward the browser's AbortSignal so FastAPI stops if client disconnects
      signal: req.signal,
      // @ts-expect-error – node-fetch duplex option required for streaming body
      duplex: "half",
    });
  } catch (err: unknown) {
    const isAbort =
      err instanceof Error && err.name === "AbortError";
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
    return new Response(
      ndjsonErrorStream("Backend returned an empty body"),
      {
        status: 502,
        headers: { "Content-Type": "application/x-ndjson" },
      }
    );
  }

  // ── Pipe the stream straight through ──────────────────────────────────
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
