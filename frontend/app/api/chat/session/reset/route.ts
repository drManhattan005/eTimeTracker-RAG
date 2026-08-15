import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const FASTAPI_BASE_URL =
    process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest): Promise<Response> {
    let body: unknown;

    try {
        body = await req.json();
    } catch {
        return Response.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
    }

    const { session_id } = (body ?? {}) as { session_id?: string };

    if (!session_id || typeof session_id !== "string" || session_id.trim() === "") {
        return Response.json(
            { ok: false, error: "session_id is required and must be a non-empty string" },
            { status: 422 }
        );
    }

    try {
        const fastapiResponse = await fetch(`${FASTAPI_BASE_URL}/session/reset`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: session_id.trim() }),
            signal: req.signal,
            cache: "no-store",
        });

        const data = await fastapiResponse.json().catch(() => ({ ok: fastapiResponse.ok }));

        return Response.json(data, { status: fastapiResponse.status });
    } catch {
        return Response.json(
            { ok: false, error: "Backend unavailable. Is FastAPI running?" },
            { status: 502 }
        );
    }
}
