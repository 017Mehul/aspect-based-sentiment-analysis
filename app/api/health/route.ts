import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function GET() {
  const backend = process.env.ABSA_API_URL?.trim().replace(/\/$/, "");
  if (!backend) {
    return NextResponse.json(
      { status: "unconfigured", error: "ABSA_API_URL is not configured." },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }

  try {
    // Render's free instance may sleep when idle. Calling /health on page load
    // wakes it before the user submits an analysis request.
    const response = await fetch(`${backend}/health`, {
      method: "GET",
      cache: "no-store",
      signal: AbortSignal.timeout(55_000),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      return NextResponse.json(
        { status: "unavailable", error: "The Python service returned an unhealthy response.", upstreamStatus: response.status },
        { status: 503, headers: { "Cache-Control": "no-store" } },
      );
    }
    return NextResponse.json({ status: "ready", backend: data }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json(
      { status: "warming", error: "The Python service is starting up. Please retry shortly." },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
