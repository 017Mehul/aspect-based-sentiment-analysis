import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  let payload: { text?: unknown; aspects?: unknown };
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ error: "Send a valid JSON request." }, { status: 400 });
  }

  const text = typeof payload.text === "string" ? payload.text.trim() : "";
  if (!text) {
    return NextResponse.json({ error: "Enter a review before analyzing it." }, { status: 400 });
  }
  if (text.length > 4000) {
    return NextResponse.json({ error: "Keep the review under 4,000 characters." }, { status: 400 });
  }

  const aspects = Array.isArray(payload.aspects)
    ? payload.aspects.filter((item): item is string => typeof item === "string").map((item) => item.trim()).filter(Boolean).slice(0, 20)
    : [];

  const backend = process.env.ABSA_API_URL?.trim().replace(/\/$/, "");
  if (!backend) {
    return NextResponse.json(
      {
        error: "The Python inference service is not connected yet. Set ABSA_API_URL to the deployed FastAPI service URL, then restart/redeploy the web app.",
        code: "ABSA_BACKEND_NOT_CONFIGURED",
      },
      { status: 503 },
    );
  }

  try {
    const upstream = await fetch(`${backend}/analyze`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, aspects: aspects.length ? aspects : null }),
      signal: AbortSignal.timeout(50_000),
      cache: "no-store",
    });
    const result = await upstream.json().catch(() => ({ error: "The inference service returned an unreadable response." }));
    return NextResponse.json(result, { status: upstream.status, headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json(
      { error: "Could not reach the Python inference service. Check ABSA_API_URL and confirm the API is running." },
      { status: 502 },
    );
  }
}
