import { LIMITATIONS, PROVIDERS, validateInput } from "@/lib/frame-search/model";
import { runSearch } from "@/lib/frame-search/search";

export const maxDuration = 60;
export function GET() {
  return Response.json({
    version: 1, providers: PROVIDERS, limitations: LIMITATIONS,
    dates: "YYYY-MM-DD, UTC start inclusive / end exclusive. Archive-specific orbit date boundaries and time scales apply to SSOIS.",
    limits: { radiusDegrees: [0.001, 2], dateSpanDays: 3660, orbitHypotheses: 3, productsPerQuery: 300 },
    example: { mode: "field", ra: 180, dec: 30, radius: 0.05, start: "2019-01-28", end: "2019-01-29", providers: ["ztf"] },
    orbitConvention: "Heliocentric J2000 ecliptic osculating elliptic elements: a in au, angles in degrees, epochMjd passed unchanged to archive orbit services. Supply each service's expected epoch convention; no time-scale conversion is performed.",
  });
}
export async function POST(request: Request) {
  let input;
  try {
    const reader = request.body?.getReader();
    if (!reader) throw new Error("Provide a JSON request body.");
    let text = "", bytes = 0;
    const decoder = new TextDecoder();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        bytes += value.length;
        if (bytes > 16000) throw new Error("Request exceeds 16 KB.");
        text += decoder.decode(value, { stream: true });
      }
    } finally { await reader.cancel().catch(() => {}); }
    input = validateInput(JSON.parse(text + decoder.decode()));
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : "Invalid request." }, { status: 400 });
  }
  return Response.json(await runSearch(input), { headers: { "Cache-Control": "no-store" } });
}
