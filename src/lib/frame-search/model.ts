export const PROVIDERS = [
  { id: "cadc", name: "CADC image archive", modes: ["field"], url: "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/" },
  { id: "eso", name: "ESO image archive", modes: ["field"], url: "https://archive.eso.org/tap_obs/examples" },
  { id: "ztf", name: "ZTF public science images", modes: ["field"], url: "https://irsa.ipac.caltech.edu/ibe/docs/ztf.html" },
  { id: "ssois", name: "CADC Solar System Object Image Search", modes: ["object", "orbits"], url: "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/ssois/documentation.html" },
  { id: "most", name: "IRSA MOST · ZTF", modes: ["object", "orbits"], url: "https://irsa.ipac.caltech.edu/applications/MOST/MOSTProgramInterface.html" },
] as const;
export type ProviderId = typeof PROVIDERS[number]["id"];
export type Orbit = { label: string; epochMjd: number; a: number; e: number; i: number; node: number; peri: number; meanAnomaly: number };
export type SearchInput = {
  mode: "field" | "object" | "orbits";
  start: string;
  end: string;
  providers: ProviderId[];
  ra?: number;
  dec?: number;
  radius?: number;
  object?: string;
  orbits?: Orbit[];
};
export type Frame = {
  key: string;
  provider: ProviderId;
  survey: string;
  id: string;
  instrument: string;
  startMjd: number | null;
  endMjd: number | null;
  timeScale: "UTC" | "archive-defined";
  exposureSeconds: number | null;
  filter: string | null;
  ra: number | null;
  dec: number | null;
  positionKind: "image center" | "predicted target";
  url: string | null;
  linkKind: "FITS" | "DataLink" | "archive product";
  match: string;
  hypotheses: string[];
  calibrationLevel: number | null;
  raw: Record<string, string>;
};
export type Check = {
  provider: ProviderId;
  hypothesis: string;
  status: "ok" | "limited" | "error";
  count: number;
  queryUrl: string;
  message: string;
};
export type SearchResult = {
  version: 1;
  queriedAt: string;
  input: SearchInput;
  frames: Frame[];
  checks: Check[];
  unqueried: { name: string; reason: string }[];
  limitations: string[];
};
export const ROW_LIMIT = 300;
export const LIMITATIONS = [
  "This is a search of connected archive indexes, not a complete census of public surveys. The planning inventory contains additional datasets that are not connected.",
  "A footprint or predicted-position match is a candidate for inspection, not evidence of detection. Chip gaps, masks, depth, trailing, and data quality are not evaluated.",
  "Metadata and links are public; file availability and access rights are not independently verified. Some products are stacks or mosaics rather than single exposures.",
  "Orbit searches use each archive's ephemeris service and sampling. Fast or nearby targets, parallax, and uncertainty require follow-up validation. Orbit samples are not a probability distribution.",
  "SSOIS timestamps retain the archive's time convention (including TAI for some products); they are not converted to UTC. Inspect native metadata before precision work.",
];
export function mjd(date: string): number {
  return Date.parse(`${date}T00:00:00Z`) / 86400000 + 40587;
}
function number(value: unknown, name: string, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) throw new Error(`${name} must be a number from ${min} to ${max}.`);
  return value;
}
export function validateInput(value: unknown): SearchInput {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Provide a JSON search object.");
  const x = value as Record<string, unknown>;
  if (x.mode !== "field" && x.mode !== "object" && x.mode !== "orbits") throw new Error("Choose field, object, or orbits.");
  const dates = ["start", "end"].map(key => {
    const v = x[key];
    if (typeof v !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(v) || !Number.isFinite(mjd(v)) || new Date(`${v}T00:00:00Z`).toISOString().slice(0, 10) !== v) throw new Error(`${key} must be a valid YYYY-MM-DD date.`);
    if (v < "1900-01-01" || v > "2100-12-31") throw new Error("Dates must be between 1900 and 2100.");
    return v;
  });
  const span = mjd(dates[1]) - mjd(dates[0]);
  if (span <= 0 || span > 3660) throw new Error("Use an end date after the start, spanning at most 10 years (3,660 days).");
  const available = PROVIDERS.filter(p => (p.modes as readonly string[]).includes(x.mode as string)).map(p => p.id);
  const requested = x.providers ?? available;
  if (!Array.isArray(requested) || !requested.length || requested.some(p => !available.includes(p))) throw new Error("Select at least one archive supported by this input mode.");
  const result: SearchInput = { mode: x.mode, start: dates[0], end: dates[1], providers: [...new Set(requested)] };
  if (x.mode === "field") {
    result.ra = number(x.ra, "RA (degrees)", 0, 360);
    if (result.ra === 360) result.ra = 0;
    result.dec = number(x.dec, "Dec (degrees)", -90, 90);
    result.radius = number(x.radius, "Radius (degrees)", 0.001, 2);
  } else if (x.mode === "object") {
    if (typeof x.object !== "string" || !x.object.trim() || x.object.length > 100 || /[\x00-\x1f]/.test(x.object)) throw new Error("Enter a Solar System object name or designation (up to 100 characters).");
    result.object = x.object.trim();
  } else {
    if (!Array.isArray(x.orbits) || !x.orbits.length || x.orbits.length > 3) throw new Error("Provide 1–3 candidate orbits.");
    result.orbits = x.orbits.map((entry, index) => {
      if (!entry || typeof entry !== "object") throw new Error(`Orbit ${index + 1} must be an object.`);
      const o = entry as Record<string, unknown>;
      if (typeof o.label !== "string" || !o.label.trim() || o.label.length > 60) throw new Error("Each orbit needs a unique label (up to 60 characters).");
      return { label: o.label.trim(), epochMjd: number(o.epochMjd, "Epoch MJD", 15020, 88069), a: number(o.a, "Semimajor axis (au)", 0.001, 100000), e: number(o.e, "Eccentricity", 0, 0.999999), i: number(o.i, "Inclination", 0, 180), node: number(o.node, "Ascending node", 0, 360), peri: number(o.peri, "Argument of perihelion", 0, 360), meanAnomaly: number(o.meanAnomaly, "Mean anomaly", 0, 360) };
    });
    if (new Set(result.orbits.map(o => o.label)).size !== result.orbits.length) throw new Error("Orbit labels must be unique.");
  }
  return result;
}
export function safeUrl(value: string | undefined): string | null {
  if (!value) return null;
  try { const u = new URL(value); return ["https:", "http:"].includes(u.protocol) && !u.username && !u.password ? u.href : null; } catch { return null; }
}
