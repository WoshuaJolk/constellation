import { DATASETS } from "../constellation";
import { LIMITATIONS, PROVIDERS, ROW_LIMIT, mjd, safeUrl, type Check, type Frame, type Orbit, type ProviderId, type SearchInput, type SearchResult } from "./model";
import { delimited, ipac, numeric, requireColumns, type Row } from "./parsers";

export type Job = { provider: ProviderId; hypothesis: string; url: string; note: string };
const tapColumns = "obs_collection,obs_id,obs_publisher_did,instrument_name,t_min,t_max,t_exptime,s_ra,s_dec,s_region,em_min,em_max,access_url,access_format,calib_level";
function url(base: string, params: Record<string, string | number>): string {
  return `${base}?${new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]))}`;
}
export function buildJobs(input: SearchInput): Job[] {
  const jobs: Job[] = [];
  for (const provider of input.providers) {
    if (provider === "cadc" || provider === "eso") {
      const query = `SELECT TOP ${ROW_LIMIT + 1} ${tapColumns} FROM ivoa.ObsCore WHERE dataproduct_type='image' AND INTERSECTS(s_region,CIRCLE('ICRS',${input.ra},${input.dec},${input.radius}))=1 AND t_min < ${mjd(input.end)} AND t_max >= ${mjd(input.start)}`;
      jobs.push({ provider, hypothesis: "field", url: url(provider === "cadc" ? "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync" : "https://archive.eso.org/tap_obs/sync", { REQUEST: "doQuery", LANG: "ADQL", FORMAT: "csv", MAXREC: ROW_LIMIT + 1, QUERY: query }), note: "Circular field intersects the archive footprint; product time interval overlaps the requested UTC dates. Stacks may span dates outside the interval." });
    } else if (provider === "ztf") {
      jobs.push({ provider, hypothesis: "field", url: url("https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci", { POS: `${input.ra},${input.dec}`, SIZE: input.radius! * 2, INTERSECT: "OVERLAPS", WHERE: `obsjd>=${mjd(input.start) + 2400000.5} AND obsjd<${mjd(input.end) + 2400000.5}`, CT: "ipac_table" }), note: "ZTF uses a bounding square of side 2 × radius; corner-only matches can lie outside the requested circle. These are candidate exposures." });
    } else {
      const hypotheses: (Orbit | undefined)[] = input.mode === "orbits" ? input.orbits! : [undefined];
      for (const orbit of hypotheses) {
        const hypothesis = orbit?.label ?? input.object!;
        if (provider === "ssois") {
          const params: Record<string, string | number> = { lang: "en", format: "tsv", epoch1: input.start, epoch2: input.end, eellipse: 0, eunits: "arcseconds", extres: "no", xyres: "no" };
          Object.assign(params, orbit ? { search: "orbel", epoch: orbit.epochMjd, a: orbit.a, e: orbit.e, i: orbit.i, longi: orbit.node, peri: orbit.peri, anom: orbit.meanAnomaly } : { search: "bynameHorizons", object: input.object });
          jobs.push({ provider, hypothesis, url: url("https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/ssos/ssosclf.pl", params), note: "SSOIS predicted-position matches across its instrument index. Native time scales and archive ephemeris interpolation apply; detector gaps are not checked." });
        } else {
          const params: Record<string, string | number> = { catalog: "ztf", obs_begin: input.start.replaceAll("-", " "), obs_end: input.end.replaceAll("-", " "), output_mode: "Brief", ephem_step: 0.25 };
          Object.assign(params, orbit ? { input_type: "manual_input", obj_type: "Asteroid", body_designation: orbit.label, epoch: orbit.epochMjd, semimajor_axis: orbit.a, eccentricity: orbit.e, inclination: orbit.i, ascend_node: orbit.node, arg_perihelion: orbit.peri, mean_anomaly: orbit.meanAnomaly } : { input_type: "name_input", obj_name: input.object });
          jobs.push({ provider, hypothesis, url: url("https://irsa.ipac.caltech.edu/cgi-bin/MOST/nph-most", params), note: "MOST predicted-position matches to ZTF. Ephemeris step is 0.25 days; verify fast or nearby targets independently." });
        }
      }
    }
  }
  return jobs;
}
export function ztfUrl(row: Row): string | null {
  const frac = row.filefracday, field = row.field, ccd = row.ccdid, q = row.qid, filter = row.filtercode;
  if (!/^\d{14}$/.test(frac ?? "") || !/^\d{1,6}$/.test(field ?? "") || !/^\d{1,2}$/.test(ccd ?? "") || Number(ccd) < 1 || Number(ccd) > 16 || !/^[1-4]$/.test(q ?? "") || !["zg", "zr", "zi"].includes(filter)) return null;
  return `https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/${frac.slice(0, 4)}/${frac.slice(4, 8)}/${frac.slice(8)}/ztf_${frac}_${field.padStart(6, "0")}_${filter}_c${ccd.padStart(2, "0")}_o_q${q}_sciimg.fits`;
}
export function parseFrames(job: Job, text: string, input: SearchInput): Frame[] {
  const base = { provider: job.provider, hypotheses: [job.hypothesis], calibrationLevel: null, endMjd: null };
  if (job.provider === "cadc" || job.provider === "eso") {
    requireColumns(text, tapColumns.split(","));
    return delimited(text).map(r => ({
      ...base, key: `${job.provider}:${r.obs_publisher_did || r.obs_id}`, survey: r.obs_collection || job.provider.toUpperCase(), id: r.obs_id, instrument: r.instrument_name,
      startMjd: numeric(r.t_min), endMjd: numeric(r.t_max), timeScale: "UTC", exposureSeconds: numeric(r.t_exptime), filter: null,
      ra: numeric(r.s_ra), dec: numeric(r.s_dec), positionKind: "image center", url: safeUrl(r.access_url),
      linkKind: /datalink/i.test(r.access_format) ? "DataLink" : /fits/i.test(r.access_format) ? "FITS" : "archive product",
      match: "footprint intersection", calibrationLevel: numeric(r.calib_level), raw: r,
    }));
  }
  if (job.provider === "ssois") {
    requireColumns(text, ["Image", "MJD", "Telescope/Instrument", "Datalink"]);
    return delimited(text, "\t").map(r => ({
      ...base, key: `ssois:${r["Telescope/Instrument"]}:${r.Image}`, survey: r["Telescope/Instrument"], id: r.Image, instrument: r["Telescope/Instrument"],
      startMjd: numeric(r.MJD), timeScale: "archive-defined", exposureSeconds: numeric(r.Exptime), filter: r.Filter || null,
      ra: numeric(r.Object_RA), dec: numeric(r.Object_Dec), positionKind: "predicted target", url: safeUrl(r.Datalink), linkKind: "archive product", match: "predicted position", raw: r,
    }));
  }
  requireColumns(text, ["pid", "obsjd", "filefracday", "field", "ccdid", "qid", "filtercode", ...(job.provider === "most" ? ["match"] : [])]);
  return ipac(text, job.provider === "ztf").filter(r => {
    const date = numeric(r.obsjd);
    if (date === null || !r.pid) throw new Error("IRSA returned a product without a valid ID or observation time.");
    return date >= mjd(input.start) + 2400000.5 && date < mjd(input.end) + 2400000.5 && (job.provider !== "most" || r.match === "1");
  }).map(r => ({
    ...base, key: `ztf:${r.pid}`, survey: "ZTF", id: r.pid, instrument: "ZTF", startMjd: numeric(r.obsjd)! - 2400000.5, timeScale: "UTC",
    exposureSeconds: numeric(r.exptime), filter: r.filtercode, ra: numeric(r.ra), dec: numeric(r.dec), positionKind: "image center", url: ztfUrl(r), linkKind: "FITS",
    match: job.provider === "ztf" ? "bounding-square intersection" : "predicted position", raw: r,
  }));
}
export async function fetchTable(url: string): Promise<string> {
  const response = await fetch(url, { signal: AbortSignal.timeout(22000), cache: "no-store", headers: { Accept: "text/plain, text/csv, */*" } });
  if (!response.body) throw new Error("Archive returned no response body.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let size = 0, text = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 6_000_000) throw new Error("Archive response exceeded 6 MB. Narrow the field or dates and retry.");
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
    if (!response.ok) {
      if (response.status === 400 && /Multiple matches|Object Name Error/i.test(text)) throw new Error("The archive could not uniquely resolve the object. Try a numbered designation; MOST also accepts a NAIF ID with the :NID suffix.");
      throw new Error(`Archive returned HTTP ${response.status}.`);
    }
    return text;
  } finally { await reader.cancel().catch(() => {}); }
}
export async function runSearch(input: SearchInput, fetcher = fetchTable): Promise<SearchResult> {
  const jobs = buildJobs(input);
  const outcomes: { frames: Frame[]; check: Check }[] = new Array(jobs.length);
  let cursor = 0;
  async function worker() {
    while (cursor < jobs.length) {
      const index = cursor++, job = jobs[index];
      try {
        const frames = parseFrames(job, await fetcher(job.url), input);
        const limited = frames.length > ROW_LIMIT;
        outcomes[index] = { frames: frames.slice(0, ROW_LIMIT), check: { provider: job.provider, hypothesis: job.hypothesis, status: limited ? "limited" : "ok", count: Math.min(frames.length, ROW_LIMIT), queryUrl: job.url, message: `${job.note}${limited ? ` Only the first ${ROW_LIMIT} products are included; narrow your search.` : " Query returned successfully; archive completeness is not guaranteed."}` } };
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown archive error.";
        outcomes[index] = { frames: [], check: { provider: job.provider, hypothesis: job.hypothesis, status: "error", count: 0, queryUrl: job.url, message: `${/timeout|abort/i.test(message) ? "Archive request timed out after 22 seconds." : message} Coverage is unknown. Retry or narrow the search.` } };
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(3, jobs.length) }, worker));
  const unique = new Map<string, Frame>();
  for (const frame of outcomes.flatMap(o => o.frames)) {
    const existing = unique.get(frame.key);
    if (existing) existing.hypotheses = [...new Set([...existing.hypotheses, ...frame.hypotheses])];
    else unique.set(frame.key, frame);
  }
  return {
    version: 1, queriedAt: new Date().toISOString(), input,
    frames: [...unique.values()].sort((a, b) => (a.startMjd ?? Infinity) - (b.startMjd ?? Infinity) || a.key.localeCompare(b.key)),
    checks: outcomes.map(o => o.check),
    unqueried: PROVIDERS.filter(p => !input.providers.includes(p.id)).map(p => ({ name: p.name, reason: (p.modes as readonly string[]).includes(input.mode) ? "Not selected." : "Not connected for this input mode." })),
    limitations: [...LIMITATIONS, `The planning inventory lists ${DATASETS.length} datasets. Adapter coverage varies by collection and epoch; inclusion in that inventory does not mean a dataset was searched.`, "Results are capped at 300 products per archive per hypothesis. Overlapping archives may return duplicate products under different identifiers."],
  };
}
