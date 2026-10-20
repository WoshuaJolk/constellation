import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateInput, safeUrl, type SearchInput, ROW_LIMIT } from "../src/lib/frame-search/model";
import { buildJobs, parseFrames, runSearch, ztfUrl } from "../src/lib/frame-search/search";
import { delimited, ipac } from "../src/lib/frame-search/parsers";
import { framesCsv } from "../src/lib/frame-search/export";
import { GET, POST } from "../src/app/api/frames/route";

const fixture = (name: string) => readFileSync(new URL(`fixtures/${name}`, import.meta.url), "utf8");
const field: SearchInput = { mode: "field", ra: 180, dec: 30, radius: 0.05, start: "2019-01-28", end: "2019-01-29", providers: ["ztf"] };
const orbit = { label: "a", epochMjd: 55200, a: 2.74, e: 0.08, i: 6.98, node: 43, peri: 120, meanAnomaly: 206 };
test("validation rejects malformed dates, injection, unsupported providers, and unbounded requests", () => {
  for (const change of [{ start: "2025-02-30" }, { ra: "0); DROP TABLE" }, { radius: 10 }, { end: field.start }, { providers: ["ssois"] }, { providers: [] }, { ra: NaN }, { start: "1900-01-01" }]) {
    assert.throws(() => validateInput({ ...field, ...change }));
  }
  assert.equal(validateInput({ ...field, ra: 360 }).ra, 0);
  assert.equal(validateInput({ ...field, dec: -90 }).dec, -90);
  assert.throws(() => validateInput({ ...field, mode: "orbits", providers: ["ssois"], orbits: [orbit, orbit] }), /unique/);
  assert.throws(() => validateInput({ ...field, mode: "orbits", providers: ["ssois"], orbits: [{ ...orbit, e: 1.2 }] }));
});
test("queries preserve spatial footprint and half-open UTC time bounds, including RA wrap", () => {
  const jobs = buildJobs({ ...field, ra: 359.99, providers: ["cadc", "eso", "ztf"] });
  const query = new URL(jobs[0].url).searchParams.get("QUERY")!;
  assert.match(query, /INTERSECTS\(s_region,CIRCLE\('ICRS',359.99,30,0.05\)\)=1/);
  assert.match(query, /t_min < 58512 AND t_max >= 58511/);
  assert.equal(new URL(jobs[2].url).searchParams.get("WHERE"), "obsjd>=2458511.5 AND obsjd<2458512.5");
});
test("real ZTF fixed-width data preserves IDs, nulls, timestamps, and retrieval paths", () => {
  const job = buildJobs(field)[0];
  const frames = parseFrames(job, fixture("ztf.tbl"), field);
  assert.equal(frames.length, 6);
  assert.equal(frames[0].id, "757447121115");
  assert.equal(frames[0].exposureSeconds, 30);
  assert.equal(frames[0].filter, "zg");
  assert.match(frames[0].url!, /2019\/0128\/447118\/ztf_20190128447118_000671_zg_c03_o_q4_sciimg.fits$/);
  assert.equal(frames[0].raw.infobits, "0");
  assert.equal(parseFrames(job, fixture("ztf.tbl"), { ...field, start: field.end, end: "2019-01-30" }).length, 0);
  assert.equal(ztfUrl({ filefracday: "../etc/passwd" }), null);
});
test("table errors never masquerade as empty coverage", () => {
  assert.throws(() => ipac(fixture("ztf.tbl").replace("RowsRetrieved = 6", "RowsRetrieved = 7"), true), /row count/);
  assert.throws(() => ipac(fixture("ztf.tbl").replace("'OK'", "'ERROR'"), true), /successful/);
  assert.throws(() => parseFrames(buildJobs(field)[0], "<html>service unavailable</html>", field), /columns/);
  assert.throws(() => delimited('a,b\n"truncated'), /incomplete/);
  assert.throws(() => delimited("a,b\n1,2,3"), /malformed/);
  assert.deepEqual(delimited('a,b\n"one,two","three\nfour"\n'), [{ a: "one,two", b: "three\nfour" }]);
});
test("real ObsCore products preserve stack time ranges and DataLink retrieval type", () => {
  const input = { ...field, providers: ["cadc"] } as SearchInput;
  const frames = parseFrames(buildJobs(input)[0], fixture("cadc.csv"), input);
  assert.equal(frames.length, 3);
  assert.equal(frames[0].survey, "JCMT");
  assert.equal(frames[0].linkKind, "DataLink");
  assert.equal(frames[1].calibrationLevel, 3);
  assert.equal(frames[1].exposureSeconds, null);
  assert.ok(frames[1].endMjd! > 59000);
});
test("SSOIS records retain native time scales and merge only identical hypothesis products", async () => {
  const input = validateInput({ ...field, mode: "orbits", providers: ["ssois"], orbits: [orbit, { ...orbit, label: "b" }] });
  const result = await runSearch(input, async () => fixture("ssois.tsv"));
  assert.equal(result.frames.length, 3);
  assert.deepEqual(result.frames[0].hypotheses, ["a", "b"]);
  assert.equal(result.frames[0].timeScale, "archive-defined");
  assert.equal(result.frames[0].positionKind, "predicted target");
  assert.equal(result.checks.length, 2);
  const query = new URL(buildJobs(input)[0].url).searchParams;
  assert.equal(query.get("epoch"), "55200");
  assert.equal(query.get("longi"), "43");
});
test("one archive failure retains the other results and an explicit error receipt", async () => {
  const result = await runSearch({ ...field, providers: ["cadc", "ztf"] }, async url => {
    if (url.includes("argus")) throw new Error("Archive returned HTTP 503.");
    return fixture("ztf.tbl");
  });
  assert.equal(result.frames.length, 6);
  assert.equal(result.checks[0].status, "error");
  assert.match(result.checks[0].message, /Coverage is unknown/);
  assert.equal(result.checks[1].status, "ok");
});
test("row limits are explicit and successful empty MOST tables remain distinct from errors", async () => {
  const table = fixture("ssois.tsv").trim().split("\n");
  const result = await runSearch(validateInput({ ...field, mode: "object", object: "Sedna", providers: ["ssois"] }), async () => `${table[0]}\n${Array.from({ length: ROW_LIMIT + 1 }, (_, i) => table[1].replace("890045p", `image${i}`)).join("\n")}`);
  assert.equal(result.checks[0].status, "limited");
  assert.equal(result.frames.length, ROW_LIMIT);
  const empty = await runSearch(validateInput({ ...field, mode: "object", object: "Sedna", providers: ["most"] }), async () => fixture("most-empty.tbl"));
  assert.equal(empty.checks[0].status, "ok");
  assert.equal(empty.frames.length, 0);
});
test("manifest export escapes remote spreadsheet formulas and rejects unsafe links", async () => {
  const result = await runSearch(field, async () => fixture("ztf.tbl"));
  result.frames[0].survey = '=HYPERLINK("bad")';
  assert.match(framesCsv(result), /"'=HYPERLINK\(""bad""\)"/);
  assert.equal(safeUrl("javascript:alert(1)"), null);
  assert.equal(safeUrl("https://user:password@example.org"), null);
  assert.equal(safeUrl("https://example.org/image.fits"), "https://example.org/image.fits");
});
test("live MOST orbit fixture produces ZTF request links without inventing exposure metadata", () => {
  const input = validateInput({ mode: "orbits", start: "2019-01-01", end: "2019-02-01", providers: ["most"], orbits: [orbit] });
  const frames = parseFrames(buildJobs(input)[0], fixture("most-orbit.tbl"), input);
  assert.equal(frames.length, 17);
  assert.ok(frames.every(f => f.url?.endsWith("_sciimg.fits")));
  assert.ok(frames.every(f => f.exposureSeconds === null));
  assert.ok(frames.every(f => f.raw.match === "1"));
  assert.ok(frames.every(f => f.timeScale === "UTC"));
});
test("API describes capabilities and rejects malformed or oversized requests before archive work", async () => {
  const capabilities = await GET().json();
  assert.equal(capabilities.providers.length, 5);
  for (const body of ["{broken", JSON.stringify({ ...field, radius: -1 }), " ".repeat(16001)]) {
    const response = await POST(new Request("http://localhost/api/frames", { method: "POST", body }));
    assert.equal(response.status, 400);
    assert.ok((await response.json()).error);
  }
});
