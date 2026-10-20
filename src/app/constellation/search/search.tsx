"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { PROVIDERS, type ProviderId, type SearchInput, type SearchResult } from "@/lib/frame-search/model";
import { framesCsv } from "@/lib/frame-search/export";

const EXAMPLE_ORBITS = JSON.stringify([{ label: "candidate-1", epochMjd: 55200, a: 2.7413806, e: 0.0855139, i: 6.98707, node: 43.05408, peri: 120.67443, meanAnomaly: 206.88213 }], null, 2);
const inputClass = "mt-2 w-full rounded border border-[var(--border)] bg-white px-3 py-2.5 text-sm outline-offset-4 focus:outline-[var(--known)]";
const buttonClass = "rounded border border-[var(--border)] px-3 py-2 text-xs hover:bg-[var(--stripe)] disabled:opacity-50";
function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function FrameSearch() {
  const [mode, setMode] = useState<SearchInput["mode"]>("field");
  const [selected, setSelected] = useState<ProviderId[]>(["cadc", "eso", "ztf"]);
  const [ra, setRa] = useState("180"), [dec, setDec] = useState("30"), [radius, setRadius] = useState("0.05");
  const [start, setStart] = useState("2019-01-28"), [end, setEnd] = useState("2019-01-29");
  const [object, setObject] = useState("2006 WG206"), [orbits, setOrbits] = useState(EXAMPLE_ORBITS);
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [survey, setSurvey] = useState("all");
  const available = PROVIDERS.filter(p => (p.modes as readonly string[]).includes(mode));
  const surveys = result ? [...new Set(result.frames.map(f => f.survey))].sort() : [];
  const frames = result?.frames.filter(f => survey === "all" || f.survey === survey) ?? [];
  function chooseMode(next: SearchInput["mode"]) {
    setMode(next); setSelected(PROVIDERS.filter(p => (p.modes as readonly string[]).includes(next)).map(p => p.id));
  }
  async function search(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(""); setResult(null); setSurvey("all");
    try {
      const payload: SearchInput = { mode, start, end, providers: selected };
      if (mode === "field") Object.assign(payload, { ra: Number(ra), dec: Number(dec), radius: Number(radius) });
      if (mode === "object") payload.object = object;
      if (mode === "orbits") payload.orbits = JSON.parse(orbits);
      const response = await fetch("/api/frames", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: AbortSignal.timeout(65000) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? "Search failed.");
      setResult(body);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed. Please retry.");
    } finally { setBusy(false); }
  }
  return (
    <main className="mx-auto w-full min-w-0 max-w-6xl px-4 py-8 font-mono sm:px-8 sm:py-12">
      <nav className="flex items-center justify-between text-xs text-[var(--text-secondary)]">
        <Link href="/constellation" className="hover:underline">const ellation;</Link>
        <a href="/api/frames" className="hover:underline">API / capabilities ↗</a>
      </nav>
      <header className="mt-12 max-w-3xl">
        <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Public archive search · v0.1</p>
        <h1 className="mt-3 text-4xl font-medium tracking-tight sm:text-5xl">Find the frames.</h1>
        <p className="mt-5 text-sm leading-6 text-[var(--text-secondary)]">A field of sky. A window in time. Find which connected archives hold matching images—and get the product links to inspect them.</p>
      </header>
      <div className="mt-10 grid gap-8 lg:grid-cols-[minmax(0,2fr)_minmax(240px,1fr)]">
        <form onSubmit={search} className="rounded-lg border border-[var(--border)] bg-white p-5 sm:p-7">
          <fieldset disabled={busy}>
            <legend className="text-xs font-semibold">01 / Define the search</legend>
            <div className="mt-5 flex flex-wrap gap-2" role="group" aria-label="Search input">
              {(["field", "object", "orbits"] as const).map(m => <button key={m} type="button" aria-pressed={mode === m} onClick={() => chooseMode(m)} className={`${buttonClass} ${mode === m ? "bg-[var(--foreground)] text-white hover:bg-black" : ""}`}>{m === "field" ? "Sky field" : m === "object" ? "Known moving object" : "Candidate orbits"}</button>)}
            </div>
            {mode === "field" && <>
              <div className="mt-6 grid gap-4 sm:grid-cols-3">
                <label className="text-xs">RA / degrees<input required type="number" min="0" max="360" step="any" value={ra} onChange={e => setRa(e.target.value)} className={inputClass} /></label>
                <label className="text-xs">Dec / degrees<input required type="number" min="-90" max="90" step="any" value={dec} onChange={e => setDec(e.target.value)} className={inputClass} /></label>
                <label className="text-xs">Radius / degrees<input required type="number" min="0.001" max="2" step="any" value={radius} onChange={e => setRadius(e.target.value)} className={inputClass} /></label>
              </div>
              <p className="mt-3 text-[11px] leading-5 text-[var(--text-muted)]">ICRS coordinates. Any target type. Circular fields up to 2° radius; ZTF returns candidates from the surrounding square.</p>
            </>}
            {mode === "object" && <div className="mt-6">
              <label className="text-xs">Solar System name or designation<input required maxLength={100} value={object} onChange={e => setObject(e.target.value)} className={inputClass} /></label>
              <p className="mt-3 text-[11px] leading-5 text-[var(--text-muted)]">Resolved by the archive ephemeris services. For stars, galaxies, or other fixed sources, use their coordinates in Sky field.</p>
              <button type="button" className="mt-2 text-xs underline" onClick={() => { setObject("2006 WG206"); setStart("2007-01-18"); setEnd("2007-01-19"); }}>Use historical CFHT example</button>
            </div>}
            {mode === "orbits" && <div className="mt-6">
              <label className="text-xs">Candidate orbits / JSON<textarea required rows={12} value={orbits} onChange={e => setOrbits(e.target.value)} spellCheck={false} className={`${inputClass} text-xs`} /></label>
              <p className="mt-3 text-[11px] leading-5 text-[var(--text-muted)]">1–3 heliocentric J2000 ecliptic osculating orbits. a: au; e: 0 ≤ e &lt; 1; i, node, peri, meanAnomaly: degrees. epochMjd is passed unchanged to each service; no epoch time-scale conversion is performed. Hyperbolic element sets and uncertainty distributions are not supported in v0.1.</p>
            </div>}
            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <label className="text-xs">Start date / UTC, inclusive<input required type="date" min="1900-01-01" max="2100-12-31" value={start} onChange={e => setStart(e.target.value)} className={inputClass} /></label>
              <label className="text-xs">End date / UTC, exclusive<input required type="date" min="1900-01-01" max="2100-12-31" value={end} onChange={e => setEnd(e.target.value)} className={inputClass} /></label>
            </div>
            <p className="mt-3 text-[11px] text-[var(--text-muted)]">Up to 10 years per query. SSOIS retains native archive date/time conventions.</p>
          </fieldset>
          <fieldset disabled={busy} className="mt-8 border-t border-[var(--border)] pt-6">
            <legend className="sr-only">Archives</legend>
            <p className="text-xs font-semibold">02 / Choose archives</p>
            <div className="mt-4 space-y-3">{available.map(p => <label key={p.id} className="flex items-center gap-3 text-xs"><input type="checkbox" checked={selected.includes(p.id)} onChange={e => setSelected(e.target.checked ? [...selected, p.id] : selected.filter(id => id !== p.id))} className="accent-black" />{p.name}</label>)}</div>
            <button disabled={busy || !selected.length} className="mt-7 w-full rounded bg-[var(--foreground)] px-5 py-3 text-sm text-white disabled:opacity-50">{busy ? "Searching archive indexes…" : "Find matching images →"}</button>
          </fieldset>
          <div aria-live="polite" className="mt-3 text-xs leading-5">{busy && "Each archive can take up to 22 seconds; multiple orbit queries may take longer."}</div>
          {error && <p role="alert" className="mt-3 text-sm text-red-800">{error}</p>}
        </form>
        <aside className="space-y-7 py-2 text-xs leading-6 text-[var(--text-secondary)]">
          <div><h2 className="font-semibold text-[var(--foreground)]">Coverage is the first question.</h2><p className="mt-2">This tool finds candidate image products. It does not establish whether your target is detectable, or whether it falls on usable detector pixels.</p></div>
          <div><h2 className="font-semibold text-[var(--foreground)]">A search you can audit.</h2><p className="mt-2">Every archive reports its own outcome, query link, and limits. A timeout is unknown coverage. A successful empty query only means that index returned no matches.</p></div>
          <div><h2 className="font-semibold text-[var(--foreground)]">Small requests, useful manifests.</h2><p className="mt-2">Up to 300 products per archive and orbit. Narrow the dates for dense fields. Download JSON for the full query receipt and original metadata, or CSV for a frame list.</p></div>
          <Link href="/constellation" className="inline-block underline underline-offset-4">See the wider dataset inventory ↗</Link>
        </aside>
      </div>
      {result && <section className="mt-12" aria-label="Search results">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div><p className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">03 / Archive results</p><h2 className="mt-2 text-2xl">{result.frames.length} image products <span className="text-sm text-[var(--text-muted)]">/ {surveys.length} collections</span></h2>
            <p className="mt-2 text-xs text-[var(--text-secondary)]">Submitted {result.input.start} → {result.input.end} · {result.input.mode === "field" ? `RA ${result.input.ra}°, Dec ${result.input.dec}°, radius ${result.input.radius}°` : result.input.mode === "object" ? result.input.object : `${result.input.orbits?.length} orbit hypotheses`}</p>
          </div>
          <div className="flex gap-2">
            <button className={buttonClass} onClick={() => download("constellation-manifest.json", JSON.stringify(result, null, 2), "application/json")}>Manifest JSON ↓</button>
            <button className={buttonClass} onClick={() => download("constellation-frames.csv", framesCsv(result), "text/csv")}>All frames CSV ↓</button>
          </div>
        </div>
        <div className="mt-6 grid gap-3 md:grid-cols-2">{result.checks.map(c => <div key={`${c.provider}:${c.hypothesis}`} className={`rounded border p-4 text-xs ${c.status === "ok" ? "border-[var(--border)]" : "border-amber-400 bg-amber-50"}`}>
          <div className="flex justify-between gap-2"><strong>{PROVIDERS.find(p => p.id === c.provider)?.name}</strong><span>{c.status === "error" ? "Unknown coverage" : c.status === "limited" ? "Result limit reached" : `${c.count} products`}</span></div>
          {result.input.mode !== "field" && <p className="mt-2">{c.hypothesis}</p>}
          <p className="mt-2 leading-5 text-[var(--text-secondary)]">{c.message}</p>
          <a href={c.queryUrl} target="_blank" rel="noreferrer" className="mt-2 inline-block underline">Open archive query ↗</a>
        </div>)}</div>
        {!result.frames.length && <p className="my-8 rounded border border-[var(--border)] p-6 text-sm leading-6">{result.checks.some(c => c.status === "error") ? "No products returned. One or more archives could not be checked; their coverage remains unknown." : "The selected archive indexes returned no matches for this request. Other archives or dates may still contain observations."}</p>}
        {!!result.frames.length && <>
          <label className="mt-6 flex flex-wrap items-center gap-3 text-xs">Collection<select value={survey} onChange={e => setSurvey(e.target.value)} className="max-w-full rounded border border-[var(--border)] bg-white p-2"><option value="all">All collections ({result.frames.length})</option>{surveys.map(s => <option key={s}>{s}</option>)}</select><span>{frames.length} shown</span></label>
          <div className="mt-4 overflow-x-auto rounded border border-[var(--border)]">
            <table className="w-full text-left text-xs"><thead className="bg-[var(--stripe)] text-[10px] uppercase tracking-wider"><tr>{["Collection / product", "MJD / time scale", "Band / exposure", "Match", "Retrieve"].map(h => <th key={h} className="p-3 font-medium">{h}</th>)}</tr></thead>
              <tbody>{frames.map(f => <tr key={f.key} className="border-t border-[var(--border)] align-top">
                <td className="max-w-72 break-words p-3"><div className="font-semibold">{f.survey}</div><div className="mt-1 text-[var(--text-secondary)]">{f.id}</div>{f.calibrationLevel !== null && <div className="mt-1 text-[10px] text-[var(--text-muted)]">Calibration level {f.calibrationLevel}{f.calibrationLevel >= 3 ? " · combined/advanced product" : ""}</div>}</td>
                <td className="whitespace-nowrap p-3 tabular-nums">{f.startMjd?.toFixed(5) ?? "Unknown"}<div className="mt-1 text-[10px] text-[var(--text-muted)]">{f.timeScale}</div>{f.endMjd !== null && <div className="mt-1 text-[10px]">to {f.endMjd.toFixed(5)}</div>}</td>
                <td className="whitespace-nowrap p-3">{f.filter ?? "See metadata"}<div className="mt-1 text-[var(--text-secondary)]">{f.exposureSeconds !== null ? `${f.exposureSeconds.toFixed(1)} s` : "Exposure unknown"}</div></td>
                <td className="p-3">{f.match}{result.input.mode === "orbits" && <div className="mt-1 text-[10px]">{f.hypotheses.length}/{result.input.orbits?.length} supplied hypotheses<br />{f.hypotheses.join(", ")}</div>}</td>
                <td className="whitespace-nowrap p-3">{f.url ? <a href={f.url} target="_blank" rel="noreferrer" className="underline underline-offset-4">{f.linkKind} ↗</a> : "No product link"}</td>
              </tr>)}</tbody>
            </table>
          </div>
        </>}
        <details className="mt-6 border-t border-[var(--border)] py-5 text-xs leading-6"><summary className="cursor-pointer font-semibold">Search scope and limitations</summary>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-[var(--text-secondary)]">{result.limitations.map(l => <li key={l}>{l}</li>)}</ul>
          <h3 className="mt-4 font-semibold">Other adapters not queried</h3><ul className="mt-2">{result.unqueried.map(p => <li key={p.name}>{p.name}: {p.reason}</li>)}</ul>
          <p className="mt-3">Query receipt: {result.queriedAt}. Exports include all returned collections regardless of the table filter.</p>
        </details>
      </section>}
      <footer className="mt-14 border-t border-[var(--border)] pt-5 text-[10px] leading-5 text-[var(--text-muted)]">Powered by public CADC, ESO, and IRSA archive services. Please acknowledge the original surveys when using their data.</footer>
    </main>
  );
}
