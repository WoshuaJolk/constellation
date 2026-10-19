import type { Metadata } from "next";
import { Reach } from "../reach";
import { DATASETS, PRIVATE, RECORDS, PIXEL_STEPS, RECORD_STEPS, type Step } from "@/lib/constellation";

export const metadata: Metadata = {
  title: "const ellation",
  description: "Every 2 sigma detection in every public dataset, every public record of every known object, and the difference.",
};

const TH = "px-2 py-1.5 text-left align-bottom text-[10px] font-medium uppercase tracking-[0.08em] text-[var(--text-muted)]";
const TD = "px-2 py-2 align-top text-[12.5px] leading-[18px] text-[var(--text-secondary)]";

function Steps({ step, names }: { step: Step; names: string[] }) {
  return (
    <span className="flex items-center gap-1" title={`${names[step - 1] ?? "none"}`}>
      {names.map((n, i) => (
        <span
          key={n}
          className={`h-2.5 w-2.5 rounded-sm border ${i < step ? "border-[var(--added)] bg-[var(--added)]" : "border-[var(--axis)]"}`}
        />
      ))}
      <span className="ml-1 font-mono text-[11px] text-[var(--text-muted)]">{names[step - 1]}</span>
    </span>
  );
}

function Count({ steps, names }: { steps: Step[]; names: string[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] tabular-nums text-[var(--text-muted)]">
      {names.map((n, i) => (
        <span key={n}>
          <span className="text-[var(--foreground)]">{steps.filter((s) => s >= i + 1).length}</span> {n}
        </span>
      ))}
    </div>
  );
}

function Volume(tb: number) {
  return tb >= 1000 ? `${(tb / 1000).toFixed(1)} PB` : `${Math.round(tb)} TB`;
}

export default function Page() {
  const totalTb = DATASETS.reduce((a, d) => a + d.tb, 0);
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6">
      <h1 className="font-mono text-2xl font-semibold tracking-tight">const ellation;</h1>
      <div className="mt-2 max-w-[640px] space-y-3 text-sm leading-6 text-[var(--text-secondary)]">
        <p>
          One bucket, two tables. The first holds every detection down to 2 sigma from every public image of the sky,
          with flux and error. The second holds every public record of every known object, moving and fixed. Subtract
          the second from the first, link what remains across nights and years, and what is left has never been
          catalogued.
        </p>
        <p>Progress is tracked per dataset: listed, access tested, footprint indexed, 2 sigma pass written.</p>
      </div>
      <Reach />

      <section className="mt-10">
        <h2 className="text-sm font-semibold">1. Every public image</h2>
        <p className="mt-1 max-w-[640px] text-sm leading-6 text-[var(--text-secondary)]">
          {DATASETS.length} datasets, about {Volume(totalTb)} of pixels. Depth is one exposure at 5 sigma; the pass
          keeps everything above 2.
        </p>
        <div className="mt-2">
          <Count steps={DATASETS.map((d) => d.step)} names={PIXEL_STEPS} />
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[1100px] border-collapse">
            <thead>
              <tr className="border-b border-[var(--border)]">
                {["Dataset", "Instrument", "Years", "Sky", "Depth", "Exposures", "Volume", "Archive · compute", "Status"].map((h) => (
                  <th key={h} className={TH}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {DATASETS.map((d) => (
                <tr key={d.name} className="odd:bg-[var(--stripe)]">
                  <td className={`${TD} w-48`}>
                    <span className="font-medium text-[var(--foreground)]">{d.name}</span>
                    {d.note && <div className="mt-0.5 text-[11px] leading-4 text-[var(--text-muted)]">{d.note}</div>}
                  </td>
                  <td className={TD}>{d.instrument}</td>
                  <td className={`${TD} whitespace-nowrap font-mono text-[11.5px]`}>{d.years}</td>
                  <td className={TD}>{d.sky}</td>
                  <td className={`${TD} whitespace-nowrap`}>{d.depth}</td>
                  <td className={`${TD} whitespace-nowrap font-mono text-[11.5px] tabular-nums`}>{d.exposures}</td>
                  <td className={`${TD} whitespace-nowrap font-mono text-[11.5px] tabular-nums`}>{d.volume}</td>
                  <td className={TD}>{d.where}</td>
                  <td className={`${TD} whitespace-nowrap`}>
                    <Steps step={d.step} names={PIXEL_STEPS} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 max-w-[640px] text-[12px] leading-5 text-[var(--text-muted)]">
          Not public, so not in the pass: {PRIVATE.join("; ")}.
        </p>
      </section>

      <section className="mt-12">
        <h2 className="text-sm font-semibold">2. Every public record of a known object</h2>
        <p className="mt-1 max-w-[640px] text-sm leading-6 text-[var(--text-secondary)]">
          {RECORDS.length} sources. Moving bodies first, then the static sky that every detection is matched against.
        </p>
        <div className="mt-2">
          <Count steps={RECORDS.map((r) => r.step)} names={RECORD_STEPS} />
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[900px] border-collapse">
            <thead>
              <tr className="border-b border-[var(--border)]">
                {["Source", "Holds", "Size", "Years", "Where", "Status"].map((h) => (
                  <th key={h} className={TH}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {RECORDS.map((r) => (
                <tr key={r.name} className="odd:bg-[var(--stripe)]">
                  <td className={`${TD} w-56`}>
                    <span className="font-medium text-[var(--foreground)]">{r.name}</span>
                    {r.note && <div className="mt-0.5 text-[11px] leading-4 text-[var(--text-muted)]">{r.note}</div>}
                  </td>
                  <td className={TD}>{r.holds}</td>
                  <td className={`${TD} whitespace-nowrap font-mono text-[11.5px] tabular-nums`}>{r.size}</td>
                  <td className={`${TD} whitespace-nowrap font-mono text-[11.5px]`}>{r.years}</td>
                  <td className={TD}>{r.where}</td>
                  <td className={`${TD} whitespace-nowrap`}>
                    <Steps step={r.step} names={RECORD_STEPS} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-12 border-t border-[var(--border)] pt-4">
        <h2 className="text-sm font-semibold">3. The difference</h2>
        <p className="mt-1 max-w-[640px] text-sm leading-6 text-[var(--text-secondary)]">
          Nothing yet. It needs the first pass on at least one dataset and the first two record sources mirrored into
          one schema.
        </p>
      </section>
    </main>
  );
}
