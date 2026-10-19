import type { Metadata } from "next";
import { Explorer } from "../explorer";
import { DATASETS, RECORDS, type Step } from "@/lib/constellation";

export const metadata: Metadata = {
  title: "const ellation",
  description: "Every 2 sigma detection in every public dataset, every public record of every known object, and the difference.",
};

const TH = "px-2 py-1.5 text-left align-bottom text-[10px] font-medium uppercase tracking-[0.08em] text-[var(--text-muted)]";
const TD = "px-2 py-2 align-top text-[12.5px] leading-[18px] text-[var(--text-secondary)]";

function Status({ step }: { step: Step }) {
  const name = ["empty", "half", "full"][step];
  return (
    <span className="relative inline-block h-3 w-3 overflow-hidden rounded-full border border-[var(--text-secondary)] align-[-1px]" title={name}>
      {step > 0 && <span className={`absolute inset-y-0 left-0 bg-[var(--text-secondary)] ${step === 2 ? "w-full" : "w-1/2"}`} />}
    </span>
  );
}

function Arrow({ href }: { href: string }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="inline-block text-[var(--text-muted)] hover:text-[var(--foreground)]" aria-label="open archive">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M7 17 17 7M9 7h8v8" />
      </svg>
    </a>
  );
}

function Table({ head, rows }: { head: string[]; rows: { key: string; step: Step; cells: string[]; url: string }[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse">
        <thead>
          <tr className="border-b border-[var(--border)]">
            <th className={`${TH} w-10`} />
            {head.map((h) => (
              <th key={h} className={TH}>
                {h}
              </th>
            ))}
            <th className={`${TH} w-10`} />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="odd:bg-[var(--stripe)]">
              <td className={`${TD} text-center`}>
                <Status step={r.step} />
              </td>
              {r.cells.map((c, i) => (
                <td key={i} className={`${TD} ${i === 0 ? "font-medium text-[var(--foreground)]" : ""} ${i === head.length - 1 ? "whitespace-nowrap text-[11.5px] tabular-nums" : ""}`}>
                  {c}
                </td>
              ))}
              <td className={`${TD} text-center`}>
                <Arrow href={r.url} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Page() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-10 font-mono sm:px-6">
      <h1 className="text-2xl font-semibold tracking-tight">const ellation;</h1>
      <p className="mt-2 text-sm text-[var(--text-secondary)]">mission: leverage the entire corpus of astronomical data all at once.</p>
      <Explorer />
      <section className="mt-12">
        <h2 className="mb-3 text-sm font-semibold">Part 1: Compile every public image.</h2>
        <Table
          head={["Dataset", "Depth", "Volume"]}
          rows={DATASETS.map((d) => ({ key: d.name, step: d.step, cells: [d.name, d.depth, d.volume], url: d.url }))}
        />
      </section>
      <section className="mt-12">
        <h2 className="mb-3 text-sm font-semibold">Part 2: Compile every known object.</h2>
        <Table
          head={["Source", "Holds", "Size"]}
          rows={RECORDS.map((r) => ({ key: r.name, step: r.step, cells: [r.name, r.holds, r.size], url: r.url }))}
        />
      </section>
    </main>
  );
}
