import type { Metadata } from "next";
import { Explorer } from "../explorer";
import { DATASETS, RECORDS, type Step } from "@/lib/constellation";

const TITLE = "const ellation;";
const DESC = "mission: leverage the entire corpus of astronomical data all at once.";

export const metadata: Metadata = {
  title: TITLE,
  description: DESC,
  openGraph: { title: TITLE, description: DESC, url: "https://const-ellation.com", siteName: TITLE, images: [{ url: "/og.png", width: 1200, height: 630 }] },
  twitter: { card: "summary_large_image", title: TITLE, description: DESC, images: ["/og.png"] },
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

function Table({ head, rows }: { head: string[]; rows: { key: string; step: Step; cells: string[]; url: string; note?: string; reference?: string }[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-[var(--border)]">
            <th className={`${TH} w-10`} />
            {head.map((h) => (
              <th key={h} className={TH}>
                {h}
              </th>
            ))}
            <th className={`${TH} hidden w-10 md:table-cell`} />
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
                  {i === 0 ? (
                    <>
                      <a href={r.url} target="_blank" rel="noreferrer" className="hover:underline">
                        {c}
                      </a>
                      {r.note && (
                        <p className="mt-1 max-w-xl text-[11px] font-normal leading-4 text-[var(--text-muted)]">
                          {r.note}
                          {r.reference && (
                            <> <a href={r.reference} target="_blank" rel="noreferrer" className="underline underline-offset-2">Source notes</a></>
                          )}
                        </p>
                      )}
                    </>
                  ) : (
                    c
                  )}
                </td>
              ))}
              <td className={`${TD} hidden text-center md:table-cell`}>
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
      <a href="/constellation/search" className="mt-5 inline-block rounded bg-[var(--foreground)] px-4 py-2.5 text-xs text-white">Find survey frames by field or orbit →</a>
      <Explorer />
      <section className="mt-12">
        <h2 className="mb-3 text-sm font-semibold">Part 1: Compile every public image.</h2>
        <p className="mb-4 max-w-3xl text-xs leading-5 text-[var(--text-secondary)]">
          Planning inventory reviewed October 20, 2026. Tool roles describe proposed use, not integration status.
          Index observation metadata first and retrieve images when candidate orbits intersect.
          Access-dependent and evaluate entries need release or product checks before inclusion.
        </p>
        <p className="mb-4 max-w-3xl text-xs leading-5 text-[var(--text-muted)]">
          Depths use different bands and product types; they are not uniform single-exposure limits.
          Counts and volumes are rough planning estimates, not verified archive totals.
          “Not verified” means unknown, not zero. Deduplicate exposures shared by surveys and archives.
        </p>
        <Table
          head={["Dataset", "Tool role", "Depth / product", "Volume"]}
          rows={DATASETS.map((d) => ({ key: d.name, step: d.step, cells: [d.name, d.role, d.depth, d.volume], url: d.url, note: d.note, reference: d.reference }))}
        />
      </section>
      <section className="mt-12">
        <h2 className="mb-3 text-sm font-semibold">Part 2: Compile every known object.</h2>
        <p className="mb-4 max-w-3xl text-xs leading-5 text-[var(--text-secondary)]">
          Orbits support predictions; detection catalogs help recover candidates; static-source and variable catalogs
          help interpret them. SSOIS, MOST and CATCH provide archive searches. These sources complement the image
          inventory and do not each represent additional exposures. Planned releases are labeled below.
        </p>
        <Table
          head={["Source", "Holds", "Size"]}
          rows={RECORDS.map((r) => ({ key: r.name, step: r.step, cells: [r.name, r.holds, r.size], url: r.url, note: r.note, reference: r.reference }))}
        />
      </section>
      <section className="mt-12">
        <h2 className="text-sm font-semibold">Part 3: Cross-analyze the data to uncover secrets across the solar system.</h2>
      </section>
    </main>
  );
}
