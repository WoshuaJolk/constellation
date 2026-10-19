import Link from "next/link";
import { loadCatalog, type Scale } from "@/lib/catalog";
import { List } from "./list";
import { Reach } from "./reach";

function Code({ text }: { text: string }) {
  const parts = text.split(/(`[^`]+`)/);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("`") ? (
          <code key={i} className="font-mono text-[13px] text-[var(--foreground)]">
            {p.slice(1, -1)}
          </code>
        ) : (
          p
        ),
      )}
    </>
  );
}

const LABEL = "text-[10px] font-medium uppercase tracking-[0.08em] text-[var(--text-muted)]";

function Spectrum({ scale }: { scale: Scale }) {
  const stops = [...scale.stops].sort((a, b) => a.value - b.value);
  const align = ["items-start text-left", "items-center text-center", "items-end text-right"];
  return (
    <div className="grid grid-cols-[4.5rem_minmax(0,1fr)] items-start gap-x-3">
      <span className={`${LABEL} pt-[3px]`}>{scale.name}</span>
      <div>
        <div className="relative h-4">
          <span className="absolute inset-x-0 top-0 h-0.5 rounded-full bg-[var(--axis)]" />
          {["left-0", "left-1/2", "right-0"].map((c) => (
            <span key={c} className={`absolute top-0 h-4 w-px bg-[var(--text-muted)] ${c}`} />
          ))}
        </div>
        <div className="grid grid-cols-3">
          {stops.map((st, i) => (
            <span key={st.value} className={`flex flex-col ${align[i] ?? align[2]}`}>
              <span className="font-mono text-[11px] tabular-nums text-[var(--foreground)]">{st.value}</span>
              <span className="text-[11px] leading-4 text-[var(--text-secondary)]">{st.text}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const cat = loadCatalog();
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6">
      <div className="flex items-baseline justify-between gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">Magic tools</h1>
        <Link href="/constellation" className="font-mono text-[12px] text-[var(--text-muted)] hover:text-[var(--foreground)]">
          const ellation;
        </Link>
      </div>
      <div className="mt-2 max-w-[640px] space-y-3 text-sm leading-6 text-[var(--text-secondary)]">
        {cat.intro.map((p, i) => (
          <p key={i}>
            <Code text={p} />
          </p>
        ))}
      </div>
      <Reach />
      <div className="mt-8 max-w-[640px] space-y-4">
        {cat.scales.map((sc) => (
          <Spectrum key={sc.name} scale={sc} />
        ))}
      </div>
      <details className="mt-4 max-w-[640px] text-sm leading-6 text-[var(--text-secondary)]">
        <summary className="cursor-pointer text-[var(--foreground)]">Glossary</summary>
        <p className="mt-2">{cat.glossary}</p>
      </details>

      <List sections={cat.sections} />

      {cat.outro && (
        <section className="mt-10 border-t border-[var(--border)] pt-4">
          <h2 className="text-sm font-semibold">Out of scope here: instruments, missions and conditions</h2>
          <p className="mt-1 max-w-[640px] text-sm leading-6 text-[var(--text-secondary)]">
            <Code text={cat.outro} />
          </p>
        </section>
      )}
    </main>
  );
}
