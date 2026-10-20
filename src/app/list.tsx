"use client";

import { useMemo, useState } from "react";
import type { Section, Tool } from "@/lib/catalog";


type SortKey = "order" | "feasibility" | "impact" | "sum";
type SortId = "order" | `${Exclude<SortKey, "order">}-${"desc" | "asc"}`;
const SORTS: { id: SortId; label: string }[] = [
  { id: "order", label: "Catalogue order" },
  { id: "feasibility-desc", label: "Feasibility, high to low" },
  { id: "feasibility-asc", label: "Feasibility, low to high" },
  { id: "impact-desc", label: "Impact, high to low" },
  { id: "impact-asc", label: "Impact, low to high" },
  { id: "sum-desc", label: "Feasibility + impact, high to low" },
  { id: "sum-asc", label: "Feasibility + impact, low to high" },
];
const FIELD = "h-10 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 text-sm outline-none focus:border-[var(--text-secondary)]";
const CHEVRON =
  "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%237a7974' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='m6 9 6 6 6-6'/></svg>\")";
const SELECT = `${FIELD} appearance-none cursor-pointer pr-8 bg-no-repeat bg-[length:14px_14px] bg-[position:right_0.6rem_center] hover:border-[var(--text-secondary)]`;

function Code({ text }: { text: string }) {
  const parts = text.split(/(`[^`]+`)/);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("`") ? (
          <code key={i} className="font-mono text-[12px] text-[var(--foreground)]">
            {p.slice(1, -1)}
          </code>
        ) : (
          p
        ),
      )}
    </>
  );
}

const COLS = ["In", "Out", "Ships", "Exists", "Why"];
const TH = "px-2 py-1.5 text-left align-bottom text-[10px] font-medium uppercase tracking-[0.08em] text-[var(--text-muted)]";
const TD = "px-2 py-2 align-top text-[12.5px] leading-[18px] text-[var(--text-secondary)]";

function Row({ tool, onTag }: { tool: Tool; onTag: (t: string) => void }) {
  const by = new Map(tool.fields.map((f) => [f.label, f.text]));
  const what = by.get("What");
  return (
    <tr className="odd:bg-[var(--stripe)]">
      <td className={`${TD} w-44`}>
        <code className="font-mono text-[13px] font-medium text-[var(--foreground)]">{tool.name}</code>
        <div className="mt-0.5 flex flex-wrap gap-x-2 font-mono text-[11px] tabular-nums text-[var(--text-muted)]">
          <span title={`feasibility ${tool.feasibility} of 10, impact ${tool.impact} of 10`}>
            f{tool.feasibility} i{tool.impact}
          </span>
        </div>
        <div className="mt-0.5 flex flex-wrap gap-x-2 text-[11px] text-[var(--text-muted)]">
          {tool.tags.map((t) => (
            <button key={t} onClick={() => onTag(t)} className="hover:text-[var(--foreground)]">
              {t}
            </button>
          ))}
        </div>
      </td>
      {what ? (
        <td className={TD} colSpan={3}>
          <Code text={what} />
        </td>
      ) : (
        COLS.slice(0, 3).map((c) => (
          <td key={c} className={TD}>
            {by.get(c) && <Code text={by.get(c)!} />}
          </td>
        ))
      )}
      {COLS.slice(3).map((c) => (
        <td key={c} className={TD}>
          {by.get(c) && <Code text={by.get(c)!} />}
        </td>
      ))}
    </tr>
  );
}

function Table({ rows, onTag }: { rows: Tool[]; onTag: (t: string) => void }) {
  return (
    <div className="-mx-4 mt-4 overflow-x-auto sm:-mx-6">
      <table className="w-full min-w-[960px] border-collapse">
        <thead>
          <tr className="border-b border-[var(--border)]">
            <th className={`${TH} w-44`}>tool</th>
            {COLS.map((c) => (
              <th key={c} className={TH}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((t) => (
            <Row key={t.name} tool={t} onTag={onTag} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

const norm = (s: string) => s.toLowerCase();

export function List({ sections }: { sections: Section[] }) {
  const [query, setQuery] = useState("");
  const [tag, setTag] = useState("");
  const [sortId, setSortId] = useState<SortId>("order");
  const [sort, dir] = sortId.split("-") as [SortKey, "desc" | "asc" | undefined];
  const desc = dir !== "asc";

  const allTags = useMemo(() => {
    const seen = new Set<string>();
    sections.forEach((s) => s.tools.forEach((t) => t.tags.forEach((x) => seen.add(x))));
    return [...seen];
  }, [sections]);

  const toggleTag = (t: string) => setTag((prev) => (prev === t ? "" : t));

  const q = norm(query);
  const keep = (t: Tool) =>
    (!tag || t.tags.includes(tag)) &&
    (!q || norm(t.name).includes(q) || t.fields.some((f) => norm(f.text).includes(q)));

  const flat = useMemo(() => {
    const rows = sections.flatMap((s) => s.tools.map((t) => ({ tool: t, section: s.title })));
    const val = (t: Tool) =>
      sort === "feasibility" ? t.feasibility : sort === "impact" ? t.impact : t.feasibility + t.impact;
    rows.sort((a, b) => val(b.tool) - val(a.tool) || a.tool.name.localeCompare(b.tool.name));
    return rows;
  }, [sections, sort]);

  const shown = sort === "order" ? null : flat.filter((r) => keep(r.tool));
  const grouped =
    sort === "order"
      ? sections.map((s) => ({ ...s, tools: (desc ? s.tools : [...s.tools].reverse()).filter(keep) })).filter((s) => s.tools.length)
      : null;
  const total = shown ? shown.length : grouped!.reduce((n, s) => n + s.tools.length, 0);

  return (
    <>
      <div className="sticky top-0 z-30 -mx-4 mt-6 bg-[var(--background)] px-4 py-3 sm:-mx-6 sm:px-6">
        <div className="flex flex-wrap gap-2">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Find a tool (e.g. orbit, runaway, Gaia)"
            className={`${FIELD} min-w-0 flex-1 basis-56`}
          />
          <select
            value={sortId}
            onChange={(e) => setSortId(e.target.value as SortId)}
            aria-label="Sort"
            className={SELECT}
            style={{ backgroundImage: CHEVRON }}
          >
            {SORTS.map((o) => (
              <option key={o.id} value={o.id}>
                {o.label}
              </option>
            ))}
          </select>
          <select
            value={tag}
            onChange={(e) => setTag(e.target.value)}
            aria-label="Tag"
            className={SELECT}
            style={{ backgroundImage: CHEVRON }}
          >
            <option value="">All tags</option>
            {allTags.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <span className="self-center text-xs tabular-nums text-[var(--text-muted)]">{total} tools</span>
        </div>
      </div>

      {grouped && <Table rows={grouped.flatMap((s) => s.tools)} onTag={toggleTag} />}
      {shown && <Table rows={desc ? shown.map((r) => r.tool) : [...shown].reverse().map((r) => r.tool)} onTag={toggleTag} />}

      {total === 0 && <p className="mt-6 text-sm text-[var(--text-secondary)]">No tool matches.</p>}
    </>
  );
}
