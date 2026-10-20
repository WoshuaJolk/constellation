export type Row = Record<string, string>;
/** CSV/TSV parser with quoted newlines, escaped quotes, and strict row widths. */
export function delimited(text: string, delimiter = ","): Row[] {
  const records: string[][] = [];
  let row: string[] = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (c === delimiter && !quoted) { row.push(cell); cell = ""; }
    else if ((c === "\n" || c === "\r") && !quoted) {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      if (row.some(v => v.trim())) records.push(row);
      row = []; cell = "";
    } else cell += c;
  }
  if (quoted) throw new Error("Archive returned an incomplete quoted table.");
  if (cell || row.length) { row.push(cell); records.push(row); }
  const head = records.shift()?.map(s => s.trim().replace(/^\uFEFF/, ""));
  if (!head?.length) throw new Error("Archive returned no table header.");
  return records.map(values => {
    if (values.length !== head.length) throw new Error("Archive returned a malformed table row.");
    return Object.fromEntries(head.map((key, i) => [key, values[i].trim()]));
  });
}
/** IPAC data cells align with the header's pipe positions; whitespace splitting loses nulls. */
export function ipac(text: string, ibe: boolean): Row[] {
  if (ibe && !/\\QUERY_STATUS\s*=\s*['"]OK['"]/.test(text)) throw new Error("IRSA did not report a successful query.");
  if (!ibe && !text.includes("MOST ODBC query result")) throw new Error("MOST did not return its result table.");
  const lines = text.split(/\r?\n/);
  const header = lines.findIndex(l => l.startsWith("|"));
  if (header < 0) throw new Error("IRSA returned no table header.");
  const positions = [...lines[header].matchAll(/\|/g)].map(m => m.index!);
  const names = positions.slice(0, -1).map((p, i) => lines[header].slice(p + 1, positions[i + 1]).trim());
  const rows = lines.slice(header + 1).filter(l => l.trim() && !l.startsWith("|") && !l.startsWith("\\")).map(l => {
    if (l.length < positions.at(-1)!) throw new Error("IRSA returned a truncated data row.");
    return Object.fromEntries(names.map((n, i) => [n, l.slice(positions[i] + 1, positions[i + 1]).trim()]));
  });
  const count = text.match(/\\RowsRetrieved\s*=\s*(\d+)/);
  if (ibe && (!count || Number(count[1]) !== rows.length)) throw new Error("IRSA row count does not match the returned table.");
  return rows;
}
export function requireColumns(text: string, names: string[]) {
  const header = text.split(/\r?\n/).find(l => !l.startsWith("\\") && l.trim()) ?? "";
  const columns = header.startsWith("|") ? header.split("|").map(s => s.trim()) : header.split(/,|\t/).map(s => s.trim().replace(/^"|"$/g, ""));
  if (names.some(n => !columns.includes(n))) throw new Error("Archive response is missing required columns.");
}
export function numeric(s: string | undefined): number | null {
  if (!s || s.toLowerCase() === "null" || !Number.isFinite(Number(s))) return null;
  return Number(s);
}
