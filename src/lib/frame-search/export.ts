import type { SearchResult } from "./model";
function cell(value: unknown): string {
  let text = value === null || value === undefined ? "" : String(value);
  // Prevent spreadsheet formula execution in remote archive strings.
  if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}
export function framesCsv(result: SearchResult): string {
  const columns = ["provider", "survey", "id", "startMjd", "endMjd", "timeScale", "exposureSeconds", "filter", "match", "hypotheses", "url", "linkKind", "calibrationLevel", "queryStatus"];
  return [columns.join(","), ...result.frames.map(f => columns.map(k => cell(k === "queryStatus" ? result.checks.filter(c => c.provider === f.provider && f.hypotheses.includes(c.hypothesis)).map(c => `${c.hypothesis}:${c.status}`).join("; ") : k === "hypotheses" ? f.hypotheses.join("; ") : f[k as keyof typeof f])).join(","))].join("\r\n");
}
