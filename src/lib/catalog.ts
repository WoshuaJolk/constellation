import { readFileSync } from "node:fs";
import path from "node:path";

export type Tool = {
  name: string;
  feasibility: number;
  impact: number;
  tags: string[];
  fields: { label: string; text: string }[];
};

export type Section = { title: string; note?: string; tools: Tool[] };

export type Scale = { name: string; stops: { value: number; text: string }[] };
export type TagDef = { name: string; text?: string };
export type Catalog = {
  intro: string[];
  scales: Scale[];
  tagDefs: TagDef[];
  glossary: string;
  sections: Section[];
  outro: string;
};

function parseLegend(p: string, scales: Scale[], tagDefs: TagDef[]) {
  for (const m of p.matchAll(/(Feasibility|Impact): ([^.]*)\./g)) {
    const stops = [...m[2].matchAll(/(\d+) = ([^,]+)/g)].map((x) => ({ value: +x[1], text: x[2].trim() }));
    scales.push({ name: m[1].toLowerCase(), stops });
  }
  const t = /Tags: (.*)$/.exec(p);
  if (t) for (const x of t[1].matchAll(/(\w+)(?: \(([^)]*)\))?/g)) tagDefs.push({ name: x[1], text: x[2] });
}

const HEAD = /^- `(\w+)` \((?:feasibility )?(\d+), (?:impact )?(\d+)\)(?: \[([^\]]*)\])?:?\s*(.*)$/;
const LABELS = ["In", "Out", "Ships", "Exists", "Why"];

function splitFields(body: string): Tool["fields"] {
  const re = new RegExp(`(?:^|\\s)(${LABELS.join("|")}): `, "g");
  const parts: Tool["fields"] = [];
  let m: RegExpExecArray | null;
  let last: { label: string; start: number } | null = null;
  while ((m = re.exec(body))) {
    if (last) parts.push({ label: last.label, text: body.slice(last.start, m.index).trim() });
    last = { label: m[1], start: m.index + m[0].length };
  }
  if (last) parts.push({ label: last.label, text: body.slice(last.start).trim() });
  else if (body.trim()) parts.push({ label: "What", text: body.trim() });
  return parts;
}

export function loadCatalog(): Catalog {
  const src = readFileSync(path.join(process.cwd(), "MAGIC_TOOLING.md"), "utf8");
  const lines = src.split("\n");
  const intro: string[] = [];
  const scales: Scale[] = [];
  const tagDefs: TagDef[] = [];
  let glossary = "";
  const sections: Section[] = [];
  let outro = "";
  let cur: Section | null = null;
  let tool: Tool | null = null;
  let body = "";
  let para = "";

  const flushTool = () => {
    if (tool) tool.fields = splitFields(body);
    tool = null;
    body = "";
  };
  const flushPara = () => {
    const p = para.trim();
    if (p) {
      if (!cur) {
        if (p.startsWith("Glossary.")) glossary = p.replace(/^Glossary\.\s*/, "");
        else if (p.startsWith("Feasibility:")) parseLegend(p, scales, tagDefs);
        else intro.push(p);
      } else if (cur.title.startsWith("Out of scope")) outro += (outro ? " " : "") + p;
      else cur.note = (cur.note ? cur.note + " " : "") + p;
    }
    para = "";
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (line.startsWith("# ")) continue;
    if (line.startsWith("## ")) {
      flushTool();
      flushPara();
      cur = { title: line.slice(3).trim(), tools: [] };
      sections.push(cur);
      continue;
    }
    const h = HEAD.exec(line);
    if (h) {
      if (!cur) {
        cur = { title: "", tools: [] };
        sections.push(cur);
      }
      flushTool();
      flushPara();
      tool = {
        name: h[1],
        feasibility: +h[2],
        impact: +h[3],
        tags: h[4] ? h[4].split(",").map((t) => t.trim()) : [],
        fields: [],
      };
      cur.tools.push(tool);
      body = h[5] ?? "";
      continue;
    }
    if (tool && line.startsWith("  ")) {
      body += " " + line.trim();
      continue;
    }
    if (line === "") {
      flushTool();
      flushPara();
      continue;
    }
    para += (para ? " " : "") + line.trim();
  }
  flushTool();
  flushPara();
  return { intro, scales, tagDefs, glossary, sections: sections.filter((s) => !s.title.startsWith("Out of scope")), outro };
}
