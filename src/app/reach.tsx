// Smallest body with at least one detection, by distance from the Sun, for a wide-field survey.
// Reflected light at opposition: m = H + 5 log10(r (r - 1)), D = 1329 / sqrt(p) * 10^(-H/5) km, albedo p = 0.1.
const PASS = 26; // deep wide-field frames summed to the noise floor along the track
const ALBEDO = 0.1;
const R = [2, 100000] as const;
const D = [1, 1000000] as const;
const W = 820;
const H = 300;
const PAD = { l: 56, r: 40, t: 10, b: 64 };

const dlim = (r: number, m: number) => (1329 / Math.sqrt(ALBEDO)) * 10 ** (-(m - 5 * Math.log10(r * (r - 1))) / 5);
const lx = (r: number) => PAD.l + ((Math.log10(r) - Math.log10(R[0])) / (Math.log10(R[1]) - Math.log10(R[0]))) * (W - PAD.l - PAD.r);
const ly = (d: number) => PAD.t + (1 - (Math.log10(d) - Math.log10(D[0])) / (Math.log10(D[1]) - Math.log10(D[0]))) * (H - PAD.t - PAD.b);
const clampD = (d: number) => Math.min(Math.max(d, D[0]), D[1]);

const REGIONS: { name: string; from: number; to: number }[] = [
  { name: "Main belt", from: 2.1, to: 3.3 },
  { name: "Trojans", from: 4.9, to: 5.5 },
  { name: "Centaurs", from: 5.5, to: 30 },
  { name: "Kuiper belt", from: 30, to: 50 },
  { name: "Scattered", from: 50, to: 150 },
  { name: "Sedna-like", from: 150, to: 1500 },
  { name: "Inner Oort", from: 1500, to: 20000 },
  { name: "Oort cloud", from: 20000, to: 100000 },
];

const BODIES: { name: string; r: number; d: number }[] = [
  { name: "1 km main-belt asteroid", r: 2.7, d: 1 },
  { name: "10 km Trojan", r: 5.2, d: 10 },
  { name: "50 km Centaur", r: 15, d: 50 },
  { name: "100 km Kuiper belt object", r: 42, d: 100 },
  { name: "300 km scattered object at aphelion", r: 100, d: 300 },
  { name: "Sedna now, 85 AU", r: 85, d: 1000 },
  { name: "Sedna at aphelion", r: 900, d: 1000 },
  { name: "Planet Nine", r: 500, d: 25000 },
  { name: "500 km inner Oort body", r: 3000, d: 500 },
  { name: "Earth-mass body in the Oort cloud", r: 30000, d: 12700 },
];

const XT = [2, 10, 100, 1000, 10000, 100000];
const YT = [1, 10, 100, 1000, 10000, 100000, 1000000];
const fmt = (n: number) => (n >= 1000000 ? `${n / 1000000}M` : n >= 1000 ? `${n / 1000}k` : `${n}`);

function path(m: number) {
  const pts: string[] = [];
  for (let i = 0; i <= 120; i++) {
    const r = 10 ** (Math.log10(R[0]) + (i / 120) * (Math.log10(R[1]) - Math.log10(R[0])));
    pts.push(`${lx(r).toFixed(1)},${ly(clampD(dlim(r, m))).toFixed(1)}`);
  }
  return pts;
}

export function Reach() {
  const pass = path(PASS);
  const yFloor = ly(D[0]);
  const yTop = ly(D[1]);
  const band = `M${pass.join(" L")} L${lx(R[1])},${yTop} L${lx(R[0])},${yTop} Z`;
  const under = `M${pass.join(" L")} L${lx(R[1])},${yFloor} L${lx(R[0])},${yFloor} Z`;
  const status = (b: { r: number; d: number }) => (b.d >= dlim(b.r, PASS) ? "pass" : "out");
  const fill = { pass: "var(--added)", out: "var(--background)" } as const;
  return (
    <section className="mt-6 max-w-[820px]">
      <h2 className="text-base font-semibold tracking-tight">What a unified search of every survey could reach</h2>
      <p className="mt-1 text-sm leading-6 text-[var(--text-secondary)]">
        Smallest body that leaves a detection, by distance from the Sun, when every deep wide-field frame is summed along
        the track to the noise floor. Reflected light at opposition, albedo 0.1. Above the line is searchable. Below it
        left too few photons for any algorithm.
      </p>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[var(--text-secondary)]">
        <span><span className="mr-1 inline-block h-2.5 w-4 align-[-1px] bg-[var(--added)] opacity-20" />searchable, summed to m {PASS}</span>
        <span><span className="mr-1 inline-block h-2.5 w-4 align-[-1px]" style={{ backgroundImage: "repeating-linear-gradient(135deg, var(--axis) 0 1px, transparent 1px 3px)" }} />out of reach</span>
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full border-[1.5px] border-[var(--text-secondary)] align-[-1px]" />example bodies, filled when reachable</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 w-full" role="img" aria-label="Smallest searchable body against distance from the Sun with a unified search of every survey">
        <defs>
          <pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="4" stroke="var(--axis)" strokeWidth="1" />
          </pattern>
        </defs>
        {YT.map((d) => (
          <g key={d}>
            <line x1={PAD.l} x2={W - PAD.r} y1={ly(d)} y2={ly(d)} stroke="var(--border)" strokeWidth="1" />
            <text x={PAD.l - 6} y={ly(d) + 3.5} textAnchor="end" fontSize="10" fill="var(--text-muted)" fontFamily="var(--font-geist-mono)">
              {d === D[1] ? `${fmt(d)} km` : fmt(d)}
            </text>
          </g>
        ))}
        {XT.map((r) => (
          <text key={r} x={lx(r)} y={H - PAD.b + 14} textAnchor="middle" fontSize="10" fill="var(--text-muted)" fontFamily="var(--font-geist-mono)">
            {r === R[1] ? `${fmt(r)} AU` : fmt(r)}
          </text>
        ))}
        <path d={under} fill="url(#hatch)" />
        <path d={band} fill="var(--added)" fillOpacity="0.12" />
        <path d={`M${pass.join(" L")}`} fill="none" stroke="var(--added)" strokeWidth="2" strokeLinejoin="round" />
        {REGIONS.map((g, i) => (
          <g key={g.name}>
            <rect x={lx(g.from)} y={H - PAD.b + 24} width={lx(g.to) - lx(g.from) - 2} height="6" rx="1" fill={i % 2 ? "var(--axis)" : "var(--border)"} />
            <text x={(lx(g.from) + lx(g.to)) / 2} y={H - PAD.b + 44 + (i % 2 ? 11 : 0)} textAnchor="middle" fontSize="10" fill="var(--text-secondary)">
              {g.name}
            </text>
          </g>
        ))}
        {BODIES.map((b) => {
          const s = status(b);
          return (
            <g key={b.name}>
              <title>{`${b.name}: ${s === "pass" ? "searchable" : "out of reach"}`}</title>
              <circle cx={lx(b.r)} cy={ly(b.d)} r="9" fill="transparent" />
              <circle cx={lx(b.r)} cy={ly(b.d)} r="4" fill={fill[s]} stroke={s === "out" ? "var(--text-secondary)" : "var(--background)"} strokeWidth="2" />
            </g>
          );
        })}
        {BODIES.map((b) => {
          const s = status(b);
          const labels: Record<string, [number, number, "start" | "end"]> = {
            "100 km Kuiper belt object": [8, -6, "start"],
            "Planet Nine": [8, 4, "start"],
            "Sedna now, 85 AU": [-8, 4, "end"],
            "Sedna at aphelion": [8, 4, "start"],
            "500 km inner Oort body": [8, 4, "start"],
            "Earth-mass body in the Oort cloud": [-8, 4, "end"],
          };
          const l = labels[b.name];
          if (!l) return null;
          return (
            <text key={b.name} x={lx(b.r) + l[0]} y={ly(b.d) + l[1]} textAnchor={l[2]} fontSize="10" fill={s === "out" ? "var(--text-muted)" : "var(--text-secondary)"}>
              {b.name}
            </text>
          );
        })}
      </svg>
    </section>
  );
}
