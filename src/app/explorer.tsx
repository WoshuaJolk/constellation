"use client";

import { useState } from "react";
import { SURVEYS, summed } from "@/lib/sky";
import { Reach } from "./reach";
import { Volume } from "./volume";

const fmt = (n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : `${Math.round(n)}`);

export function Explorer() {
  const [sel, setSel] = useState("All");
  const [hover, setHover] = useState<string | null>(null);
  const active = hover ?? sel;
  return (
    <div className="mt-4 grid grid-cols-[6.5rem_minmax(0,1fr)] items-center gap-4 md:grid-cols-[7rem_minmax(0,1fr)_minmax(0,1fr)]">
      <ul className="flex flex-col gap-1 text-[12px]" onMouseLeave={() => setHover(null)}>
        {[{ name: "All", m: 0, n: 0 }, ...SURVEYS].map((s) => {
          const on = active === s.name;
          const color = s.name === "All" ? "text-[var(--added)]" : "text-[var(--known)]";
          return (
            <li key={s.name}>
              <button
                type="button"
                onMouseEnter={() => setHover(s.name)}
                onFocus={() => setHover(s.name)}
                onBlur={() => setHover(null)}
                onClick={() => setSel(s.name)}
                title={s.n ? `${s.m} in one exposure, ${summed(s).toFixed(1)} summed over about ${fmt(s.n)} exposures` : "every survey together"}
                className={`flex w-full items-center rounded-md px-2 py-0.5 font-mono ${on ? `${color} bg-[var(--stripe)]` : "text-[var(--text-secondary)] hover:text-[var(--foreground)]"}`}
              >
                <span>{s.name}</span>
              </button>
            </li>
          );
        })}
      </ul>
      <Volume active={active} />
      <div className="col-span-2 md:col-span-1">
        <Reach active={active} />
      </div>
    </div>
  );
}
