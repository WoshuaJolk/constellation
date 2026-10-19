"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { BODIES, COVER, FRACTION, GAIN_CAP, GRID, NB, NL, R0, R1, SURVEYS, rho, rmax, summed } from "@/lib/sky";

const GREEN = 0x1baf7a;
const BLUE = 0x2a78d6;
const RINGS = [10, 30, 100, 1000, 10000, 100000];
const RING_NAMES: Record<number, string> = { 30: "Neptune", 100000: "100k AU" };

const DEG = Math.PI / 180;
const pos = (lon: number, lat: number, r: number, out = new THREE.Vector3()) =>
  out.set(Math.cos(lat * DEG) * Math.cos(lon * DEG), Math.sin(lat * DEG), -Math.cos(lat * DEG) * Math.sin(lon * DEG)).multiplyScalar(r);

type Mode = "one" | "sum";
const depth = (k: number, mode: Mode) => (mode === "sum" ? summed(SURVEYS[k]) : SURVEYS[k].m);

function radii(size: number, set: number[], mode: Mode) {
  const out = new Float32Array(GRID.length);
  for (let n = 0; n < GRID.length; n++) {
    let best = 0;
    for (const k of set) if (COVER[k][n]) best = Math.max(best, rho(rmax(size, depth(k, mode))));
    out[n] = best;
  }
  return out;
}

function shell(color: number, opacity: number) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(GRID.length * 3), 3));
  const idx: number[] = [];
  for (let j = 0; j < NB; j++)
    for (let i = 0; i < NL; i++) {
      const a = j * (NL + 1) + i;
      idx.push(a, a + 1, a + NL + 1, a + 1, a + NL + 2, a + NL + 1);
    }
  geo.setIndex(idx);
  const mat = new THREE.MeshStandardMaterial({ color, opacity, transparent: true, side: THREE.DoubleSide, roughness: 0.9, metalness: 0 });
  return new THREE.Mesh(geo, mat);
}

function fill(mesh: THREE.Mesh, rad: Float32Array) {
  const p = mesh.geometry.getAttribute("position") as THREE.BufferAttribute;
  const v = new THREE.Vector3();
  for (let n = 0; n < GRID.length; n++) {
    pos(GRID[n].lon, GRID[n].lat, rad[n], v);
    p.setXYZ(n, v.x, v.y, v.z);
  }
  p.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
}

const fmt = (n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : `${Math.round(n)}`);

export function Volume() {
  const box = useRef<HTMLDivElement>(null);
  const labels = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [logSize, setLogSize] = useState(3);
  const [cut, setCut] = useState(true);
  const [mode, setMode] = useState<Mode>("sum");
  const state = useRef<{ union: THREE.Mesh; one: THREE.Mesh; dots: THREE.Mesh[]; plane: THREE.Plane; backs: THREE.Mesh[] } | null>(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.localClippingEnabled = true;
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 50);
    camera.position.set(1.3, 1.0, 1.7);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.2;
    controls.maxDistance = 6;
    scene.add(new THREE.HemisphereLight(0xffffff, 0xd6d5d0, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 0.8);
    sun.position.set(2, 3, 1);
    scene.add(sun);

    const plane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);
    const union = shell(GREEN, 0.5);
    const one = shell(BLUE, 0.6);
    one.visible = false;
    const back = (m: THREE.Mesh, color: number) => {
      const b = new THREE.Mesh(m.geometry, new THREE.MeshStandardMaterial({ color, opacity: 0.55, transparent: true, side: THREE.BackSide, roughness: 0.9 }));
      m.add(b);
      return b;
    };
    const unionBack = back(union, 0x0f7a54);
    const oneBack = back(one, 0x1b5aa6);
    (union.material as THREE.Material).side = THREE.FrontSide;
    (one.material as THREE.Material).side = THREE.FrontSide;
    scene.add(union, one);

    const ringMat = new THREE.LineBasicMaterial({ color: 0xc9c8c3 });
    for (const r of RINGS) {
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= 128; i++) pts.push(pos((360 * i) / 128, 0, rho(r)));
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
    }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(1, 128), new THREE.MeshBasicMaterial({ color: 0xe3e2de, opacity: 0.35, transparent: true, side: THREE.DoubleSide }));
    disc.rotation.x = -Math.PI / 2;
    scene.add(disc);
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 16), new THREE.MeshBasicMaterial({ color: 0x0b0b0b })));

    const dots = BODIES.map((b) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 16), new THREE.MeshBasicMaterial({ color: 0x52514e }));
      m.position.copy(pos(b.lon, b.lat, rho(b.r)));
      scene.add(m);
      return m;
    });
    state.current = { union, one, dots, plane, backs: [unionBack, oneBack] };

    const tags = Array.from(labels.current?.children ?? []) as HTMLElement[];
    const anchors = [...BODIES.map((b) => pos(b.lon, b.lat, rho(b.r))), ...RINGS.map((r) => pos(300, 0, rho(r)))];
    const v = new THREE.Vector3();
    const o = new THREE.Vector3();
    const resize = () => {
      const w = el.clientWidth;
      const h = Math.round(w * 0.62);
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    let raf = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      const w = el.clientWidth;
      const h = renderer.domElement.clientHeight;
      o.set(0, 0, 0).project(camera);
      const ox = ((o.x + 1) / 2) * w;
      const oy = ((1 - o.y) / 2) * h;
      anchors.forEach((a, i) => {
        const t = tags[i];
        if (!t) return;
        v.copy(a).project(camera);
        let x = ((v.x + 1) / 2) * w;
        let y = ((1 - v.y) / 2) * h;
        if (i < BODIES.length) {
          const dx = x - ox;
          const dy = y - oy;
          const n = Math.hypot(dx, dy) || 1;
          x += (dx / n) * 10;
          y += (dy / n) * 10;
          t.style.textAlign = dx < 0 ? "right" : "left";
          t.style.transform = `translate(${dx < 0 ? x - t.offsetWidth : x}px, ${y}px)`;
        } else t.style.transform = `translate(${x}px, ${y}px)`;
        t.style.opacity = v.z < 1 ? "1" : "0";
      });
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      state.current = null;
    };
  }, []);

  useEffect(() => {
    const s = state.current;
    if (!s) return;
    const size = 10 ** logSize;
    fill(s.union, radii(size, SURVEYS.map((_, k) => k), mode));
    const k = SURVEYS.findIndex((x) => x.name === hover);
    s.one.visible = k >= 0;
    if (k >= 0) fill(s.one, radii(size, [k], mode));
    (s.union.material as THREE.MeshStandardMaterial).opacity = k >= 0 ? 0.18 : 0.5;
    (s.backs[0].material as THREE.MeshStandardMaterial).opacity = k >= 0 ? 0.15 : 0.55;
    const set = k >= 0 ? [k] : SURVEYS.map((_, i) => i);
    BODIES.forEach((b, i) => {
      const n = GRID.findIndex((d) => Math.abs(d.lon - b.lon) <= 180 / NL && Math.abs(d.lat - b.lat) <= 90 / NB);
      const inside = set.some((j) => COVER[j][n] && rmax(b.d, depth(j, mode)) >= b.r);
      (s.dots[i].material as THREE.MeshBasicMaterial).color.set(inside ? (k >= 0 ? BLUE : GREEN) : 0xfcfcfb);
    });
    for (const m of [s.union, s.one, ...s.backs]) (m.material as THREE.Material).clippingPlanes = cut ? [s.plane] : [];
  }, [hover, logSize, cut, mode]);

  const size = 10 ** logSize;
  return (
    <section className="mt-8 max-w-[820px]">
      <h2 className="text-base font-semibold tracking-tight">The searched volume</h2>
      <p className="mt-1 text-sm leading-6 text-[var(--text-secondary)]">
        How far from the Sun a body of one size leaves a detection, in every direction. One exposure is the 5 sigma depth
        of a single frame. Summed along the track adds 1.25 log10 N for the N exposures a survey has of each sky position,
        capped at +{GAIN_CAP} mag where systematics take over. Radius is logarithmic from {R0} to {fmt(R1)} AU. The green
        shell is the union; where it dips to the centre, nothing public has looked. Footprints and visit counts are
        typical values until the exposure index exists.
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-[var(--text-secondary)]">
        <label className="flex items-center gap-2">
          <span className="text-[var(--text-muted)]">body</span>
          <input type="range" min={0} max={5} step={0.05} value={logSize} onChange={(e) => setLogSize(Number(e.target.value))} className="w-40 accent-[var(--foreground)]" />
          <span className="w-16 font-mono tabular-nums text-[var(--foreground)]">{fmt(size)} km</span>
        </label>
        <span className="flex items-center gap-2">
          {(["one", "sum"] as Mode[]).map((x) => (
            <button
              key={x}
              type="button"
              onClick={() => setMode(x)}
              className={`rounded-md border px-2 py-0.5 ${mode === x ? "border-[var(--foreground)] text-[var(--foreground)]" : "border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--foreground)]"}`}
            >
              {x === "one" ? "one exposure" : "summed along the track"}
            </button>
          ))}
        </span>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={cut} onChange={(e) => setCut(e.target.checked)} className="accent-[var(--foreground)]" />
          cut at the ecliptic
        </label>
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full border-[1.5px] border-[var(--text-secondary)] align-[-1px]" />example bodies, filled when inside at their own size</span>
      </div>
      <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[11px]" onMouseLeave={() => setHover(null)}>
        <span className="text-[var(--text-muted)]">{mode === "one" ? "one exposure of" : "summed"}</span>
        {SURVEYS.map((s, k) => (
          <button
            key={s.name}
            type="button"
            onMouseEnter={() => setHover(s.name)}
            onFocus={() => setHover(s.name)}
            onBlur={() => setHover(null)}
            onClick={() => setHover(hover === s.name ? null : s.name)}
            title={`${s.m} in one exposure, ${summed(s).toFixed(1)} summed over about ${fmt(s.n)} exposures; ${Math.round(FRACTION[k].sky * 100)}% of the sky, ${Math.round(FRACTION[k].ecliptic * 100)}% of the ecliptic`}
            className={`font-mono tabular-nums ${hover === s.name ? "text-[var(--known)]" : "text-[var(--text-secondary)] hover:text-[var(--foreground)]"}`}
          >
            {s.name} <span className="text-[var(--text-muted)]">{depth(k, mode).toFixed(1).replace(/\.0$/, "")} · {Math.round(FRACTION[k].sky * 100)}%</span>
          </button>
        ))}
      </div>
      <div ref={box} className="relative mt-2 w-full cursor-grab overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
        <div ref={labels} className="pointer-events-none absolute inset-0">
          {BODIES.map((b) => (
            <span key={b.name} className="absolute left-0 top-0 whitespace-nowrap text-[10px] leading-none text-[var(--text-secondary)]" style={{ marginTop: -5 }}>
              {b.name}
            </span>
          ))}
          {RINGS.map((r) => (
            <span key={r} className="absolute left-0 top-0 whitespace-nowrap font-mono text-[9.5px] leading-none text-[var(--text-muted)]" style={{ marginTop: -12, marginLeft: -4 }}>
              {RING_NAMES[r] ?? `${fmt(r)}`}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
