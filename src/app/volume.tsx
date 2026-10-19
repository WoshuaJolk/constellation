"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { COVER, GRID, NB, NL, R0, R1, SURVEYS, rho, rmax, summed, toEcliptic } from "@/lib/sky";

const GREEN = 0x1baf7a;
const BLUE = 0x2a78d6;
const SIZE = 25000; // km, a Planet Nine of a few Earth masses
const RINGS = [10, 30, 100, 1000, 10000, 100000];
const RING_NAMES: Record<number, string> = { 30: "Neptune", 100000: "100k AU" };
const P9 = { ...toEcliptic(50, 20), r: 500 };

const DEG = Math.PI / 180;
const pos = (lon: number, lat: number, r: number, out = new THREE.Vector3()) =>
  out.set(Math.cos(lat * DEG) * Math.cos(lon * DEG), Math.sin(lat * DEG), -Math.cos(lat * DEG) * Math.sin(lon * DEG)).multiplyScalar(r);

function radii(set: number[]) {
  const out = new Float32Array(GRID.length);
  for (let n = 0; n < GRID.length; n++) {
    let best = 0;
    for (const k of set) if (COVER[k][n]) best = Math.max(best, rho(rmax(SIZE, summed(SURVEYS[k]))));
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
  const mat = new THREE.MeshStandardMaterial({ color, opacity, transparent: true, side: THREE.DoubleSide, roughness: 0.9, metalness: 0, depthWrite: false });
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
  const [sel, setSel] = useState("All");
  const [hover, setHover] = useState<string | null>(null);
  const active = hover ?? sel;
  const state = useRef<{ union: THREE.Mesh; one: THREE.Mesh } | null>(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 50);
    camera.position.set(1.3, 0.9, 1.7);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.2;
    controls.maxDistance = 6;
    scene.add(new THREE.HemisphereLight(0xffffff, 0xd6d5d0, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 0.8);
    sun.position.set(2, 3, 1);
    scene.add(sun);

    const ringMat = new THREE.LineBasicMaterial({ color: 0xc9c8c3 });
    for (const r of RINGS) {
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= 128; i++) pts.push(pos((360 * i) / 128, 0, rho(r)));
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
    }
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 16), new THREE.MeshBasicMaterial({ color: 0x0b0b0b })));
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 16), new THREE.MeshBasicMaterial({ color: 0x0b0b0b }));
    dot.position.copy(pos(P9.lon, P9.lat, rho(P9.r)));
    scene.add(dot);

    const union = shell(GREEN, 0.28);
    const one = shell(BLUE, 0.4);
    one.visible = false;
    scene.add(union, one);
    fill(union, radii(SURVEYS.map((_, k) => k)));
    state.current = { union, one };

    const tags = Array.from(labels.current?.children ?? []) as HTMLElement[];
    const anchors = [dot.position.clone(), ...RINGS.map((r) => pos(300, 0, rho(r)))];
    const v = new THREE.Vector3();
    const resize = () => {
      const w = el.clientWidth;
      const h = Math.round(w * 0.7);
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
      anchors.forEach((a, i) => {
        const t = tags[i];
        if (!t) return;
        v.copy(a).project(camera);
        t.style.transform = `translate(${((v.x + 1) / 2) * w}px, ${((1 - v.y) / 2) * h}px)`;
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
    const k = SURVEYS.findIndex((x) => x.name === active);
    s.one.visible = k >= 0;
    if (k >= 0) fill(s.one, radii([k]));
    (s.union.material as THREE.MeshStandardMaterial).opacity = k >= 0 ? 0.1 : 0.28;
  }, [active]);

  return (
    <section className="mt-8 max-w-[820px]">
      <h2 className="text-base font-semibold tracking-tight">The searched volume</h2>
      <p className="mt-1 text-sm leading-6 text-[var(--text-secondary)]">
        How far from the Sun a {fmt(SIZE)} km body, a Planet Nine, leaves a detection in every direction when each
        survey&apos;s exposures of that sky are summed along the track. Radius is logarithmic from {R0} to {fmt(R1)} AU.
        Where the shell dips to the centre, nothing public has looked. Footprints and visit counts are typical values until
        the exposure index exists.
      </p>
      <div className="mt-3 grid grid-cols-[minmax(0,1fr)_8rem] gap-4">
        <div ref={box} className="relative w-full cursor-grab overflow-hidden">
          <div ref={labels} className="pointer-events-none absolute inset-0">
            <span className="absolute left-0 top-0 whitespace-nowrap pl-2 text-[10px] leading-none text-[var(--text-secondary)]" style={{ marginTop: -5 }}>
              Planet Nine, one proposed direction
            </span>
            {RINGS.map((r) => (
              <span key={r} className="absolute left-0 top-0 whitespace-nowrap font-mono text-[9.5px] leading-none text-[var(--text-muted)]" style={{ marginTop: -12, marginLeft: -4 }}>
                {RING_NAMES[r] ?? `${fmt(r)}`}
              </span>
            ))}
          </div>
        </div>
        <ul className="flex flex-col gap-1 pt-2 text-[12px]" onMouseLeave={() => setHover(null)}>
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
                  className={`flex w-full items-center justify-between rounded-md px-2 py-0.5 font-mono tabular-nums ${on ? `${color} bg-[var(--stripe)]` : "text-[var(--text-secondary)] hover:text-[var(--foreground)]"}`}
                >
                  <span>{s.name}</span>
                  <span className="text-[var(--text-muted)]">{s.n ? summed(s).toFixed(1) : ""}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
