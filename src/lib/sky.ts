const DEG = Math.PI / 180;
const EPS = 23.439 * DEG;
const G = [
  [-0.0548755604, -0.8734370902, -0.4838350155],
  [0.4941094279, -0.44482963, 0.7469822445],
  [-0.867666149, -0.1980763734, 0.4559837762],
];

export type Dir = { ra: number; dec: number; l: number; b: number; lon: number; lat: number };

export function fromEcliptic(lon: number, lat: number): Dir {
  const x = Math.cos(lat * DEG) * Math.cos(lon * DEG);
  const y = Math.cos(lat * DEG) * Math.sin(lon * DEG);
  const z = Math.sin(lat * DEG);
  const ye = y * Math.cos(EPS) - z * Math.sin(EPS);
  const ze = y * Math.sin(EPS) + z * Math.cos(EPS);
  const ra = ((Math.atan2(ye, x) / DEG) + 360) % 360;
  const dec = Math.asin(ze) / DEG;
  const gx = G[0][0] * x + G[0][1] * ye + G[0][2] * ze;
  const gy = G[1][0] * x + G[1][1] * ye + G[1][2] * ze;
  const gz = G[2][0] * x + G[2][1] * ye + G[2][2] * ze;
  return { ra, dec, l: ((Math.atan2(gy, gx) / DEG) + 360) % 360, b: Math.asin(gz) / DEG, lon, lat };
}

export function toEcliptic(ra: number, dec: number): { lon: number; lat: number } {
  const x = Math.cos(dec * DEG) * Math.cos(ra * DEG);
  const y = Math.cos(dec * DEG) * Math.sin(ra * DEG);
  const z = Math.sin(dec * DEG);
  const ye = y * Math.cos(EPS) + z * Math.sin(EPS);
  const ze = -y * Math.sin(EPS) + z * Math.cos(EPS);
  return { lon: ((Math.atan2(ye, x) / DEG) + 360) % 360, lat: Math.asin(ze) / DEG };
}

const inRa = (ra: number, a: number, b: number) => (a <= b ? ra >= a && ra <= b : ra >= a || ra <= b);
const box = (d: Dir, a: number, b: number, d0: number, d1: number) => inRa(d.ra, a, b) && d.dec >= d0 && d.dec <= d1;

// Single-exposure 5 sigma depth, typical exposures per sky position, and an approximate footprint.
export const SURVEYS: { name: string; m: number; n: number; cover: (d: Dir) => boolean }[] = [
  { name: "TESS", m: 17, n: 3000, cover: () => true },
  { name: "ZTF", m: 20.5, n: 1500, cover: (d) => d.dec > -31 },
  { name: "Catalina", m: 21.5, n: 1000, cover: (d) => d.dec > -35 && d.dec < 75 },
  { name: "Pan-STARRS", m: 22, n: 60, cover: (d) => d.dec > -30 },
  { name: "SDSS", m: 22.5, n: 1, cover: (d) => (d.b > 30 && d.dec > -10 && d.dec < 70) || box(d, 310, 60, -1.3, 1.3) },
  { name: "DECam", m: 23.5, n: 20, cover: (d) => d.dec < 32 && (Math.abs(d.b) > 10 || d.dec < -20) },
  { name: "OmegaCAM", m: 24, n: 20, cover: (d) => (d.dec > -40 && d.dec < -10 && Math.abs(d.b) > 20) || box(d, 128, 238, -5, 4) || (Math.abs(d.b) < 5 && d.dec < 0) },
  { name: "MegaCam", m: 24.5, n: 6, cover: (d) => (d.dec > 30 && d.dec < 80 && Math.abs(d.b) > 25) || box(d, 30, 40, -7, -3) || box(d, 208, 221, 51, 58) },
  { name: "HSC", m: 26, n: 30, cover: (d) => box(d, 128, 227, -2, 5) || box(d, 330, 40, -7, 7) || box(d, 200, 250, 42.5, 44.5) },
];

export const GAIN_CAP = 4; // mag; beyond this flat-field and background systematics set the floor
export const summed = (s: { m: number; n: number }) => s.m + Math.min(1.25 * Math.log10(s.n), GAIN_CAP);

const ALBEDO = 0.1;
export const R0 = 2;
export const R1 = 100000;

export function rmax(d: number, m: number) {
  const h = 5 * Math.log10(1329 / (Math.sqrt(ALBEDO) * d));
  const x = 10 ** ((m - h) / 5);
  return (1 + Math.sqrt(1 + 4 * x)) / 2;
}

export const rho = (r: number) => Math.min(Math.max(Math.log10(r / R0) / Math.log10(R1 / R0), 0), 1);

export const NL = 180;
export const NB = 90;

export const GRID: Dir[] = (() => {
  const g: Dir[] = [];
  for (let j = 0; j <= NB; j++) {
    const lat = -90 + (180 * j) / NB;
    for (let i = 0; i <= NL; i++) g.push(fromEcliptic((360 * i) / NL, lat));
  }
  return g;
})();

export const COVER: boolean[][] = SURVEYS.map((s) => GRID.map((d) => s.cover(d)));

export const FRACTION = SURVEYS.map((_, k) => {
  let all = 0, hit = 0, ecl = 0, eclHit = 0;
  GRID.forEach((d, n) => {
    const w = Math.cos(d.lat * (Math.PI / 180));
    all += w;
    if (COVER[k][n]) hit += w;
    if (Math.abs(d.lat) < 5) {
      ecl += 1;
      if (COVER[k][n]) eclHit += 1;
    }
  });
  return { sky: hit / all, ecliptic: eclHit / ecl };
});

export const BODIES: { name: string; lon: number; lat: number; r: number; d: number }[] = [
  { name: "100 km Kuiper belt object", lon: 200, lat: 0, r: 42, d: 100 },
  { name: "Sedna now, 85 AU", ...toEcliptic(55, 8), r: 85, d: 1000 },
  { name: "Planet Nine, one proposed direction", ...toEcliptic(50, 20), r: 500, d: 25000 },
  { name: "500 km inner Oort body", lon: 75, lat: -25, r: 3000, d: 500 },
  { name: "Earth-mass body in the Oort cloud", lon: 300, lat: 40, r: 30000, d: 12700 },
];
