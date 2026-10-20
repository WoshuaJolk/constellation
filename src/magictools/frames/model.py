"""Input contract shared by the CLI and optional local HTTP interface."""
import copy
import csv
import io
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from .profiles import EXTRA_PROVIDERS

MOST_CATALOGS = ["ztf", "ptf", "wise_merge", "spitzer_bcd", "2mass", "sofia", "spherex"]
FIELD = ["field", "ephemeris"]
PROVIDERS = {
    "ssois": {"name": "CADC SSOIS", "modes": ["object", "orbits", "arc", "ephemeris"], "collections": ["SSOIS indexed instruments"]},
    "smoka": {"name": "SMOKA direct Subaru search", "modes": FIELD, "collections": ["HSC", "Suprime-Cam"]},
    "noirlab": {"name": "NOIRLab direct DECam search", "modes": FIELD, "collections": ["DECam", "DES", "DECaLS", "DEEP"]},
    "hsc": {"name": "HSC-SSP CAS", "modes": FIELD, "collections": ["HSC-SSP PDR3"], "credentials": ["HSC_SSP_CAS_USER", "HSC_SSP_CAS_PASSWORD"]},
    "cadc": {"name": "CADC ObsCore", "modes": FIELD, "collections": ["CFHT", "NEOSSat", "JCMT", "CADC collections"]},
    "eso": {"name": "ESO ObsCore", "modes": FIELD, "collections": ["VISTA", "VST", "VLT", "ESO collections"]},
    "irsa": {"name": "IRSA SIA2", "modes": FIELD, "collections": ["WISE", "NEOWISE", "Spitzer", "2MASS", "PTF", "ZTF", "IRAS", "AKARI", "IRSA collections"]},
    "mast": {"name": "MAST image archive", "modes": FIELD, "collections": ["HST", "JWST", "GALEX", "TESS", "Kepler", "K2", "MAST collections"]},
    "catch": {"name": "CATCH / PDS survey archive", "modes": ["field", "ephemeris", "object"], "collections": ["ATLAS", "Catalina", "LONEOS", "NEAT", "Pan-STARRS1 DR2", "SkyMapper DR4", "Spacewatch"]},
    "ztf": {"name": "ZTF IBE", "modes": FIELD, "collections": ["ZTF"]},
    **{f"most:{c}": {"name": f"IRSA MOST / {c}", "modes": ["object", "orbits"], "collections": [c]} for c in MOST_CATALOGS},
}
PROVIDERS.update(EXTRA_PROVIDERS)

LIMITATIONS = [
    "Completeness is relative to each queried index and its published releases, not all astronomical observations.",
    "Footprint and pointing-envelope matches are candidates, not detections; depth, masks, chip gaps, and access rights need inspection.",
    "SSOIS retains archive-defined time scales. Ephemeris rows use UTC MJD and the declared observer; no observer conversion is performed.",
    "Ephemerides are interpolated along great-circle segments. Sampling must resolve the target motion; no extrapolation or uncertainty probability is assumed.",
    "Products with different processing levels or detectors are retained separately, even when they share an exposure.",
]


def mjd(value):
    if isinstance(value, (int, float)):
        return finite(value, "MJD")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", str(value).strip()):
        return finite(value, "MJD")
    stamp = str(value).strip().replace("Z", "+00:00")
    # Python 3.10 accepts only 3/6 fractional digits; archives also emit
    # tenths, hundredths, and nanoseconds. Normalize to datetime precision.
    stamp = re.sub(r"(\d{2}:\d{2}:\d{2})[.,](\d+)",
                   lambda m: m[1] + "." + m[2].ljust(6, "0")[:6], stamp)
    dt = datetime.fromisoformat(stamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400 + 40587


def iso(value):
    return datetime.fromtimestamp((value - 40587) * 86400, timezone.utc).isoformat().replace("+00:00", "Z")


def finite(value, name, low=-math.inf, high=math.inf):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        v = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be numeric") from None
    if not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f"{name} must be finite, between {low} and {high}")
    return v


def sigma(value):
    s = str(value).strip()
    factors = {"u": 1e-6, "m": 1e-3, "d": 3600}
    return finite(s[:-1], "sigma", 0) * factors[s[-1]] if s[-1:] in factors else finite(s, "sigma", 0)


def sexagesimal(value, ra=False):
    s = str(value).strip()
    if ":" not in s and len(s.split()) == 1:
        return float(s)
    p = [float(t) for t in re.split(r"[:\s]+", s.lstrip("+-"))]
    if len(p) != 3 or not (0 <= p[1] < 60 and 0 <= p[2] < 60):
        raise ValueError("Expected three valid sexagesimal components")
    return (-1 if s.startswith("-") else 1) * (p[0] + p[1] / 60 + p[2] / 3600) * (15 if ra else 1)


def read_ephemeris(text):
    """JSON/CSV and find_orb's dated sexagesimal output, including m/u/d sigma."""
    if text.lstrip().startswith("["):
        return __import__("json").loads(text)
    lines = text.splitlines()
    if any("ra" in l.lower() and "," in l for l in lines[:3]):
        rows = list(csv.DictReader(io.StringIO(text)))
        return [{"mjd": mjd(r.get("mjd") or r.get("date")), "ra": float(r["ra"]),
                 "dec": float(r["dec"]), "sigmaArcsec": sigma(r.get("sigmaArcsec") or r.get("sigma") or "0"),
                 "label": r.get("label", "ephemeris")} for r in rows]
    rows = []
    pattern = re.compile(r"^(\d{4}) (\d{2}) (\d{2})(?:\s+(\d{2}):(\d{2})(?::([\d.]+))?|\s+(\d{2}))?\s+(\d+) (\d+) ([\d.]+)\s+([+-]\d+) (\d+) ([\d.]+)\s+(.*)$")
    for line in lines:
        if not re.match(r"^\d{4} \d{2} \d{2}", line):
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError("Unrecognized dated find_orb row; use CSV with date,ra,dec,sigmaArcsec")
        y, mo, d, hh, mm, ss, h, rh, rm, rs, dd, dm, ds, rest = match.groups()
        tail = rest.split()
        if len(tail) < 2:
            raise ValueError("find_orb row lacks the trailing sigma / PA columns")
        rows.append({"mjd": mjd(f"{y}-{mo}-{d}") + (int(hh or h or 0) * 3600 + int(mm or 0) * 60 + float(ss or 0)) / 86400,
                     "ra": sexagesimal(f"{rh}:{rm}:{rs}", True), "dec": sexagesimal(f"{dd}:{dm}:{ds}"),
                     "sigmaArcsec": sigma(tail[-2]), "label": "ephemeris"})
    if not rows:
        raise ValueError("No ephemeris rows found; use JSON, CSV, or dated find_orb output")
    return rows


def validate(value, base=Path(".")):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    x = copy.deepcopy(value)
    mode = x.get("mode")
    if mode not in {"field", "object", "orbits", "ephemeris", "arc"}:
        raise ValueError("mode must be field, object, orbits, ephemeris, or arc")
    for key in ("start", "end"):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(x.get(key, ""))):
            raise ValueError(f"{key} must be YYYY-MM-DD")
        mjd(x[key])
    if mjd(x["end"]) <= mjd(x["start"]):
        raise ValueError("end must be after start (UTC end exclusive)")
    x["timeoutSeconds"] = finite(x.get("timeoutSeconds", 600), "timeoutSeconds", 1, 86400)
    x["concurrency"] = int(finite(x.get("concurrency", 3), "concurrency", 1, 16))
    x["eellipse"] = finite(x.get("eellipse", 0), "eellipse", 0, 648000)
    x["sigmaMultiplier"] = finite(x.get("sigmaMultiplier", 1), "sigmaMultiplier", 0, 100)
    x["radius"] = finite(x.get("radius", 0), "radius", 0, 180)
    x["ephemStepDays"] = finite(x.get("ephemStepDays", 0.25), "ephemStepDays", 1e-6, 365)
    if mode == "field":
        x["ra"] = finite(x.get("ra"), "ra", 0, 360) % 360
        x["dec"] = finite(x.get("dec"), "dec", -90, 90)
    if mode == "object":
        if not isinstance(x.get("object"), str) or not x["object"].strip():
            raise ValueError("object designation is required")
        x["resolver"] = x.get("resolver", "bynameCADC")
        if x["resolver"] not in {"bynameCADC", "bynameMPC", "bynameHorizons", "bynameLowell"}:
            raise ValueError("Unknown SSOIS resolver")
    if mode == "arc":
        if "arcFile" in x:
            x["arc"] = (base / x.pop("arcFile")).read_text()
        lines = [line for line in x.get("arc", "").splitlines() if line.strip()]
        if len(lines) < 2 or any(len(line) != 80 for line in lines):
            raise ValueError("arc must contain at least two intact MPC80 observation lines")
        x["arc"] = "\n".join(lines) + "\n"
        x["arcFitter"] = x.get("arcFitter", "bern")
        if x["arcFitter"] not in {"bern", "mpc"}:
            raise ValueError("arcFitter must be bern or mpc (SSOIS wire search values)")
    if mode == "orbits":
        if not isinstance(x.get("orbits"), list) or not x["orbits"]:
            raise ValueError("orbits must be a nonempty array")
        labels = set()
        for o in x["orbits"]:
            if not isinstance(o, dict):
                raise ValueError("Each orbit must be an object")
            label = o.get("label")
            if not isinstance(label, str) or not label or label in labels:
                raise ValueError("Orbit labels must be nonempty and unique")
            labels.add(label)
            if "epochJd" in o:
                o["epochMjd"] = finite(o.pop("epochJd"), "epochJd") - 2400000.5
            o["epochMjd"] = finite(o.get("epochMjd"), "epochMjd", -1000000, 1000000)
            for k, hi in [("e", 1000), ("i", 180), ("node", 360), ("peri", 360)]:
                o[k] = finite(o.get(k), k, 0, hi)
            o["kind"] = o.get("kind", "cometary" if "q" in o else "elliptic")
            if o["kind"] == "cometary":
                o["q"] = finite(o.get("q"), "q (au)", 1e-9)
                o["tpJd"] = finite(o.get("tpJd"), "tpJd (JD, not MJD)", 1000000, 4000000)
            elif o["kind"] == "elliptic":
                if o["e"] >= 1:
                    raise ValueError("For e >= 1 provide kind=cometary, q, and tpJd")
                o["a"] = finite(o.get("a"), "a (au)", 1e-9)
                o["meanAnomaly"] = finite(o.get("meanAnomaly"), "meanAnomaly", 0, 360)
            else:
                raise ValueError("Orbit kind must be elliptic or cometary")
    if "ephemerisFile" in x:
        x["ephemeris"] = read_ephemeris((base / x.pop("ephemerisFile")).read_text())
    if mode == "ephemeris" or "ephemeris" in x:
        if not isinstance(x.get("ephemeris"), list) or len(x["ephemeris"]) < 2:
            raise ValueError("Provide at least two ephemeris rows")
        if not isinstance(x.get("observer"), str) or not x["observer"].strip():
            raise ValueError("Declare observer for the ephemeris, e.g. T09 or geocenter")
        groups = {}
        for p in x["ephemeris"]:
            if not isinstance(p, dict):
                raise ValueError("Each ephemeris row must be an object")
            p["mjd"] = mjd(p.get("mjd", p.get("date")))
            p["ra"] = finite(p.get("ra"), "ephemeris RA", 0, 360) % 360
            p["dec"] = finite(p.get("dec"), "ephemeris Dec", -90, 90)
            p["sigmaArcsec"] = finite(p.get("sigmaArcsec", 0), "sigmaArcsec", 0, 648000)
            p["label"] = str(p.get("label", "ephemeris"))
            groups.setdefault(p["label"], []).append(p)
        for points in groups.values():
            points.sort(key=lambda p: p["mjd"])
            if len(points) < 2 or any(a["mjd"] == b["mjd"] for a, b in zip(points, points[1:])):
                raise ValueError("Each track needs at least two unique epochs")
        x["ephemeris"] = [p for points in groups.values() for p in points]
    from .vo import validate_services
    services = validate_services(x.get("services", {}))
    if services:
        x["services"] = services
    requested = x.get("providers", "all")
    if requested == "all":
        requested = list(PROVIDERS) + list(services)
    if not isinstance(requested, list) or not requested:
        raise ValueError("providers must be 'all' or a nonempty list")
    expanded = [f"most:{c}" for c in x.get("mostCatalogs", MOST_CATALOGS)]
    requested = [p for v in requested for p in (expanded if v == "most" else [v])]
    if any(not isinstance(p, str) or p not in PROVIDERS and p not in services for p in requested):
        raise ValueError("Unknown provider; run `magictools frames capabilities`")
    x["providers"] = list(dict.fromkeys(requested))
    return x
