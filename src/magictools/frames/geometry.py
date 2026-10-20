"""Spherical geometry for exposure-time matching; no orbital propagation."""
import bisect
import math
from .model import mjd

RAD = math.pi / 180


def vector(ra, dec):
    return (math.cos(dec * RAD) * math.cos(ra * RAD),
            math.cos(dec * RAD) * math.sin(ra * RAD), math.sin(dec * RAD))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def angle(a, b):
    return math.atan2(norm(cross(a, b)), dot(a, b))


def distance(ra, dec, ra2, dec2):
    return angle(vector(ra, dec), vector(ra2, dec2)) / RAD


def coords(v):
    return math.atan2(v[1], v[0]) / RAD % 360, math.atan2(v[2], math.hypot(v[0], v[1])) / RAD


def interpolate(points, when):
    times = [p["mjd"] for p in points]
    if when < times[0] or when > times[-1]:
        return None
    index = min(max(bisect.bisect_right(times, when) - 1, 0), len(points) - 2)
    a, b = points[index:index+2]
    fraction = (when - a["mjd"]) / (b["mjd"] - a["mjd"])
    va, vb = vector(a["ra"], a["dec"]), vector(b["ra"], b["dec"])
    theta = angle(va, vb)
    if theta > math.pi - 1e-5:
        raise ValueError("Antipodal ephemeris samples do not define a unique track; add intermediate samples")
    if theta < 1e-12:
        ra, dec = a["ra"], a["dec"]
    else:
        weights = (math.sin((1-fraction)*theta)/math.sin(theta), math.sin(fraction*theta)/math.sin(theta))
        ra, dec = coords(tuple(weights[0]*va[i]+weights[1]*vb[i] for i in range(3)))
    # Sigma is a supplied angular envelope, linearly interpolated, not a probability.
    return {"ra": ra, "dec": dec, "sigmaArcsec": a["sigmaArcsec"] * (1-fraction) + b["sigmaArcsec"] * fraction}


def windows(x):
    """Monthly conservative bounding cones, with explicit track and epoch interval."""
    if "ephemeris" not in x:
        if x["mode"] == "field":
            yield {"ra": x["ra"], "dec": x["dec"], "radius": x["radius"],
                   "start": mjd(x["start"]), "end": mjd(x["end"]), "label": "field"}
        return
    groups = {}
    for p in x["ephemeris"]:
        groups.setdefault(p["label"], []).append(p)
    for label, points in groups.items():
        start, end = max(mjd(x["start"]), points[0]["mjd"]), min(mjd(x["end"]), points[-1]["mjd"])
        while start < end:
            stop = min(start + 30, end)
            samples = [interpolate(points, start), *[p for p in points if start < p["mjd"] < stop], interpolate(points, stop)]
            va = [vector(p["ra"], p["dec"]) for p in samples]
            center = tuple(sum(v[i] for v in va) for i in range(3))
            ra, dec = coords(center) if norm(center) > 1e-10 else (samples[0]["ra"], samples[0]["dec"])
            # Triangle inequality covers each great-circle segment, even if the
            # farthest point on it is not a supplied sample.
            padding = max(angle(a, b) / RAD for a, b in zip(va, va[1:])) / 2
            radius = min(180, max(distance(ra, dec, p["ra"], p["dec"]) for p in samples)
                         + padding + max(p["sigmaArcsec"] for p in samples) * x["sigmaMultiplier"] / 3600 + x["eellipse"] / 3600 + x["radius"])
            local_points = [dict(samples[0], mjd=start), *[p for p in points if start < p["mjd"] < stop], dict(samples[-1], mjd=stop)]
            yield {"ra": ra, "dec": dec, "radius": radius, "start": start, "end": stop, "label": label, "points": local_points}
            start = stop


def polygon_intersects(ra, dec, radius, corners):
    """Circle vs minor great-circle edges of a convex, sub-hemisphere footprint."""
    p, vertices = vector(ra, dec), [vector(*v) for v in corners]
    center = tuple(sum(v[i] for v in vertices) for i in range(3))
    if norm(center) < 1e-8:
        return None  # No defensible convex interior; retain as unresolved candidate.
    inside = True
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        n = cross(a, b)
        if norm(n) < 1e-12:
            continue
        n = tuple(v / norm(n) for v in n)
        if dot(p, n) * dot(center, n) < -1e-12:
            inside = False
        if min(angle(p, a), angle(p, b)) <= radius * RAD + 1e-12:
            return True
        projection = tuple(p[i] - dot(p, n) * n[i] for i in range(3))
        if norm(projection) > 1e-12:
            projection = tuple(v / norm(projection) for v in projection)
            if abs(angle(a, projection) + angle(projection, b) - angle(a, b)) < 1e-8 and angle(p, projection) <= radius * RAD + 1e-12:
                return True
    return inside


def region_intersects(region, ra, dec, radius):
    if not region:
        return None
    tokens = region.strip().split()
    if not tokens or tokens[0].upper() not in {"POLYGON", "CIRCLE"}:
        return None
    try:
        values = [float(t) for t in tokens[1:] if t.upper() not in {"ICRS", "FK5", "J2000"}]
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    if tokens[0].upper() == "CIRCLE" and len(values) == 3:
        return distance(ra, dec, *values[:2]) <= radius + values[2]
    if len(values) >= 6 and len(values) % 2 == 0:
        return polygon_intersects(ra, dec, radius, list(zip(values[::2], values[1::2])))
    return None


def select(frame, window, x):
    """False: definitely outside. Unknown geometry remains visibly a candidate."""
    when = frame.get("startMjd")
    if when is not None:
        stop = frame.get("endMjd")
        if stop is None:
            stop = when + (frame.get("exposureSeconds") or 0)/86400
        if stop < window["start"] or when >= window["end"]:
            return False
    if "points" in window:
        if when is None or frame["timeScale"] != "UTC":
            frame["match"] = "candidate: observation time unavailable in UTC"
            return True
        # Use exposure midpoint only for exposure products, not broad mosaics.
        if frame.get("endMjd") and frame["endMjd"] - when > 1:
            frame["match"] = "candidate: combined product spans multiple dates"
            return True
        when += (frame.get("exposureSeconds") or 0) / 172800
        predicted = interpolate(window["points"], when)
        if predicted is None:
            frame["match"] = "candidate: exposure midpoint outside ephemeris"
            return True
        ra, dec = predicted["ra"], predicted["dec"]
        radius = x["radius"] + (predicted["sigmaArcsec"] * x["sigmaMultiplier"] + x["eellipse"]) / 3600
        frame["predictions"] = [{"hypothesis": window["label"], "mjdUtc": when, **predicted, "radiusDegrees": radius, "observer": x["observer"]}]
    else:
        ra, dec, radius = window["ra"], window["dec"], window["radius"]
    hit = region_intersects(frame.get("footprint"), ra, dec, radius)
    if hit is not None:
        frame["match"] = "exposure-time footprint intersection" if "points" in window else "footprint intersection"
        return hit
    if frame.get("fovRadius") is not None and frame.get("ra") is not None and frame.get("dec") is not None:
        frame["match"] = "pointing-envelope candidate; detector footprint unverified"
        return distance(ra, dec, frame["ra"], frame["dec"]) <= radius + frame["fovRadius"]
    frame["match"] = "candidate: footprint unavailable"
    return True
