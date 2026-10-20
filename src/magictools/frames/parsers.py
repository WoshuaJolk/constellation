"""Tolerant row parsing, strict table-level status, and native identity."""
import csv
import html
import io
import math
import re
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from urllib.parse import urlsplit


def numeric(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def safe_url(value):
    if not value:
        return None
    u = urlsplit(str(value))
    return str(value) if u.scheme in {"http", "https"} and u.netloc and not u.username and not u.password else None


def delimited(text, stats, delimiter=","):
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    header = next(reader, [])
    header = [s.strip().lstrip("\ufeff") for s in header]
    if not header or header[0].startswith("<"):
        raise ValueError("Archive did not return a delimited table")
    stats["columns"] = header
    for row in reader:
        if not row or not any(s.strip() for s in row):
            continue
        stats["received"] += 1
        if len(row) != len(header):
            stats["rejected"] += 1
            continue
        yield dict(zip(header, (s.strip() for s in row)))


def ipac(text, stats, ibe=False, *, generic=False):
    if ibe and not re.search(r"QUERY_STATUS\s*=\s*['\"]OK['\"]", text):
        raise ValueError("IRSA did not report QUERY_STATUS=OK")
    if not ibe and not generic and "MOST ODBC query result" not in text:
        raise ValueError("MOST did not return its result table")
    lines = text.splitlines()
    index = next((i for i, l in enumerate(lines) if l.startswith("|")), None)
    if index is None:
        raise ValueError("No IPAC table header")
    positions = [m.start() for m in re.finditer(r"\|", lines[index])]
    names = [lines[index][a+1:b].strip() for a, b in zip(positions, positions[1:])]
    stats["columns"] = names
    for line in lines[index+1:]:
        if not line.strip() or line.startswith(("|", "\\")):
            continue
        stats["received"] += 1
        if len(line) < positions[-1]:
            stats["rejected"] += 1
            continue
        yield {name: line[a+1:b].strip() for name, a, b in zip(names, positions, positions[1:])}
    expected = re.search(r"RowsRetrieved\s*=\s*(\d+)", text)
    if ibe and (not expected or int(expected[1]) != stats["received"]):
        stats["incomplete"] = "IRSA row count is missing or inconsistent with the response"


def votable(text, stats):
    root = ET.fromstring(text)
    local = lambda e: e.tag.rsplit("}", 1)[-1]
    infos = [e for e in root.iter() if local(e) == "INFO" and e.get("name", "").upper() == "QUERY_STATUS"]
    stats.setdefault("queryStatuses", []).extend(info.get("value", "").upper() for info in infos)
    if not infos:
        stats["missingQueryStatusResponses"] = stats.get("missingQueryStatusResponses", 0) + 1
    for info in infos:
        if info.get("value", "").upper() == "ERROR":
            raise ValueError("VO query error: " + "".join(info.itertext()).strip()[:500])
        if info.get("value", "").upper() == "OVERFLOW":
            stats["incomplete"] = "Archive reports QUERY_STATUS=OVERFLOW"
    table = next((e for e in root.iter() if local(e) == "TABLE"), None)
    if table is None:
        raise ValueError("No VOTable table")
    fields = [e.get("name") or e.get("ID") for e in table if local(e) == "FIELD"]
    stats["columns"] = fields
    stats["fields"] = {e.get("name") or e.get("ID"): dict(e.attrib) for e in table if local(e) == "FIELD"}
    if any(local(e) in {"BINARY", "BINARY2", "FITS"} for e in table.iter()):
        # Astropy supports binary serializations, including services ignoring FORMAT.
        from astropy.io.votable import parse_single_table
        t = parse_single_table(io.BytesIO(text.encode())).to_table()
        for record in t:
            stats["received"] += 1
            def cell(value):
                import numpy as np
                mask = np.asarray(getattr(value, "mask", False))
                if mask.all():
                    return ""
                if np.ndim(value):
                    return " ".join("" if m else str(v) for v, m in zip(np.asarray(value).flat, np.broadcast_to(mask, np.shape(value)).flat))
                return value.decode() if isinstance(value, bytes) else str(value)
            yield {k: cell(record[k]) for k in t.colnames}
        return
    for row in table.iter():
        if local(row) != "TR":
            continue
        stats["received"] += 1
        values = ["".join(e.itertext()) for e in row if local(e) == "TD"]
        if len(values) != len(fields):
            stats["rejected"] += 1
            continue
        yield dict(zip(fields, values))


def smoka(text, stats):
    if re.search(r"No matching frame", text, re.I):
        stats["columns"] = ["FRAMEID"]
        return
    match = re.search(r"<pre[^>]*>(.*?)</pre>", text, re.S | re.I)
    clean = html.unescape(match[1] if match else text)
    lines = clean.strip().splitlines()
    if not lines or not lines[0].strip().startswith("FRAMEID"):
        raise ValueError("SMOKA did not return a frame table")
    names = lines[0].split()
    stats["columns"] = names
    for line in lines[1:]:
        if not line.strip():
            continue
        stats["received"] += 1
        values = line.split()
        if len(values) != len(names):
            stats["rejected"] += 1
            continue
        yield dict(zip(names, values))
    if stats["received"] >= 20000:
        stats["incomplete"] = "SMOKA 20,000-row service boundary"


def instrument(value):
    s = re.sub(r"[^a-z0-9]", "", str(value).lower())
    for fragments, name in [
        (["megacam", "megaprime"], "CFHT/MegaCam"), (["decam"], "DECam"),
        (["hyper", "hsc"], "Subaru/HSC"), (["suprime", "supcam"], "Subaru/Suprime-Cam"),
        (["ztf"], "ZTF"), (["panstarrs", "ps1"], "Pan-STARRS1"),
        (["sdss"], "SDSS"), (["skymapper"], "SkyMapper"),
    ]:
        if any(f in s for f in fragments):
            return name
    return str(value).strip() or "unknown"


def canonical(frame):
    inst = instrument(frame["instrument"] or frame["survey"])
    native = str(frame["id"]).strip()
    url = safe_url(frame.get("url"))
    basename = PurePosixPath(urlsplit(url).path).name if url else ""
    # Exact filenames are much stronger identities than pointing/time proximity.
    if re.match(r"ztf_\d{14}_", basename) or re.match(r"frame-[ugriz]-", basename):
        native = basename
    native = re.sub(r"\.(fits|fit)(\.(fz|gz|bz2))?$", "", native, flags=re.I)
    if inst == "CFHT/MegaCam":
        native = re.sub(r"^(\d+)[op]$", r"\1", native)
    if inst == "DECam":
        native = re.sub(r"\.(fz|gz)$", "", native)
    return f"{inst.lower()}:{native.lower()}"


def merge_frames(a, b):
    a["hypotheses"] = sorted(set(a["hypotheses"] + b["hypotheses"]))
    for field in ("provenance", "products", "predictions"):
        a.setdefault(field, [])
        for item in b.get(field, []):
            if item not in a[field]:
                a[field].append(item)
    return a
