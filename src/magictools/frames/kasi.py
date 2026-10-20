"""KASI's public, coordinate-indexed native image metadata."""
import json
import math

from .adapters import convert_rows, frame, url
from .model import mjd
from .parsers import numeric
from .transport import ArchiveError


def query_kasi(job, x, transport, s):
    w = job.window
    facility = "KMTNet" if job.provider == "kmtnet" else "MIRIS"
    # KASI searches image centres. Enclose the full camera, not just its centre.
    padding = 1.6 if facility == "KMTNet" else 3.0
    limit = 2000000
    endpoint = f"https://data.kasi.re.kr/api/{facility}/search"
    query = url(endpoint, dict(ra=w["ra"], dec=w["dec"], rad=min(180, w["radius"]+padding), unit="degree", limit=limit, datatype="OBJECT"))
    s["canSplitTime"] = False  # This API provides no documented date filter.
    # Avoid downloading the same whole time-series for every ephemeris month.
    raw = transport.cached_request(query) if hasattr(transport, "cached_request") else transport.request(query, auth={})
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ArchiveError("KASI response must be a JSON metadata array")
    s["received"] += len(rows)
    if len(rows) >= limit:
        s["incomplete"] = "KASI reached its requested row boundary; no documented offset or temporal pagination is available"
    s["warnings"].append("KASI returns centre-selected image candidates. Full-camera padding is conservative; inspect FITS WCS and chip gaps. API supplies no total-count or overflow marker.")
    s["indexExhausted"] = False
    s["partial"] = "KASI has no total-count/overflow marker; index exhaustion is unverified"
    def convert(r):
        miris = facility == "MIRIS"
        width, height = numeric(r.get("IMWIDTH")), numeric(r.get("IMHEIGHT"))
        fov = math.hypot(width, height)/2 if width and height else padding
        return frame(job, r, id=r.get("OBS-ID") if miris else r.get("DATAID"), survey=facility,
                     instrument=facility, startMjd=mjd(r.get("STOPEN") or r["DATE-OBS"]),
                     exposureSeconds=numeric(r.get("EXPTIME")), filter=r.get("OBS-FILT") if miris else r.get("FILTER"),
                     ra=numeric(r.get("RA-CNT") if miris else r.get("RA")),
                     dec=numeric(r.get("DEC-CNT") if miris else r.get("DEC")),
                     fovRadius=fov, url=r.get("DATAURL"), linkKind="FITS", productKind="image")
    yield from convert_rows(rows, convert, job, x, s)
