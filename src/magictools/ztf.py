"""Bounded, explicit access to public ZTF science exposures through IRSA IBE."""

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
import numpy as np

from .io import _fields, _provenance, read_json
from .photometry import validate_psf_kernel

BASE = "https://irsa.ipac.caltech.edu/ibe"


def _download(url, limit):
    if limit <= 0:
        raise ValueError("Archive download exhausted max_total_bytes")
    request = Request(url, headers={"User-Agent": "constellation (public astronomical research)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read(limit + 1)
            break
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(0.5 * 2**attempt)
    if len(body) > limit:
        raise ValueError("Archive response exceeded the explicit byte limit; narrow the request")
    return body


def _position(ra, dec):
    if not np.all(np.isfinite([ra, dec])) or not 0 <= ra < 360 or not -90 <= dec <= 90:
        raise ValueError("RA must be in [0,360), Dec in [-90,90]")


def product_url(row, suffix):
    if suffix not in {"sciimg.fits", "mskimg.fits", "sciimgdaopsfcent.fits"}:
        raise ValueError("Unsupported ZTF product")
    frac = str(row["filefracday"])
    if not re.fullmatch(r"\d{14}", frac) or row["filtercode"] not in {"zg", "zr", "zi"}:
        raise ValueError("Invalid ZTF product metadata")
    field, ccd, quad = (int(row[k]) for k in ["field", "ccdid", "qid"])
    if not 0 <= field <= 999999 or not 1 <= ccd <= 16 or not 1 <= quad <= 4:
        raise ValueError("Invalid ZTF field/CCD/quadrant")
    name = f"ztf_{frac}_{field:06d}_{row['filtercode']}_c{ccd:02d}_o_q{quad}_{suffix}"
    return f"{BASE}/data/ztf/products/sci/{frac[:4]}/{frac[4:8]}/{frac[8:]}/{name}"


def parse_index(body):
    text = body.decode("utf-8")
    status = re.search(r"QUERY_STATUS\s*=\s*['\"]([^'\"]+)", text)
    if status is None or status.group(1) != "OK":
        raise ValueError("Archive query did not report QUERY_STATUS=OK (possibly truncated)")
    table = Table.read(text, format="ascii.ipac")
    count = re.search(r"RowsRetrieved\s*=\s*(\d+)", text)
    if count is None or int(count.group(1)) != len(table):
        raise ValueError("Archive row count is absent or inconsistent")
    required = {"pid", "obsjd", "exptime", "filefracday", "field", "ccdid", "qid", "filtercode", "infobits"}
    if not required <= set(table.colnames):
        raise ValueError("Archive response lacks required exposure metadata")
    rows = []
    for row in table:
        converted = {}
        for key in table.colnames:
            value = row[key]
            if np.ma.is_masked(value):
                converted[key] = None
            else:
                value = value.item() if isinstance(value, np.generic) else value
                converted[key] = None if isinstance(value, float) and not np.isfinite(value) else value
        if any(converted[key] is None for key in required):
            raise ValueError("Archive response contains missing required metadata")
        converted["pid"] = str(converted["pid"])
        product_url(converted, "sciimg.fits")  # Validate path components now.
        rows.append(converted)
    return rows


def run_ztf_search(manifest_path):
    path = Path(manifest_path).resolve()
    query = read_json(path)
    _fields(query, ["schema_version", "regions", "start_mjd_utc", "end_mjd_utc", "filtercode", "max_exposures"], ["require_zero_infobits"])
    if query["schema_version"] != 1 or query["filtercode"] not in {"zg", "zr", "zi"}:
        raise ValueError("Use schema_version 1 and a single ZTF filter: zg, zr, or zi")
    start, end = query["start_mjd_utc"], query["end_mjd_utc"]
    if not np.all(np.isfinite([start, end])) or not 58000 <= start < end <= 88069 or end - start > 366:
        raise ValueError("Query an increasing interval of at most 366 days after MJD 58000")
    cap = query["max_exposures"]
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 10000:
        raise ValueError("max_exposures must be an integer in [1,10000]")
    regions = query["regions"]
    if not isinstance(regions, list) or not 1 <= len(regions) <= 100:
        raise ValueError("Supply 1..100 explicit search regions")
    where = f"obsjd>={start + 2400000.5:.10f} AND obsjd<{end + 2400000.5:.10f} AND filtercode='{query['filtercode']}'"
    quality = query.get("require_zero_infobits", True)
    if not isinstance(quality, bool):
        raise ValueError("require_zero_infobits must be boolean")
    if quality:
        where += " AND infobits=0"
    requests, exposures = [], {}
    for region in regions:
        _fields(region, ["ra_deg", "dec_deg", "width_deg"])
        _position(region["ra_deg"], region["dec_deg"])
        if not np.isfinite(region["width_deg"]) or not 0 < region["width_deg"] <= 10:
            raise ValueError("Region width must be in (0,10] degrees")
        parameters = {"POS": f"{region['ra_deg']},{region['dec_deg']}", "SIZE": region["width_deg"],
                      "INTERSECT": "OVERLAPS", "WHERE": where, "CT": "ipac_table"}
        url = f"{BASE}/search/ztf/products/sci?{urlencode(parameters)}"
        body = _download(url, 32 * 1024**2)
        rows = parse_index(body)
        requests.append({"url": url, "response_sha256": hashlib.sha256(body).hexdigest(), "rows": len(rows)})
        for row in rows:
            # Query-local in_row_id / in_ra / in_dec can differ across regions.
            row = {key: value for key, value in row.items() if not key.startswith("in_")}
            if row["pid"] in exposures and exposures[row["pid"]] != row:
                raise ValueError("Conflicting archive metadata for the same product ID")
            exposures[row["pid"]] = row
        if len(exposures) > cap:
            raise ValueError("Archive returned more than max_exposures; narrow the query (nothing was truncated)")
    return {"schema_version": 1, "tool": "ztf_search", "archive": "IRSA ZTF public science",
            "retrieved_utc": datetime.now(timezone.utc).isoformat(), "query": query,
            "requests": requests, "provenance": _provenance([path]),
            "scope": "Metadata footprints overlapping the supplied boxes and exposure-start UTC interval; no claim of all-orbit or all-survey completeness",
            "exposures": sorted(exposures.values(), key=lambda row: (row["obsjd"], row["pid"]))}


def _plane(body):
    with fits.open(io.BytesIO(body), memmap=False) as hdus:
        data, header = np.array(hdus[0].data, dtype=float), hdus[0].header.copy()
    if data.ndim != 2 or min(data.shape) < 2:
        raise ValueError("Archive product is not a 2D FITS image")
    return data, header


def prepare_exposure(science_body, mask_body, psf_body, row, reference_mag):
    image, header = _plane(science_body)
    mask, mask_header = _plane(mask_body)
    kernel, _ = _plane(psf_body)
    validate_psf_kernel(kernel)
    if image.shape != mask.shape:
        raise ValueError("Science/mask cutouts have different shapes")
    wcs, mask_wcs = WCS(header).celestial, WCS(mask_header).celestial
    if not wcs.has_celestial or not mask_wcs.has_celestial:
        raise ValueError("Archive science and mask need celestial WCS")
    # Match both corners and center, not just CRPIX (distortion can differ).
    h, w = image.shape
    x, y = np.array([0, w - 1, (w - 1) / 2]), np.array([0, h - 1, (h - 1) / 2])
    mx, my = mask_wcs.world_to_pixel(wcs.pixel_to_world(x, y))
    if not np.allclose([mx, my], [x, y], atol=1e-4, rtol=0):
        raise ValueError("Science/mask WCS do not align")
    if str(header["DBPID"]) != row["pid"] or header["FILTERID"] != {"zg": 1, "zr": 2, "zi": 3}[row["filtercode"]]:
        raise ValueError("Downloaded science product does not match the selected product ID/filter")
    if abs(header["OBSJD"] - row["obsjd"]) > 1e-5:
        raise ValueError("Downloaded exposure time disagrees with index")
    gain, zp, fwhm, duration = [float(header[k]) for k in ["GAIN", "MAGZP", "SEEING", "EXPTIME"]]
    if not np.all(np.isfinite([gain, zp, fwhm, duration])) or min(gain, fwhm, duration) <= 0:
        raise ValueError("Missing or invalid gain, zero point, seeing, or exposure time")
    invalid = (mask != 0) | ~np.isfinite(image) | (image >= float(header["SATURATE"]))
    sky = image[~invalid]
    if sky.size < 100:
        raise ValueError("Need at least 100 unmasked pixels to estimate sky noise")
    background = float(np.median(sky))
    sigma = float(1.4826022185 * np.median(np.abs(sky - background)))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Cannot estimate a positive sky noise scale")
    variance = sigma**2 + np.maximum(image - background, 0) / gain
    # OBSJD labels UTC-OBS near shutter opening. Prefer actual shutter endpoints.
    if "SHUTOPEN" in header and "SHUTCLSD" in header:
        opened, closed = Time(header["SHUTOPEN"], scale="utc"), Time(header["SHUTCLSD"], scale="utc")
        duration = float((closed - opened).sec)
        if not 0 < duration < 3600 or abs(float(opened.jd) - float(header["OBSJD"])) > 5 / 86400:
            raise ValueError("Invalid/inconsistent shutter times")
        midpoint = opened + (closed - opened) / 2
    else:
        midpoint = Time(header["OBSJD"] + duration / 172800, format="jd", scale="utc")
    exposure = {"id": f"ztf-{row['pid']}", "image": "", "image_hdu": "SCI", "variance": "", "variance_hdu": "VAR",
                "mask": "", "mask_hdu": "MASK", "psf_image": "", "psf_hdu": "PSF",
                "psf_sigma_pixels": fwhm / 2.354820045, "flux_scale": float(10**(0.4 * (zp - reference_mag))),
                "mjd_tdb": float(midpoint.tdb.mjd), "exposure_seconds": duration,
                "observatory": {"longitude_deg": float(header["OBSLON"]), "latitude_deg": float(header["OBSLAT"]), "height_m": float(header["OBSALT"])}}
    notes = {"pid": row["pid"], "filtercode": row["filtercode"], "sky_median_adu": background,
             "sky_sigma_adu": sigma, "gain_e_per_adu": gain, "magzp": zp, "catalog_maglimit": row.get("maglimit"),
             "masked_fraction": float(invalid.mean()), "psf_negative_mass": float(-kernel[kernel < 0].sum() / kernel.sum()),
             "noise_model": "Estimated local MAD sky variance plus nonnegative observed source excess/gain; not a survey uncertainty product",
             "psf_model": "Survey empirical PSF at quadrant center; spatial variation and native-pixel interpolation error are not fitted"}
    hdus = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(image, header, name="SCI"),
                         fits.ImageHDU(variance, name="VAR"), fits.ImageHDU(invalid.astype(np.uint8), name="MASK"),
                         fits.ImageHDU(kernel, name="PSF")])
    return exposure, notes, hdus


def run_ztf_fetch(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ["schema_version", "index", "selections", "directory", "reference_mag"], ["max_total_bytes"])
    if manifest["schema_version"] != 1 or not np.isfinite(manifest["reference_mag"]) or not -10 <= manifest["reference_mag"] <= 40:
        raise ValueError("Use schema_version 1 and reference_mag in [-10,40]")
    index_path = (path.parent / manifest["index"]).resolve()
    index = read_json(index_path)
    if index.get("tool") != "ztf_search" or index.get("schema_version") != 1:
        raise ValueError("index must be a ztf_search report")
    rows = {row["pid"]: row for row in index["exposures"]}
    selections = manifest["selections"]
    if not isinstance(selections, list) or not 1 <= len(selections) <= 100:
        raise ValueError("Select 1..100 distinct products explicitly")
    seen, filters = set(), set()
    for selection in selections:
        _fields(selection, ["pid", "ra_deg", "dec_deg", "size_arcsec"])
        pid = selection["pid"]
        if pid not in rows or pid in seen:
            raise ValueError("Selection needs a unique product ID from the index")
        seen.add(pid)
        filters.add(rows[pid]["filtercode"])
        _position(selection["ra_deg"], selection["dec_deg"])
        if not np.isfinite(selection["size_arcsec"]) or not 20 <= selection["size_arcsec"] <= 1200:
            raise ValueError("Cutout size must be in [20,1200] arcseconds")
    if len(filters) != 1:
        raise ValueError("Fetch one filter at a time; no implicit color model is supplied")
    budget = manifest.get("max_total_bytes", 128 * 1024**2)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 1024**3:
        raise ValueError("max_total_bytes must be an integer in [1,1 GiB]")
    directory = (path.parent / manifest["directory"]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    exposures, notes, products, files = [], [], [], [path, index_path]
    for selection in selections:
        row = rows[selection["pid"]]
        bodies = []
        for suffix in ["sciimg.fits", "mskimg.fits", "sciimgdaopsfcent.fits"]:
            url = product_url(row, suffix)
            if suffix != "sciimgdaopsfcent.fits":
                url += "?" + urlencode({"center": f"{selection['ra_deg']},{selection['dec_deg']}", "size": f"{selection['size_arcsec']}arcsec", "gzip": "false"})
            body = _download(url, min(budget, 32 * 1024**2))
            budget -= len(body)
            raw = directory / f"{row['pid']}-{suffix}"
            raw.write_bytes(body)
            files.append(raw)
            products.append({"url": url, "path": str(raw), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
            bodies.append(body)
        exposure, diagnostic, hdus = prepare_exposure(*bodies, row, manifest["reference_mag"])
        processed = directory / f"{row['pid']}-prepared.fits"
        hdus.writeto(processed)
        files.append(processed)
        for key in ["image", "variance", "mask", "psf_image"]:
            exposure[key] = str(processed)
        exposures.append(exposure)
        notes.append(diagnostic)
    result = {"schema_version": 1, "tool": "ztf_fetch", "scope": "Explicitly selected public science cutouts; coverage is limited to downloaded cutouts",
              "filtercode": next(iter(filters)), "reference_mag": manifest["reference_mag"], "exposures": exposures,
              "photometry_exposures": [{k: v for k, v in e.items() if k not in {"mjd_tdb", "exposure_seconds", "observatory"}} for e in exposures],
              "calibration": notes, "products": products, "provenance": _provenance(files)}
    (directory / "inventory.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
