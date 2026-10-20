"""Archive protocols. Every request is recorded by Transport, including POST bodies."""
import json
import math
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

from .geometry import select
from .model import iso, mjd, sexagesimal
from .parsers import canonical, delimited, ipac, numeric, safe_url, smoka, votable
from .transport import ArchiveError, Blocked

SSOIS = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/ssos/ssosclf.pl"
MOST = "https://irsa.ipac.caltech.edu/cgi-bin/MOST/nph-most"
NOIRLAB = "https://astroarchive.noirlab.edu/api"
HSC = "https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/"
CATCH = "https://catch-api.astro.umd.edu"
TAP = {"cadc": "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync",
       "eso": "https://archive.eso.org/tap_obs/sync"}
TAP_COLUMNS = "obs_collection,obs_id,obs_publisher_did,instrument_name,t_min,t_max,t_exptime,s_ra,s_dec,s_region,em_min,em_max,access_url,access_format,calib_level"


@dataclass
class Job:
    provider: str
    hypothesis: str
    window: dict | None = None
    orbit: dict | None = None
    params: dict = field(default_factory=dict)
    id: str = ""


def stats():
    return {"received": 0, "rejected": 0, "excluded": 0, "warnings": []}


def url(base, params):
    return base + "?" + urlencode(params, doseq=True)


def frame(job, row, **values):
    result = dict(provider=job.provider, survey=job.provider, instrument=job.provider,
                  id="", startMjd=None, endMjd=None, timeScale="UTC",
                  exposureSeconds=None, filter=None, ra=None, dec=None, url=None,
                  linkKind="archive product", match="archive match", raw=row,
                  hypotheses=[job.hypothesis], calibrationLevel=None, footprint=None)
    result.update(values)
    if not result["id"] or result["id"] in ("null", "None"):
        raise ValueError("missing native product ID")
    result["url"] = safe_url(result["url"])
    result["key"] = canonical(result)
    result["products"] = [{"provider": job.provider, "id": result["id"], "url": result["url"],
                           "linkKind": result["linkKind"], "raw": row}]
    result["provenance"] = [{"jobId": job.id, "provider": job.provider, "hypothesis": job.hypothesis}]
    return result


def obscore(job, r):
    return frame(job, r, id=r.get("obs_id") or r.get("obs_publisher_did"),
                 survey=r.get("obs_collection", job.provider), instrument=r.get("instrument_name", ""),
                 startMjd=numeric(r.get("t_min")), endMjd=numeric(r.get("t_max")),
                 exposureSeconds=numeric(r.get("t_exptime")), filter=r.get("energy_bandpassname") or r.get("filters"),
                 ra=numeric(r.get("s_ra")), dec=numeric(r.get("s_dec")), footprint=r.get("s_region"),
                 url=r.get("access_url"), linkKind="DataLink" if "datalink" in (r.get("access_format") or "").lower() else "archive product",
                 calibrationLevel=numeric(r.get("calib_level")))


def ssois_row(job, r):
    when = numeric(r.get("MJD"))
    if when is None:
        raise ValueError("missing observation MJD")
    return frame(job, r, id=r.get("Image"), survey=r.get("Telescope/Instrument", ""),
                 instrument=r.get("Telescope/Instrument", ""), startMjd=when, timeScale="archive-defined",
                 filter=r.get("Filter"), exposureSeconds=numeric(r.get("Exptime")),
                 ra=numeric(r.get("Object_RA")), dec=numeric(r.get("Object_Dec")),
                 url=r.get("Datalink"), match="SSOIS predicted-position / uncertainty-box intersection")


def irsa_row(job, r, query):
    if job.provider in {"ztf", "most:ztf"}:
        from ..ztf import product_url
        when = numeric(r.get("obsjd"))
        if when is None:
            raise ValueError("missing obsjd")
        return frame(job, r, id=r.get("pid"), survey="ZTF", instrument="ZTF", startMjd=when-2400000.5,
                     exposureSeconds=numeric(r.get("exptime")), filter=r.get("filtercode"),
                     ra=numeric(r.get("ra")), dec=numeric(r.get("dec")),
                     url=product_url(r, "sciimg.fits"), linkKind="FITS",
                     footprint="POLYGON ICRS " + " ".join(str(r.get(k, "")) for i in range(1, 5) for k in (f"ra{i}", f"dec{i}")))
    identity = next((r[k] for k in ("pid", "cntr", "bcdid", "bcd_id", "filename", "fname", "scangrp", "coadd_id", "frame_id", "scan_id", "obsid") if r.get(k)), None)
    if r.get("scan_id") and r.get("frame_num"):
        identity = f"{r['scan_id']}:{r['frame_num']}:{r.get('band', '')}"
    if not identity:
        # A hash is deliberately NOT substituted for an unknown native identity.
        raise ValueError("catalog row has no recognized native product ID")
    when = numeric(r.get("mjd_obs") or r.get("mjd") or r.get("obsmjd"))
    if when is None and numeric(r.get("obsjd")) is not None:
        when = numeric(r["obsjd"]) - 2400000.5
    if when is None:
        for k in ("obsdate", "date_obs", "dateobs"):
            if r.get(k):
                try:
                    when = mjd(r[k])
                    break
                except ValueError:
                    pass
    link = next((safe_url(r.get(k)) for k in ("access_url", "download_url", "data_url", "url", "bcdurl", "fname") if safe_url(r.get(k))), None)
    return frame(job, r, id=identity, instrument=job.provider.removeprefix("most:"),
                 survey=job.provider.removeprefix("most:"), startMjd=when,
                 ra=numeric(r.get("ra")), dec=numeric(r.get("dec")), filter=r.get("filter") or r.get("band"),
                 exposureSeconds=numeric(r.get("exptime")), url=link or query,
                 linkKind="archive product" if link else "archive query",
                 match="MOST predicted-position match; see native metadata")


def convert_rows(rows, convert, job, x, s):
    for r in rows:
        try:
            f = convert(r)
            if job.window and not select(f, job.window, x):
                s["excluded"] += 1
                continue
            yield f
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            s["rejected"] += 1
            if len(s["warnings"]) < 5:
                s["warnings"].append("Dropped a row with invalid native ID, time, coordinates, or product path")


def ssois_params(job, x):
    p = dict(lang="en", format="tsv", epoch1=x["start"], epoch2=x["end"], eellipse=x["eellipse"],
             eunits="arcseconds", extres="no", xyres="no")
    if x["mode"] == "object":
        p.update(search=x["resolver"], object=x["object"])
    elif x["mode"] == "arc":
        p.update(search=x["arcFitter"], obs=x["arc"])
    elif job.orbit:
        o = job.orbit
        a, mean = o.get("a"), o.get("meanAnomaly")
        if o["kind"] == "cometary":
            if o["e"] == 1:
                raise Blocked("SSOIS orbel has no finite semimajor axis for e=1. Use MOST or a supplied ephemeris for this parabolic orbit.")
            a = o["q"] / (1-o["e"])
            mean = math.degrees(0.01720209895 / abs(a)**1.5 * (o["epochMjd"]+2400000.5-o["tpJd"]))
            if o["e"] < 1:
                mean %= 360
        p.update(search="orbel", epoch=o["epochMjd"], a=a, e=o["e"], i=o["i"], longi=o["node"], peri=o["peri"], anom=mean)
    else:
        w = job.window
        from .geometry import interpolate
        points = [dict(interpolate(w["points"], w["start"]), mjd=w["start"]),
                  *[r for r in w["points"] if w["start"] < r["mjd"] < w["end"]],
                  dict(interpolate(w["points"], w["end"]), mjd=w["end"])]
        # SSOIS exposes one uncertainty box per request, not per-row ellipses.
        p.update(search="userephem", ephem="\n".join(f"{r['mjd']:.9f} {r['ra']:.10f} {r['dec']:.10f}" for r in points),
                 eellipse=max(r["sigmaArcsec"] for r in points)*x["sigmaMultiplier"]+x["eellipse"]+x["radius"]*3600)
    return p


def most_params(job, x):
    p = dict(catalog=job.provider.removeprefix("most:"), obs_begin=x["start"].replace("-", " "),
             obs_end=x["end"].replace("-", " "), output_mode="Brief", ephem_step=x.get("ephemStepDays", 0.25))
    if job.orbit:
        o = job.orbit
        p.update(input_type="manual_input", obj_type="Comet" if o["kind"] == "cometary" else "Asteroid",
                 body_designation=o["label"], epoch=o["epochMjd"], eccentricity=o["e"], inclination=o["i"],
                 ascend_node=o["node"], arg_perihelion=o["peri"])
        p.update(dict(perih_dist=o["q"], perih_time=o["tpJd"]) if o["kind"] == "cometary"
                 else dict(semimajor_axis=o["a"], mean_anomaly=o["meanAnomaly"]))
    else:
        p.update(input_type="name_input", obj_name=x["object"])
    return p


def execute(job, x, transport, s):
    provider, w = job.provider, job.window
    if provider == "ssois":
        p = ssois_params(job, x)
        # Native CGI GET remains the proven path for pinned element/name runs.
        text = transport.request(SSOIS, data=urlencode(p), headers={"Content-Type": "application/x-www-form-urlencoded"}) if p["search"] in {"userephem", "bern", "mpc"} else transport.request(url(SSOIS, p))
        if re.search(r"(?:error getting|error occured getting) the ephemeris", text, re.I):
            raise ArchiveError("SSOIS could not generate the ephemeris for this input. MOST accepts cometary elements; supplied ephemerides can be searched directly against SMOKA/NOIRLab and other field archives.")
        yield from convert_rows(delimited(text, s, "\t"), lambda r: ssois_row(job, r), job, x, s)
        if "Image" not in s.get("columns", []):
            raise ArchiveError("SSOIS response lacks its image table header")
        return
    if provider.startswith("most:"):
        if x["eellipse"]:
            s["warnings"].append("MOST does not accept this uncertainty-box parameter; eellipse applies to SSOIS and direct ephemeris searches, not this MOST check.")
        query = url(MOST, most_params(job, x))
        text = transport.request(query)
        rows = ipac(text, s)
        yield from convert_rows((r for r in rows if r.get("match") in (None, "1")), lambda r: irsa_row(job, r, query), job, x, s)
        return
    if provider == "ztf":
        query = url("https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci",
                    dict(POS=f"{w['ra']},{w['dec']}", SIZE=max(1e-8, 2*w["radius"]), INTERSECT="OVERLAPS",
                         WHERE=f"obsjd>={w['start']+2400000.5} AND obsjd<{w['end']+2400000.5}", CT="ipac_table"))
        yield from convert_rows(ipac(transport.request(query), s, True), lambda r: irsa_row(job, r, query), job, x, s)
        return
    if provider in TAP:
        query = f"SELECT {TAP_COLUMNS} FROM ivoa.ObsCore WHERE dataproduct_type='image' AND INTERSECTS(s_region,CIRCLE('ICRS',{w['ra']},{w['dec']},{w['radius']}))=1 AND t_min < {w['end']} AND t_max >= {w['start']}"
        text = transport.request(url(TAP[provider], dict(REQUEST="doQuery", LANG="ADQL", FORMAT="votable", MAXREC=2000000, QUERY=query)))
        yield from convert_rows(votable(text, s), lambda r: obscore(job, r), job, x, s)
        return
    if provider == "irsa":
        text = transport.request(url("https://irsa.ipac.caltech.edu/SIA", dict(POS=f"CIRCLE {w['ra']} {w['dec']} {w['radius']}", TIME=f"{w['start']} {w['end']}", MAXREC=2000000)))
        yield from convert_rows((r for r in votable(text, s) if r.get("dataproduct_type", "image") == "image"), lambda r: obscore(job, r), job, x, s)
        return
    if provider == "smoka":
        yield from query_smoka(job, x, transport, s)
    elif provider == "noirlab" or provider.startswith("noirlab:"):
        yield from query_noirlab(job, x, transport, s)
    elif provider == "hsc":
        yield from query_hsc(job, x, transport, s)
    elif provider == "catch":
        yield from query_catch(job, x, transport, s)
    elif provider == "mast:targetpixels":
        yield from query_mast_targetpixels(job, x, transport, s)
    elif provider == "mast":
        yield from query_mast(job, x, transport, s)
    else:
        from .extended import execute_extended
        yield from execute_extended(job, x, transport, s)


def query_smoka(job, x, transport, s):
    w = job.window
    # Both cameras' full pointing envelopes are included. Individual CCDs and
    # gaps are not determined from encoder coordinates in fssearch.
    for inst, extent in [("HSC", 0.85), ("SUP", 0.45)]:
        params = [("resolver", "NONE"), ("coordsys", "Equatorial"), ("equinox", "J2000"),
                  ("fieldofview", "0"), ("RadOrRec", "radius"), ("longitudeC", str(w["ra"])),
                  ("latitudeC", str(w["dec"])), ("radius", min(10800, (w["radius"]+extent)*60)),
                  ("asciitable", "Ascii"), ("action", "Search"), ("obs_mod", "IMAG"), ("data_typ", "OBJECT"),
                  ("instruments", inst), ("date_obs", f"{iso(w['start'])[:10]}..{iso(w['end'])[:10]}"),
                  ("orderby", "FRAMEID"), ("asc", "1"), ("diff", "20000")]
        params += [("dispcol", name) for name in ["FRAMEID", "DATE_OBS", "UT_START", "RA2000", "DEC2000", "FILTER", "EXPTIME", "PROP_ID"]]
        local = stats()
        text = transport.request("https://smoka.nao.ac.jp/fssearch?" + urlencode(params))
        def convert(r):
            return frame(job, r, id=r["FRAMEID"], survey="Subaru/HSC" if inst == "HSC" else "Subaru/Suprime-Cam",
                         instrument="Subaru/HSC" if inst == "HSC" else "Subaru/Suprime-Cam",
                         startMjd=mjd(r["DATE_OBS"]+"T"+r["UT_START"]), exposureSeconds=numeric(r.get("EXPTIME")),
                         filter=r.get("FILTER"), ra=sexagesimal(r["RA2000"], True), dec=sexagesimal(r["DEC2000"]),
                         fovRadius=extent, url=url("https://smoka.nao.ac.jp/fssearch", dict(frameid=r["FRAMEID"], action="Search", asciitable="Ascii")),
                         linkKind="registered archive request")
        yield from convert_rows(smoka(text, local), convert, job, x, local)
        accumulate(s, local)


def accumulate(target, source):
    for key in ("received", "rejected", "excluded"):
        target[key] += source[key]
    target["warnings"] = (target["warnings"] + source["warnings"])[:10]
    if source.get("incomplete"):
        target["incomplete"] = source["incomplete"]


def query_noirlab(job, x, transport, s):
    w = job.window
    instrument = job.params.get("instrument", "decam")
    # Conservative camera pointing envelope, then exposure-time filtering.
    padding = min(180, w["radius"]+1.3)
    dec0, dec1 = max(-90, w["dec"]-padding), min(90, w["dec"]+padding)
    if dec0 <= -90 or dec1 >= 90:
        ranges = [(0, 360)]
    else:
        dra = math.degrees(math.asin(min(1, math.sin(math.radians(padding))/math.cos(math.radians(w["dec"]))))) if padding < 90 else 180
        lo, hi = w["ra"]-dra, w["ra"]+dra
        ranges = [(lo+360, 360), (0, hi)] if lo < 0 else [(lo, 360), (0, hi-360)] if hi > 360 else [(lo, hi)]
    for lo, hi in ranges:
        body = dict(outfields=["md5sum", "archive_filename", "ra_center", "dec_center", "dateobs_center", "exposure", "ifilter", "instrument", "proc_type", "prod_type"],
                    search=[["instrument", instrument], ["prod_type", "image"],
                            ["dateobs_center", iso(w["start"]), iso(w["end"])], ["ra_center", lo, hi], ["dec_center", dec0, dec1]])
        if job.provider == "noirlab":
            body["search"].append(["proc_type", "instcal"])
        offset = 0
        while True:
            data = json.loads(transport.request(url(NOIRLAB+"/adv_search/find/", dict(limit=1000, offset=offset)), data=body))
            if not isinstance(data, list) or not data or "PARAMETERS" not in data[0]:
                raise ArchiveError("NOIRLab returned no query metadata")
            rows = data[1:]
            s["received"] += len(rows)
            def convert(r):
                return frame(job, r, id=r["archive_filename"].split("/")[-1], survey="DECam" if instrument == "decam" else "NOIRLab/"+instrument, instrument=instrument,
                             startMjd=mjd(r["dateobs_center"])-(numeric(r.get("exposure")) or 0)/172800,
                             exposureSeconds=numeric(r.get("exposure")),
                             filter=r.get("ifilter"), ra=numeric(r.get("ra_center")), dec=numeric(r.get("dec_center")),
                             fovRadius=1.3, url=f"{NOIRLAB}/retrieve/{r['md5sum']}/", linkKind="FITS")
            yield from convert_rows(rows, convert, job, x, s)
            if len(rows) < 1000:
                break
            offset += len(rows)


def query_hsc(job, x, transport, s):
    credentials = transport.credentials.get("hsc") if transport.credentials else {}
    if not credentials.get("username") or not credentials.get("password"):
        raise Blocked("HSC-SSP account required. Set HSC_SSP_CAS_USER and HSC_SSP_CAS_PASSWORD, or run `magictools frames credentials init`.")
    w = job.window
    for schema in ["pdr3_wide", "pdr3_dud_rev"]:
        # Unit-vector cube conservatively encloses the circular cone. areacube
        # is documented in the public PDR3 frame schema; exact corners follow.
        from .geometry import vector
        v = vector(w["ra"], w["dec"])
        half = 2*math.sin(math.radians(w["radius"])/2)
        lo, hi = [max(-1, a-half) for a in v], [min(1, a+half) for a in v]
        columns = "frame_id,visit,ccd,ra2000,dec2000,utcobs,exptime,filter01,llcra,llcdec,ulcra,ulcdec,urcra,urcdec,lrcra,lrcdec"
        sql = f"SELECT {columns} FROM {schema}.frame WHERE utcobs >= '{iso(w['start'])}' AND utcobs < '{iso(w['end'])}' AND areacube && cube(ARRAY[{','.join(map(str,lo))}],ARRAY[{','.join(map(str,hi))}])"
        catalog = {"sql": sql, "out_format": "csv", "include_metainfo_to_body": True, "release_version": "pdr3"}
        credential = {"account_name": credentials["username"], "password": credentials["password"]}
        public = {"catalog_job": catalog, "nomail": True, "skip_syntax_check": False}
        submitted = json.loads(transport.request(HSC+"submit", data=public, private_data={**public, "credential": credential}))
        job_id = submitted["id"]
        deadline = time.monotonic()+x["timeoutSeconds"]
        while True:
            state = json.loads(transport.request(HSC+"status", data={"id": job_id}, private_data={"id": job_id, "credential": credential}))
            if state["status"] == "done":
                break
            if state["status"] == "error":
                raise ArchiveError("HSC CAS query failed; inspect the recorded SQL in the receipt")
            if time.monotonic() > deadline:
                s["remoteJob"] = {"id": job_id, "url": HSC+"status"}
                raise ArchiveError("HSC CAS job is still running; remote job ID retained")
            time.sleep(2)
        text = transport.request(HSC+"download", data={"id": job_id}, private_data={"id": job_id, "credential": credential})
        text = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
        def convert(r):
            return frame(job, r, id=r["frame_id"], survey=f"HSC-SSP/{schema}", instrument="Subaru/HSC",
                         startMjd=mjd(r["utcobs"]), exposureSeconds=numeric(r.get("exptime")), filter=r.get("filter01"),
                         ra=numeric(r.get("ra2000")), dec=numeric(r.get("dec2000")),
                         footprint="POLYGON ICRS "+" ".join(r[k] for k in ["llcra", "llcdec", "ulcra", "ulcdec", "urcra", "urcdec", "lrcra", "lrcdec"]),
                         url="https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/", linkKind="registered warp/cutout request")
        yield from convert_rows(delimited(text, s), convert, job, x, s)


def query_catch(job, x, transport, s):
    if job.window:
        w = job.window
        if w["radius"] > 2:
            raise Blocked("CATCH supports a maximum 2-degree search radius; this query window is wider")
        query = url(CATCH+"/fixed", dict(ra=w["ra"], dec=w["dec"], radius=w["radius"]*60,
                                        start_date=iso(w["start"]), stop_date=iso(w["end"]), intersection_type="ImageIntersectsArea"))
        data = json.loads(transport.request(query))
    else:
        if x["eellipse"] > 7200:
            raise Blocked("CATCH accepts at most 7200 arcsec of padding; use a supplied ephemeris or a narrower uncertainty envelope")
        submitted = json.loads(transport.request(url(CATCH+"/catch", dict(target=x["object"], start_date=x["start"],
                              stop_date=x["end"], cached="true", padding=x["eellipse"]/60, uncertainty_ellipse="true"))))
        if not submitted.get("job_id"):
            raise ArchiveError("CATCH did not accept the target: " + str(submitted.get("message", "no job ID returned")))
        job_id = submitted["job_id"]
        deadline = time.monotonic()+x["timeoutSeconds"]
        while True:
            state = json.loads(transport.request(CATCH+"/status/"+str(job_id)))
            statuses = [v["status"] for v in state.get("status", [])]
            if statuses and all(v in {"finished", "errored", "error"} for v in statuses):
                if any(v != "finished" for v in statuses):
                    s["incomplete"] = "Some CATCH sources failed; see the native status response"
                break
            if time.monotonic() > deadline:
                s["remoteJob"] = {"id": job_id, "url": CATCH+"/caught/"+str(job_id)}
                raise ArchiveError("CATCH job remains pending; remote poll URL retained")
            time.sleep(2)
        data = json.loads(transport.request(CATCH+"/caught/"+str(job_id)))
    if not isinstance(data.get("data"), list) or data.get("count") != len(data["data"]):
        raise ArchiveError("CATCH response count does not match its data")
    s["received"] += len(data["data"])
    def convert(r):
        footprint = "POLYGON ICRS " + r.get("fov", "").replace(":", " ").replace(",", " ")
        return frame(job, r, id=r["product_id"], survey=r["source_name"], instrument=r["source_name"],
                     startMjd=numeric(r.get("mjd_start")), endMjd=numeric(r.get("mjd_stop")),
                     exposureSeconds=numeric(r.get("exposure")), filter=r.get("filter"), footprint=footprint,
                     url=r.get("archive_url") or r.get("cutout_url"), linkKind="archive product")
    yield from convert_rows(data["data"], convert, job, x, s)


def query_mast(job, x, transport, s):
    w = job.window
    # MAST's CAOM filtered API supports a sky-position cone and full paging.
    # Position searches are archive-selected candidates; retain the provided
    # s_region and apply the exact footprint check locally when available.
    page = 1
    while True:
        request = {"service": "Mast.Caom.Filtered.Position", "params": {
            "columns": "*", "filters": [{"paramName": "dataproduct_type", "values": ["timeseries"] if job.provider == "mast:targetpixels" else ["image"]},
                                       *([{"paramName": "obs_collection", "values": ["Kepler", "K2"]}] if job.provider == "mast:targetpixels" else []),
                                       {"paramName": "t_min", "values": [{"min": -1000000, "max": w["end"]}]},
                                       {"paramName": "t_max", "values": [{"min": w["start"], "max": 1000000}]}],
            "position": f"{w['ra']}, {w['dec']}, {w['radius']}"},
            "format": "json", "pagesize": 1000, "page": page}
        data = json.loads(transport.request("https://mast.stsci.edu/api/v0/invoke", data=urlencode({"request": json.dumps(request)}), headers={"Content-Type": "application/x-www-form-urlencoded"}))
        if data.get("status") != "COMPLETE" or not isinstance(data.get("data"), list):
            raise ArchiveError("MAST query did not complete successfully")
        rows = data["data"]
        s["received"] += len(rows)
        def convert(r):
            r = dict(r)
            r["access_url"] = r.get("dataURL") or r.get("access_url") or ""
            if r["access_url"].startswith("mast:"):
                r["access_url"] = url("https://mast.stsci.edu/api/v0.1/Download/file", {"uri": r["access_url"]})
            if not r["access_url"] and r.get("obsid"):
                r["access_url"] = url("https://mast.stsci.edu/api/v0/invoke", {"request": json.dumps({
                    "service": "Mast.Caom.Products", "params": {"obsid": str(r["obsid"])}, "format": "json"})})
                f = obscore(job, r)
                f["linkKind"] = f["products"][0]["linkKind"] = "product manifest"
                return f
            return obscore(job, r)
        yield from convert_rows(rows, convert, job, x, s)
        paging = data.get("paging", {})
        if page >= paging.get("pagesFiltered", paging.get("pages", 1)):
            break
        page += 1


def query_mast_targetpixels(job, x, transport, s):
    parents = stats()
    for observation in query_mast(job, x, transport, parents):
        obsid = observation["raw"].get("obsid")
        if not obsid:
            s["rejected"] += 1
            continue
        page = 1
        while True:
            request = {"service": "Mast.Caom.Products", "params": {"obsid": str(obsid)},
                       "format": "json", "pagesize": 1000, "page": page}
            data = json.loads(transport.request("https://mast.stsci.edu/api/v0/invoke", data=urlencode({"request": json.dumps(request)}), headers={"Content-Type": "application/x-www-form-urlencoded"}))
            if data.get("status") != "COMPLETE" or not isinstance(data.get("data"), list):
                raise ArchiveError("MAST target-pixel product query did not complete")
            rows = data["data"]
            s["received"] += len(rows)
            def convert(r):
                uri = r.get("dataURI")
                if not uri:
                    raise ValueError("Target-pixel product lacks dataURI")
                return frame(job, r, id=r.get("productFilename") or uri, survey=observation["survey"], instrument=observation["instrument"],
                             startMjd=observation["startMjd"], endMjd=observation["endMjd"],
                             ra=observation["ra"], dec=observation["dec"], footprint=observation["footprint"],
                             productKind="target-pixel time cube", url=url("https://mast.stsci.edu/api/v0.1/Download/file", {"uri": uri}), linkKind="FITS")
            targets = []
            for row in rows:
                if "targ.fits" in str(row.get("productFilename", "")).lower():
                    targets.append(row)
                else:
                    s["excluded"] += 1
            yield from convert_rows(targets, convert, job, x, s)
            paging = data.get("paging", {})
            if page >= paging.get("pagesFiltered", paging.get("pages", 1)):
                break
            page += 1
    s["rejected"] += parents["rejected"]
    if parents.get("incomplete"):
        s["incomplete"] = parents["incomplete"]
    s["warnings"].append("Target-pixel cubes cover selected postage stamps and many epochs; inspect cadence times and downloaded pixel masks before claiming a moving-object intersection.")
