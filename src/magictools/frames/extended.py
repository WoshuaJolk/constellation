"""Direct public archive adapters, with native product identities and receipts."""
import json
import math
import re
from urllib.parse import quote, urlsplit, urljoin

from .adapters import frame, obscore, convert_rows, url
from .model import mjd, iso
from .parsers import numeric, votable
from .profiles import SIA1, SIA2
from .transport import ArchiveError, Blocked


def boxes(ra, dec, radius):
    """Conservative longitude/latitude rectangles; split at RA=0 and 90 degrees."""
    low, high = max(-90, dec-radius), min(90, dec+radius)
    half = 180 if low <= -90 or high >= 90 else math.degrees(math.asin(min(1, math.sin(math.radians(radius))/math.cos(math.radians(dec)))))
    begin, end = ra-half, ra+half
    if half == 180:
        begin, end = 0, 360
    while begin < end or (begin == end and radius == 0):
        stop = min(end, (math.floor(begin/90)+1)*90)
        yield begin % 360, (stop % 360 or 360), low, high
        if stop == end:
            break
        begin = stop


def tap(t, endpoint, query, s):
    return votable(t.request(url(endpoint, dict(REQUEST="doQuery", LANG="ADQL", FORMAT="votable", MAXREC=2000000, QUERY=query))), s)


def observation_time(value):
    return mjd(value) if value and str(value).lower() not in {"null", "none", "nan"} else None


def execute_extended(job, x, t, s):
    provider, w = job.provider, job.window
    if provider in SIA2:
        if provider == "rubin" and not (t.credentials and t.credentials.get("rubin").get("token")):
            raise Blocked("Set RSP_API_TOKEN (read:image scope) or configure the rubin credential profile")
        p = dict(POS=f"CIRCLE {w['ra']} {w['dec']} {w['radius']}", TIME=f"{w['start']} {w['end']}", MAXREC=2000000)
        if provider == "koa":
            p.update(TIME=f"{w['start']}/{w['end']}", RESULTFORMAT="votable", INSTRUMENT=job.params.get("instrument", "DEIMOS"))
        def convert(r):
            if provider != "koa":
                return obscore(job, r)
            start = numeric(r.get("MJD_OBS"))
            return frame(job, r, id=r.get("koaid"), survey="Keck", instrument=r.get("instrume"),
                         ra=numeric(r.get("ra")), dec=numeric(r.get("dec")), startMjd=start,
                         exposureSeconds=numeric(r.get("elaptime")), filter=r.get("filter"), url=r.get("url"))
        yield from convert_rows(votable(t.request(url(SIA2[provider], p)), s), convert, job, x, s)
    elif provider in SIA1:
        yield from query_sia1(job, x, t, s)
    elif provider == "euclid":
        query = f"SELECT * FROM ivoa.obscore WHERE dataproduct_type='image' AND INTERSECTS(s_region,CIRCLE('ICRS',{w['ra']},{w['dec']},{w['radius']}))=1 AND t_min<{w['end']} AND t_max>={w['start']}"
        def convert(r):
            # obs_id identifies a visit, shared by multiple filters/products.
            return obscore(job, {**r, "obs_id": r.get("file_name") or r.get("obs_publisher_did"), "filters": r.get("filter")})
        yield from convert_rows(tap(t, "https://eas.esac.esa.int/tap-server/tap/sync", query, s), convert, job, x, s)
    elif provider == "gemini":
        dates = [iso(w[k]).replace("-", "").replace(":", "").split(".")[0].rstrip("Z") for k in ("start", "end")]
        query = "https://archive.gemini.edu/jsonsummary/canonical/imaging/OBJECT/" + "/".join([f"ra={w['ra']}", f"dec={w['dec']}", f"sr={min(180,w['radius']+.15)*3600}", "-".join(dates)])
        data = json.loads(t.request(query))
        if not isinstance(data, list):
            raise ArchiveError("Gemini did not return a record list")
        s["received"] += len(data)
        def convert(r):
            return frame(job, r, id=r.get("name") or r.get("filename"), survey="Gemini", instrument=r.get("instrument"),
                         startMjd=observation_time(r.get("ut_datetime")), ra=numeric(r.get("ra")), dec=numeric(r.get("dec")),
                         exposureSeconds=numeric(r.get("exposure_time")), filter=r.get("filter_name"), fovRadius=.15,
                         url="https://archive.gemini.edu/file/"+quote(r["filename"], safe=""))
        yield from convert_rows(data, convert, job, x, s)
    elif provider == "lco":
        yield from query_lco(job, x, t, s)
    elif provider == "applause":
        # Multiple exposures/solutions on a plate are deliberately retained as candidates.
        query = f"SELECT s.plate_id,s.solution_num,s.ra_icrs,s.dec_icrs,s.stc_polygon,pltexp.exposure_num,pltexp.jd_start,pltexp.jd_end,pltexp.exptime FROM applause_dr4.solution AS s JOIN applause_dr4.exposure AS pltexp ON s.plate_id=pltexp.plate_id WHERE (pltexp.jd_start<{w['end']+2400000.5} OR pltexp.jd_start IS NULL) AND (pltexp.jd_end>={w['start']+2400000.5} OR pltexp.jd_end IS NULL) AND (s.fov1 IS NULL OR s.fov2 IS NULL OR DISTANCE(POINT('ICRS',s.ra_icrs,s.dec_icrs),POINT('ICRS',{w['ra']},{w['dec']}))<={w['radius']}+s.fov1+s.fov2)"
        def convert(r):
            start, end = numeric(r.get("jd_start")), numeric(r.get("jd_end"))
            return frame(job, r, id=f"plate:{r['plate_id']}:exposure:{r['exposure_num']}:solution:{r['solution_num']}",
                         survey="APPLAUSE DR4", instrument="APPLAUSE plate", startMjd=start-2400000.5 if start else None,
                         endMjd=end-2400000.5 if end else None, timeScale="archive-defined", exposureSeconds=numeric(r.get("exptime")),
                         ra=numeric(r.get("ra_icrs")), dec=numeric(r.get("dec_icrs")), footprint=r.get("stc_polygon"),
                         productKind="photographic plate", url=url("https://www.plate-archive.org/tap/sync", dict(REQUEST="doQuery",LANG="ADQL",FORMAT="votable",QUERY=f"SELECT * FROM applause_dr4.plate WHERE plate_id={int(r['plate_id'])}")), linkKind="plate metadata")
        s["warnings"].append("A plate can have multiple exposures and astrometric solutions; their association is unverified. Epochs remain archive-defined.")
        yield from convert_rows(tap(t, "https://www.plate-archive.org/tap/sync", query, s), convert, job, x, s)
    elif provider == "herschel":
        yield from query_herschel(job, x, t, s)
    else:
        from .legacy import execute_legacy
        yield from execute_legacy(job, x, t, s)


def query_lco(job, x, t, s):
    w = job.window
    for left, right, bottom, top in boxes(w["ra"], w["dec"], max(w["radius"], 1e-8)):
        polygon = f"POLYGON(({left} {bottom},{right} {bottom},{right} {top},{left} {top},{left} {bottom}))"
        next_url = url("https://archive-api.lco.global/frames/", dict(start=iso(w["start"]), end=iso(w["end"]), configuration_type="EXPOSE", intersects=polygon, limit=100, pagination_style="cursor"))
        visited = set()
        while next_url:
            if next_url in visited:
                raise ArchiveError("LCO pagination repeated a page")
            if urlsplit(next_url).scheme != "https" or urlsplit(next_url).netloc != "archive-api.lco.global":
                raise ArchiveError("LCO returned an unexpected pagination origin")
            visited.add(next_url)
            data = json.loads(t.request(next_url))
            if not isinstance(data.get("results"), list):
                raise ArchiveError("LCO response lacks results")
            s["received"] += len(data["results"])
            def convert(r):
                area = r.get("area") or {}
                coords = area.get("coordinates", [[]])[0] if area.get("type") == "Polygon" else []
                footprint = "POLYGON ICRS " + " ".join(str(c) for p in coords for c in p) if coords else None
                return frame(job, r, id=str(r["id"]), survey="LCO", instrument=r.get("instrument_id"),
                             startMjd=observation_time(r.get("observation_date")), exposureSeconds=numeric(r.get("exposure_time")),
                             filter=r.get("primary_optical_element"), footprint=footprint,
                             url=f"https://archive-api.lco.global/frames/{int(r['id'])}/", linkKind="product manifest")
            yield from convert_rows(data["results"], convert, job, x, s)
            next_url = data.get("next")


def query_sia1(job, x, t, s):
    w = job.window
    # SIA1 SIZE is longitude/latitude extent, not a spherical angular radius.
    regions = list(boxes(w["ra"], w["dec"], max(w["radius"], 1e-8)))
    if job.provider == "dss" and w["radius"] > 1/3:
        raise Blocked("DSS cutout service accepts at most 40 arcmin per axis. Use smaller regions; no complete large-area plate index is exposed here.")
    for left, right, bottom, top in regions:
        query = url(SIA1[job.provider], dict(POS=f"{(left+right)/2},{(bottom+top)/2}", SIZE=f"{right-left},{top-bottom}", FORMAT="image/fits"))
        def convert(r):
            def field(*names, ucd=None):
                for name in names:
                    if r.get(name):
                        return r[name]
                if ucd:
                    for key, meta in s.get("fields", {}).items():
                        if meta.get("ucd", "").upper() == ucd.upper():
                            return r.get(key)
                return None
            access = field("Url", "URL", "AccessReference", ucd="VOX:Image_AccessReference")
            fmt = field("Format", ucd="VOX:Image_Format") or ""
            if "fits" not in fmt.lower():
                return None
            when = numeric(field("Image_MJDateObs", ucd="VOX:Image_MJDateObs"))
            identity = field("Name", "Title", ucd="VOX:Image_Title")
            # Titles can be shared by every field; access URL retains actual identity.
            f = frame(job, r, id=access or identity, survey=job.provider.upper(), instrument=field("Telescope_Instrument") or job.provider.upper(),
                      startMjd=when, ra=numeric(field("RA", "Ra", ucd="POS_EQ_RA_MAIN")), dec=numeric(field("DEC", "Dec", ucd="POS_EQ_DEC_MAIN")),
                      filter=field("BandPass", "Bandpass", "SurveyName"), url=access,
                      productKind="cutout" if job.provider == "dss" else "image")
            if when is None:
                s["partial"] = "Image discovery succeeded, but SIA1 omits observation dates; requested temporal coverage is unverified."
            return f
        # Filter browse products before row conversion so malformed science rows
        # remain isolated by convert_rows rather than aborting their provider.
        def rows():
            for r in votable(t.request(query), s):
                formats = [r.get(k, "") for k, meta in s.get("fields", {}).items()
                           if meta.get("ucd", "").upper() == "VOX:IMAGE_FORMAT"]
                fmt = r.get("Format") or r.get("format") or " ".join(formats)
                if "fits" in fmt.lower():
                    yield r
                else:
                    s["excluded"] += 1
        yield from convert_rows(rows(), convert, job, x, s)


def query_herschel(job, x, t, s):
    w = job.window
    query = f"SELECT o.observation_id,o.start_time,o.end_time,o.ra,o.dec,o.polygon_fov,i.instrument_name FROM hsa.v_active_observation AS o JOIN hsa.instrument AS i ON o.instrument_oid=i.instrument_oid WHERE o.is_public='true' AND i.instrument_name IN ('PACS','SPIRE') AND o.start_time<'{iso(w['end']).rstrip('Z')}' AND o.end_time>='{iso(w['start']).rstrip('Z')}'"
    def convert(r):
        raw = r.get("polygon_fov", "")
        pairs = re.findall(r"\(([+\-\d.eE]+)\s*,\s*([+\-\d.eE]+)\)", raw)
        footprint = "POLYGON ICRS " + " ".join(str(math.degrees(float(v))) for pair in pairs for v in pair) if pairs else None
        start, end = observation_time(r.get("start_time")), observation_time(r.get("end_time"))
        return frame(job, r, id=r.get("observation_id"), survey="Herschel", instrument=r.get("instrument_name"),
                     startMjd=start, endMjd=end, ra=numeric(r.get("ra")), dec=numeric(r.get("dec")), footprint=footprint,
                     productKind="observation bundle", url=url("https://archives.esac.esa.int/hsa/whsa-tap-server/data", dict(retrieval_type="OBSERVATION", observation_id=r["observation_id"])), linkKind="observation bundle")
    s["warnings"].append("Herschel returns observation bundles, which can contain multiple maps or spectra. Request bundle metadata to choose image products.")
    yield from convert_rows(tap(t, "https://archives.esac.esa.int/hsa/whsa-tap-server/tap/sync", query, s), convert, job, x, s)
