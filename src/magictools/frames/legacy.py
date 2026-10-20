"""Archive-specific indexes, photographic plates, and non-image scan products."""
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit
import xml.etree.ElementTree as ET

from .adapters import frame, convert_rows, url
from .extended import boxes, observation_time
from .geometry import distance
from .model import iso, mjd
from .parsers import delimited, ipac, numeric, votable
from .transport import ArchiveError, Blocked

AKARI = "https://data.darts.isas.jaxa.jp/pub/akari/AKARI-IRC_Image_Pointed_POIphase1-2_1.0.1/browse/dataList.html"
VSO = "https://vso.nascom.nasa.gov/cgi-bin/sunpy_vsoi"
_index_lock = threading.Lock()
_igaps_table = None


def metadata(t, url, binary=False):
    return getattr(t, "cached_request", t.request)(url, binary=binary)


class TableRows(HTMLParser):
    """Retain text and links from HTML cells without executing page content."""
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.links = [], [], None, []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "tr":
            self.row = []
        elif tag.lower() in {"td", "th"}:
            self.cell, self.links = [], []
        elif tag.lower() == "a" and self.cell is not None:
            self.links.extend(v for k, v in attrs if k.lower() == "href")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag.lower() in {"td", "th"} and self.cell is not None:
            self.row.append({"text": "".join(self.cell).strip(), "links": self.links})
            self.cell = None
        elif tag.lower() == "tr" and self.row:
            self.rows.append(self.row)
            self.row = []


def execute_legacy(job, x, t, s):
    if job.provider == "akari":
        parser = TableRows()
        parser.feed(metadata(t, AKARI))
        rows = [r for r in parser.rows if len(r) == 12 and re.fullmatch(r"\d+_\d+", r[1]["text"])]
        if not rows:
            raise ArchiveError("AKARI index does not contain the documented observation table")
        s["received"] += len(rows)
        s["warnings"].append("AKARI phase 1/2 IRC pointed bundles; all-sky maps are separately searchable with irsa. Camera envelope includes the offset MIR-L field; inspect individual FITS footprints.")
        def convert(c):
            r = {k: v["text"] for k, v in zip(["revision", "obsid", "object", "ra", "dec", "program", "mode", "aot", "aot_params", "start", "end", "wusv"], c)}
            return frame(job, r, id=r["obsid"], survey="AKARI IRC phase 1/2", instrument="AKARI/IRC",
                         ra=float(r["ra"]), dec=float(r["dec"]), startMjd=mjd(r["start"]), endMjd=mjd(r["end"]), fovRadius=.75,
                         productKind="observation bundle", url=c[1]["links"][0], linkKind="TAR observation bundle")
        yield from convert_rows(rows, convert, job, x, s)
    elif job.provider == "dasch":
        yield from query_dasch(job, x, t, s)
    elif job.provider == "splus":
        yield from query_splus(job, x, t, s)
    elif job.provider == "wfcam":
        yield from query_wfcam(job, x, t, s)
    elif job.provider == "iphas":
        yield from query_igaps(job, x, t, s)
    elif job.provider == "scanpi":
        yield from query_scanpi(job, x, t, s)
    elif job.provider == "lasco":
        yield from query_lasco(job, x, t, s)
    else:
        raise ArchiveError("No adapter is implemented for " + job.provider)


def query_dasch(job, x, t, s):
    w = job.window
    auth = t.credentials.get("dasch") if t.credentials else {}
    root = "https://api.starglass.cfa.harvard.edu/" + ("full" if auth.get("token") else "public")
    query = root + "/dasch/dr7/queryexps"
    # Official API is point-only. A nonzero region is explicitly incomplete.
    if w["radius"] or "points" in w:
        s["partial"] = "DASCH queryexps tests only the query center; the rest of the requested region or track remains unqueried."
    data = json.loads(t.request(query, data={"ra_deg": w["ra"], "dec_deg": w["dec"]}))
    if not isinstance(data, list) or not data or not all(isinstance(r, str) for r in data):
        raise ArchiveError("DASCH did not return its CSV-lines exposure table")
    def convert(r):
        plate = r["series"] + str(int(r["platenum"]))
        return frame(job, r, id=f"{plate}:exposure:{r['expnum']}:solution:{r['solnum']}",
                     survey="DASCH DR7", instrument="Harvard/"+r["series"], startMjd=observation_time(r.get("expdate")),
                     exposureSeconds=(numeric(r.get("exptime")) or 0)*60, ra=numeric(r.get("ra")), dec=numeric(r.get("dec")),
                     productKind="photographic plate", url=query, linkKind="API request recipe",
                     retrievalRequest={"url": root+"/dasch/dr7/cutout", "method": "POST", "json": {"plate_id": plate, "solution_number": int(r["solnum"]), "center_ra_deg": w["ra"], "center_dec_deg": w["dec"]}})
    yield from convert_rows(delimited("\n".join(data), s), convert, job, x, s)


def paged_list(t, endpoint, params=None, auth=None):
    skip, seen = 0, set()
    while True:
        page = json.loads(t.request(url(endpoint, {**(params or {}), "skip": skip, "limit": 100}), auth=auth))
        if not isinstance(page, list):
            raise ArchiveError("S-PLUS did not return a paginated list")
        if not page:
            break
        fingerprint = hashlib.sha256(json.dumps(page, sort_keys=True).encode()).hexdigest()
        if fingerprint in seen:
            raise ArchiveError("S-PLUS repeated a page; no complete result can be claimed")
        seen.add(fingerprint)
        yield from page
        skip += len(page)
        # Keep paging to an empty result: server may enforce a smaller page size.


def query_splus(job, x, t, s):
    w = job.window
    auth = t.credentials.get("splus") if t.credentials else {}
    if not auth.get("token") and auth.get("username") and auth.get("password"):
        result = json.loads(t.request("https://splus.cloud/adss/v1/auth/login", private_data=urlencode({"username": auth["username"], "password": auth["password"]}), headers={"Content-Type": "application/x-www-form-urlencoded"}, auth={}, sensitive_response=True))
        token = result.get("access_token")
        if not token:
            raise Blocked("S-PLUS login did not return an access token")
        t.credentials.secrets.append(token)
        auth = {"token": token}
    root = "https://splus.cloud/adss/v1/images"
    for collection in paged_list(t, root+"/collections/", auth=auth):
        endpoint = root+f"/collections/{int(collection['id'])}/files"
        # Do not constrain dates remotely: stacks often have date_obs=null.
        # Include a generous envelope for the ~2.4 degree mosaic diagonal.
        rows = paged_list(t, endpoint, dict(ra=w["ra"], dec=w["dec"], radius=min(180, w["radius"]+2)), auth)
        def valid_rows():
            for r in rows:
                s["received"] += 1
                if "weight" in str(r.get("filename", "")).lower():
                    s["excluded"] += 1
                    continue
                yield r
        def convert(r):
            when = observation_time(r.get("date_obs"))
            s["partial"] = "S-PLUS stack discovery succeeded; constituent exposure times are unavailable, so temporal coverage remains unverified."
            return frame(job, r, id=str(r["id"]), survey="S-PLUS "+collection["name"], instrument=r.get("instrument") or "T80Cam",
                         startMjd=when, timeSpanUnknown=True, ra=numeric(r.get("ra_center")), dec=numeric(r.get("dec_center")), filter=r.get("filter"),
                         fovRadius=2, productKind="stack", url=root+f"/files/{int(r['id'])}/download", linkKind="authenticated FITS download")
        yield from convert_rows(valid_rows(), convert, job, x, s)


def query_wfcam(job, x, t, s):
    w = job.window
    for database in ("UKIDSSDR11PLUS", "UHSDR3"):
        for left, right, bottom, top in boxes(w["ra"], w["dec"], min(180, w["radius"]+1)):
            # WSA raBase is HOURS. Row output limit is HTML only; VOTable contains ALL query rows.
            sql = f"SELECT multiframeID,fileName,raBase,decBase,mjdObs,dateEnd,expTime,filterName,frameType FROM Multiframe WHERE multiframeID>0 AND obstype='OBJECT' AND raBase BETWEEN {left/15} AND {right/15} AND decBase BETWEEN {bottom} AND {top} AND mjdObs<{w['end']} AND (mjdObs+expTime/86400.0)>={w['start']}"
            query = url("http://surveys.roe.ac.uk:8080/wsa/WSASQL", dict(database=database, formaction="freeform", sqlstmt=sql, format="VOT", compress="NONE", rows=1, timeout=int(t.timeout)))
            page = t.request(query, auth={})
            matches = re.findall(r'<a\b[^>]*id=["\']dl_id["\'][^>]*href=["\']([^"\']+)', page, re.I)
            if not matches:
                if re.search(r"Query returned 0 result rows", page, re.I):
                    continue
                raise ArchiveError("WFCAM SQL query did not provide its full result-file link")
            link = html.unescape(matches[0])
            if urlsplit(link).hostname not in {"wsa.roe.ac.uk", "surveys.roe.ac.uk"}:
                raise ArchiveError("Unexpected WFCAM result-file host")
            def convert(r):
                return frame(job, r, id=r.get("multiframeID"), survey=database, instrument="UKIRT/WFCAM",
                             ra=float(r["raBase"])*15, dec=float(r["decBase"]), startMjd=numeric(r.get("mjdObs")),
                             endMjd=observation_time(r.get("dateEnd")), exposureSeconds=numeric(r.get("expTime")), filter=r.get("filterName"),
                             fovRadius=1, productKind=r.get("frameType"), url=url("http://wsa.roe.ac.uk/cgi-bin/fits_download.cgi", dict(file=r["fileName"], mfid=r["multiframeID"], database=database)), linkKind="FITS")
            yield from convert_rows(votable(t.request(link, auth={}), s), convert, job, x, s)


def query_igaps(job, x, t, s):
    from astropy.table import Table
    # IGAPS supersedes the older IPHAS-only release and also includes UVEX.
    path = metadata(t, "http://www.igapsimages.org/data/images/igaps-images.fits.gz", binary=True)
    global _igaps_table
    identity = t.calls[-1].get("sha256") or str(Path(path).resolve())
    with _index_lock:
        if _igaps_table is None or _igaps_table[0] != identity:
            _igaps_table = (identity, Table.read(path, format="fits"))
        table = _igaps_table[1]
    required = {"run", "ccd", "ra", "dec", "url", "utstart"}
    if not required.issubset(table.colnames):
        raise ArchiveError("IGAPS metadata table schema changed")
    import numpy as np
    w = job.window
    ra, dec = np.asarray(table["ra"], dtype=float), np.asarray(table["dec"], dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec)
    hav = np.sin(np.radians(dec-w["dec"])/2)**2 + np.cos(np.radians(dec))*math.cos(math.radians(w["dec"]))*np.sin(np.radians(ra-w["ra"])/2)**2
    inside = valid & (2*np.degrees(np.arcsin(np.sqrt(np.clip(hav, 0, 1)))) <= min(180, w["radius"]+.25))
    s["received"] += len(table)
    s["rejected"] += int(np.count_nonzero(~valid))
    s["excluded"] += int(np.count_nonzero(valid & ~inside))
    def rows():
        for row in table[inside]:
            yield {k: str(row[k]) for k in table.colnames}
    def convert(r):
        return frame(job, r, id=f"r{r['run']}-{r['ccd']}", survey="IGAPS (IPHAS/UVEX)", instrument="INT/WFC",
                     ra=float(r["ra"]), dec=float(r["dec"]), startMjd=mjd(r["utstart"]),
                     exposureSeconds=numeric(r.get("exptime")), filter=r.get("band") or r.get("filter"), fovRadius=.25, url=r["url"], linkKind="FITS")
    yield from convert_rows(rows(), convert, job, x, s)


def query_scanpi(job, x, t, s):
    w = job.window
    if w["start"] >= mjd("1984-01-01") or w["end"] <= mjd("1983-01-01"):
        s["warnings"].append("Requested dates do not overlap the 1983 IRAS mission.")
        return
    if w["radius"] or "points" in w:
        s["partial"] = "SCANPI samples the query center only; it does not establish complete coverage of a region or orbit corridor."
    query = url("https://irsa.ipac.caltech.edu/cgi-bin/Scanpi/nph-scanpi", dict(locstr=f"{w['ra']} {w['dec']:+} equ j2000", band1="12um", band2="25um", band3="60um", band4="100um", mode="PI", det_distance=3))
    root = ET.fromstring(t.request(query))
    if root.get("status") != "ok":
        raise ArchiveError("SCANPI did not report a successful query")
    tables = root.findall(".//summaryTBL")
    if not tables:
        raise ArchiveError("SCANPI response contains no scan metadata links")
    for band, element in enumerate(tables, 1):
        link = (element.text or "").strip()
        if urlsplit(link).hostname != "irsa.ipac.caltech.edu":
            raise ArchiveError("Unexpected SCANPI metadata host")
        def convert(r):
            href = re.search(r'href=["\']([^"\']+)', r.get("table", ""), re.I)
            if not href:
                raise ValueError("Scan lacks data-table link")
            access = urljoin("https://irsa.ipac.caltech.edu", html.unescape(href[1]))
            # 'scan' is a query-local row number; native detector/scan ID is in the filename.
            identity = Path(urlsplit(access).path).name.removesuffix("_srcfit.tbl")
            when = datetime.strptime(r["utcs_start"].strip(), "%m/%d/%y %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            return frame(job, r, id=identity, survey="IRAS SCANPI", instrument="IRAS", startMjd=mjd(when.isoformat()),
                         ra=w["ra"], dec=w["dec"], filter=str(band), productKind="detector scan", url=access, linkKind="temporary scan table", retrievalRequest={"url": query, "method": "GET"})
        rows = (r for r in ipac(t.request(link), s, generic=True) if r.get("utcs_start") not in {"N/A", "", None})
        yield from convert_rows(rows, convert, job, x, s)


def query_lasco(job, x, t, s):
    from astropy.coordinates import get_sun
    from astropy.time import Time
    w = job.window
    begin = max(w["start"], mjd("1995-12-02"))
    def compact(when):
        return datetime.fromtimestamp((when-40587)*86400, timezone.utc).strftime("%Y%m%d%H%M%S")
    def parse_time(text):
        return mjd(datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat())
    s["warnings"].append("LASCO uses a conservative solar-pointing envelope; occulting disks, detector edges, and SOHO parallax require FITS WCS verification. VSO native IDs are supplied for GetData retrieval.")
    while begin < w["end"]:
        end = min(begin+1, w["end"])
        sun = get_sun(Time((begin+end)/2, format="mjd", scale="utc"))
        # Sun moves <=~1.1 deg/day; 13 degrees encloses C3 image corners and daily motion.
        if distance(w["ra"], w["dec"], sun.ra.deg, sun.dec.deg) <= w["radius"]+13:
            body = f'<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" xmlns:v="http://virtualsolar.org/VSO/VSOi"><soap:Body><v:Query><body><version>1.4</version><block><time><start>{compact(begin)}</start><end>{compact(end)}</end></time><instrument>LASCO</instrument></block></body></v:Query></soap:Body></soap:Envelope>'
            root = ET.fromstring(t.request(VSO, data=body, headers={"Content-Type": "text/xml", "SOAPAction": "http://virtualsolar.org/VSO/VSOi/Query"}))
            providers = root.findall(".//provideritem")
            if not providers:
                raise ArchiveError("VSO returned no provider status")
            for provider in providers:
                if provider.findtext("error"):
                    raise ArchiveError("VSO provider reported an error")
                found, returned = provider.findtext("no_of_records_found"), provider.findtext("no_of_records_returned")
                if found is None or returned is None or int(found) != int(returned):
                    s["incomplete"] = "VSO record counts do not certify a full response"
                records = provider.findall(".//recorditem")
                s["received"] += len(records)
                if returned is not None and len(records) != int(returned):
                    s["incomplete"] = "VSO returned fewer rows than its count"
                def convert(r):
                    identity = r.findtext("fileid")
                    start, stop = parse_time(r.findtext("time/start")), parse_time(r.findtext("time/end"))
                    center = get_sun(Time((start+stop)/2, format="mjd", scale="utc"))
                    raw = {"fileid": identity, "provider": r.findtext("provider"), "time_start": r.findtext("time/start"), "time_end": r.findtext("time/end")}
                    return frame(job, raw, id=identity, survey="SOHO LASCO", instrument="LASCO", startMjd=start, endMjd=stop,
                                 exposureSeconds=(stop-start)*86400, ra=float(center.ra.deg), dec=float(center.dec.deg), fovRadius=12,
                                 url=VSO, linkKind="VSO GetData request", retrievalRequest={"protocol": "VSO SOAP GetData", "provider": r.findtext("provider"), "fileid": identity})
                yield from convert_rows(records, convert, job, x, s)
        begin = end
