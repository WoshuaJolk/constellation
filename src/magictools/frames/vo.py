"""Configurable IVOA discovery without mutating the process-wide provider registry."""
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .adapters import convert_rows, frame
from .geometry import region_intersects
from .parsers import numeric, safe_url, votable
from .transport import ArchiveError, Blocked


def endpoint_url(endpoint, parameters):
    parts = urlsplit(endpoint)
    supplied = list(parameters.items()) if isinstance(parameters, dict) else parameters
    keys = {k.upper() for k, _ in supplied}
    params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.upper() not in keys]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params + supplied, doseq=True), ""))


def validate_services(services):
    if not isinstance(services, dict):
        raise ValueError("services must map vo:<name> to endpoint configurations")
    allowed = {"name", "protocol", "endpoint", "table", "collections", "productTypes", "credentialProfile", "maxrec", "spatial", "timeScale", "includeUnknownAxes"}
    for name, config in services.items():
        if not re.fullmatch(r"vo:[a-z0-9][a-z0-9_-]*", name) or not isinstance(config, dict):
            raise ValueError("Custom service IDs must be vo:<lowercase-name>")
        if set(config) - allowed:
            raise ValueError("Unknown service options: " + ", ".join(sorted(set(config) - allowed)))
        validate_config(config)
    return services


def validate_config(config):
    if config.get("protocol") not in {"obscore", "sia1", "sia2"}:
        raise ValueError("Service protocol must be obscore, sia1, or sia2")
    endpoint = config.get("endpoint")
    if not isinstance(endpoint, str) or not safe_url(endpoint) or urlsplit(endpoint).fragment:
        raise ValueError("Service endpoint must be an HTTP(S) URL without embedded credentials or fragment")
    if any(re.search(r"token|password|secret|api.?key|signature", k, re.I) for k, _ in parse_qsl(urlsplit(endpoint).query)):
        raise ValueError("Use credentialProfile, not credentials in the endpoint URL")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)?", config.get("table", "ivoa.obscore")):
        raise ValueError("ObsCore table must be a simple schema.table identifier")
    for key in ("collections", "productTypes"):
        if key in config and (not isinstance(config[key], list) or not config[key] or any(not isinstance(v, str) or not v for v in config[key])):
            raise ValueError(f"{key} must be a nonempty list of strings")
    if set(config.get("productTypes", ["image", "cube"])) - {"image", "cube", "visibility"}:
        raise ValueError("productTypes supports image, cube, and visibility")
    if config.get("spatial", "footprint") not in {"footprint", "pointing"}:
        raise ValueError("spatial must be footprint or pointing")
    if config.get("timeScale", "UTC") not in {"UTC", "archive-defined"}:
        raise ValueError("timeScale must be UTC or archive-defined")
    if not isinstance(config.get("includeUnknownAxes", True), bool):
        raise ValueError("includeUnknownAxes must be boolean")
    if "credentialProfile" in config and (not isinstance(config["credentialProfile"], str) or not config["credentialProfile"]):
        raise ValueError("credentialProfile must name a profile in the private credential file")
    limit = config.get("maxrec", 2000000)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("maxrec must be a positive server row limit; overflow remains incomplete")


def service_auth(config, transport):
    """An explicit origin grant lives with the secret, never in an untrusted manifest."""
    name = config.get("credentialProfile")
    if not name:
        return {}  # Do not send any host's ambient credentials to custom endpoints.
    credentials = transport.credentials
    origin = urlunsplit((*urlsplit(config["endpoint"])[:2], "", "", ""))
    profile = credentials.profiles.get(name, {}) if credentials else {}
    if not origin.startswith("https://") or origin not in profile.get("allowedOrigins", []):
        raise Blocked("Custom service credentials require an exact HTTPS allowedOrigins entry in the private credential profile")
    values = credentials.get(name)
    if not (values.get("token") or (values.get("username") and values.get("password"))):
        raise Blocked(f"Credential profile {name} needs a bearer token or username/password")
    return {k: v for k, v in values.items() if k in {"token", "username", "password"}}


def literal(value):
    return "'" + value.replace("'", "''") + "'"


def obscore_query(config, window):
    w = window
    # NULL axes remain candidates. Excluding them would manufacture completeness.
    envelope = f"(s_ra IS NULL OR s_dec IS NULL OR s_fov IS NULL OR 1=CONTAINS(POINT('ICRS',s_ra,s_dec),CIRCLE('ICRS',{w['ra']},{w['dec']},{w['radius']}+s_fov/2)))"
    spatial = envelope if config.get("spatial") == "pointing" else f"(INTERSECTS(s_region,CIRCLE('ICRS',{w['ra']},{w['dec']},{w['radius']}))=1 OR (s_region IS NULL AND {envelope}))"
    types = ",".join(literal(v) for v in config.get("productTypes", ["image", "cube"]))
    clauses = [f"dataproduct_type IN ({types})", spatial,
               f"(t_min IS NULL OR t_min<{w['end']})", f"(t_max IS NULL OR t_max<t_min OR t_max>={w['start']})"]
    if config.get("includeUnknownAxes") is False:
        spatial = f"INTERSECTS(s_region,CIRCLE('ICRS',{w['ra']},{w['dec']},{w['radius']}))=1" if config.get("spatial") != "pointing" else f"1=CONTAINS(POINT('ICRS',s_ra,s_dec),CIRCLE('ICRS',{w['ra']},{w['dec']},{w['radius']}+s_fov/2))"
        clauses = [f"dataproduct_type IN ({types})", spatial, f"t_min<{w['end']}", f"t_max>={w['start']}"]
    if config.get("collections"):
        clauses.append("obs_collection IN (" + ",".join(literal(v) for v in config["collections"]) + ")")
    return f"SELECT * FROM {config.get('table', 'ivoa.obscore')} WHERE " + " AND ".join(clauses)


def normalize_region(value):
    # HEASARC publishes ADQL geometry literals; other services use STC-S.
    if value and re.match(r"^(CIRCLE|POLYGON)\s*\(", value, re.I):
        return re.sub(r"[(),'\"]", " ", value)
    return value


def vo_row(job, row, config):
    r = {k.lower(): v for k, v in row.items()}
    # obs_id often names a visit. ALMA's publisher DID instead names a whole
    # proposal, so that verified profile explicitly chooses obs_id.
    identity = next((r.get(k) for k in config.get("identityColumns", ["obs_publisher_did", "access_url", "obs_id"]) if r.get(k)), None)
    if config.get("rowIdentityColumns"):
        identity = "|".join(str(r.get(k) or "") for k in config["rowIdentityColumns"])
        if not identity.strip("|"):
            identity = None
    kind = r.get("dataproduct_type") or "unknown"
    f = frame(job, row, id=identity, survey=r.get("obs_collection") or job.provider,
              instrument=r.get("instrument_name") or r.get("facility_name") or job.provider,
              startMjd=numeric(r.get("t_min")), endMjd=numeric(r.get("t_max")),
              exposureSeconds=numeric(r.get("t_exptime")), timeScale=config.get("timeScale", "UTC"),
              ra=numeric(r.get("s_ra")), dec=numeric(r.get("s_dec")), footprint=normalize_region(r.get("s_region")),
              url=r.get("access_url"), productKind=kind, calibrationLevel=numeric(r.get("calib_level")),
              linkKind="DataLink" if "datalink" in (r.get("access_format") or "").lower() else config.get("linkKind", "archive product"),
              wavelengthMinMeters=numeric(r.get("em_min")), wavelengthMaxMeters=numeric(r.get("em_max")),
              accessRights=r.get("data_rights") or "archive-defined", releaseDate=r.get("obs_release_date"))
    fov = numeric(r.get("s_fov"))
    if f["startMjd"] is not None and f["endMjd"] is not None and f["endMjd"] < f["startMjd"]:
        f["endMjd"] = None
        f["timeSpanUnknown"] = True
    if fov is not None and fov > 0:
        f["fovRadius"] = fov/2
    # A data cube or radio scan is not necessarily a single exposure. Preserve
    # it for moving-target follow-up without testing only a representative epoch.
    if kind in {"cube", "visibility"} or config.get("combinedProducts") or (numeric(r.get("calib_level")) or 0) >= 3:
        f["timeSpanUnknown"] = True
    if config.get("linkKind") == "archive query":
        f["retrieval"] = {"archive": f["url"], "nativeId": identity, "requiresArchiveSelection": True}
    return f


def execute_vo(job, x, transport, s, config):
    auth = service_auth(config, transport)
    w = job.window
    protocol = config["protocol"]
    s["service"] = {k: v for k, v in config.items() if k != "credentialProfile"}
    if config.get("scopeWarning"):
        s["warnings"].append(config["scopeWarning"])
    if config.get("includeUnknownAxes") is False:
        s["warnings"].append("includeUnknownAxes=false: products without indexed spatial/time axes are outside this query's scope.")
    if protocol == "sia1":
        from .extended import query_sia1
        s["canSplitTime"] = False
        yield from query_sia1(job, x, transport, s, endpoint=config["endpoint"], auth=auth, config=config)
    else:
        if protocol == "obscore":
            p = dict(REQUEST="doQuery", LANG="ADQL", FORMAT="votable", MAXREC=config.get("maxrec", 2000000), QUERY=obscore_query(config, w))
        else:
            p = dict(POS=f"CIRCLE {w['ra']} {w['dec']} {w['radius']}", TIME=f"{w['start']} {w['end']}", MAXREC=config.get("maxrec", 2000000),
                     DPTYPE=config.get("productTypes", ["image", "cube"]))
            if config.get("collections"):
                p["COLLECTION"] = config["collections"]
            s["warnings"].append("SIA2 TIME selection can omit products without archive time metadata.")
        text = transport.request(endpoint_url(config["endpoint"], p), auth=auth)
        def rows():
            for row in votable(text, s):
                if str(row.get("dataproduct_type", "")).lower() not in config.get("productTypes", ["image", "cube"]):
                    s["excluded"] += 1
                    continue
                yield row
        yield from convert_rows(rows(), lambda r: vo_row(job, r, config), job, x, s)
        required = {"dataproduct_type", "s_ra", "s_dec", "t_min", "t_max"}
        if not required.issubset({c.lower() for c in s.get("columns", [])}):
            raise ArchiveError("Service did not return the required ObsCore columns")
    if ("OK" not in s.get("queryStatuses", []) or s.get("missingQueryStatusResponses")) and not s.get("incomplete"):
        s["partial"] = "VOTable lacks QUERY_STATUS=OK; result exhaustion is unverified"
    s["indexExhausted"] = "OK" in s.get("queryStatuses", []) and not s.get("incomplete") and not s.get("missingQueryStatusResponses")


REGISTRY = "https://dc.g-vo.org/tap/sync"


def discover(transport, text="", protocol="all"):
    """Return registry candidates, not automatically trusted or complete archives."""
    from .adapters import stats
    s = stats()
    where = "intf_role='std' AND (standard_id LIKE 'ivo://ivoa.net/std/sia%' OR standard_id='ivo://ivoa.net/std/tap')"
    if text:
        where += " AND 1=ivo_nocasematch(res_title," + literal("%"+text+"%") + ")"
    if protocol != "all":
        where += " AND standard_id" + ("='ivo://ivoa.net/std/tap'" if protocol == "obscore" else " LIKE 'ivo://ivoa.net/std/sia%'")
    query = "SELECT DISTINCT ivoid,res_title,access_url,standard_id FROM rr.resource NATURAL JOIN rr.capability NATURAL JOIN rr.interface WHERE " + where
    rows = list(votable(transport.request(endpoint_url(REGISTRY, dict(REQUEST="doQuery", LANG="ADQL", FORMAT="votable", MAXREC=100000, QUERY=query)), auth={}), s))
    candidates = []
    for row in rows:
        standard = row["standard_id"]
        p = "obscore" if standard.endswith("/tap") else "sia2" if "query-2" in standard else "sia1"
        if protocol not in {"all", p}:
            continue
        endpoint = row["access_url"].rstrip("?&")
        if p == "obscore":
            endpoint = endpoint.rstrip("/") + "/sync"
        candidates.append({**row, "verified": False, "service": {"name": row["res_title"], "protocol": p, "endpoint": endpoint},
                           "nextStep": "Inspect TAP_SCHEMA for ObsCore before searching" if p == "obscore" else "Run a bounded search and inspect the coverage assessment"})
    return {"status": "incomplete" if s.get("incomplete") else "complete", "candidates": candidates,
            "registryOnly": True, "stats": s, "requests": transport.calls}


def inspect_service(transport, config):
    from .adapters import stats
    validate_config(config)
    auth, s = service_auth(config, transport), stats()
    if config["protocol"] == "obscore":
        query = "SELECT table_name,column_name,datatype,unit,ucd FROM TAP_SCHEMA.columns WHERE table_name=" + literal(config.get("table", "ivoa.obscore"))
        params = dict(REQUEST="doQuery", LANG="ADQL", FORMAT="votable", QUERY=query, MAXREC=10000)
    else:
        params = {"FORMAT": "METADATA"} if config["protocol"] == "sia1" else {"MAXREC": 0}
    rows = list(votable(transport.request(endpoint_url(config["endpoint"], params), auth=auth), s))
    columns = {r["column_name"].lower() for r in rows} if config["protocol"] == "obscore" else {k.lower() for k in s.get("columns", [])}
    required = {"dataproduct_type", "obs_id", "s_ra", "s_dec", "t_min", "t_max", "access_url"}
    return {"service": config, "schemaCompatible": required.issubset(columns) if config["protocol"] != "sia1" else None,
            "missingColumns": sorted(required-columns) if config["protocol"] != "sia1" else [],
            "columns": rows if config["protocol"] == "obscore" else s.get("fields", {}),
            "coverageEstablished": False, "stats": s, "requests": transport.calls}


def datalinks(transport, endpoint):
    """Resolve a product manifest without downloading image pixels or executing services."""
    from .adapters import stats
    if not safe_url(endpoint):
        raise ValueError("DataLink URL must be HTTP(S), without embedded credentials")
    s = stats()
    rows = list(votable(transport.request(endpoint, auth={}), s))
    if not {"semantics", "access_url", "error_message", "service_def"}.issubset(s.get("columns", [])):
        raise ArchiveError("Response is not a DataLink links table")
    links = [{**r, "access_url": safe_url(r.get("access_url"))} for r in rows]
    return {"links": links, "status": "incomplete" if s.get("incomplete") or any(r.get("error_message") for r in rows) else "complete",
            "serviceDescriptorsInRawResponse": True, "requests": transport.calls}
