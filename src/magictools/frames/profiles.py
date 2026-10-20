"""Verified archive endpoints and their protocol-specific limits."""
FIELD_MODES = ["field", "ephemeris"]
NOIRLAB_INSTRUMENTS = {
    "noirlab": ["decam"],
    "noirlab:mosaic": ["mosaic", "mosaic_1", "mosaic_1_1", "mosaic_2", "mosaic3"],
    "noirlab:90prime": ["90prime"],
    "noirlab:newfirm": ["newfirm"],
    "noirlab:odi": ["odi", "(p)odi"],
}
SIA2 = {
    "koa": "https://koa.ipac.caltech.edu/cgi-bin/VOServ/nph-searchImage",
    "rubin": "https://data.lsst.cloud/api/sia/query",
}
SIA1 = {
    "sdss": "https://skyserver.sdss.org/dr18/SkyServerWS/SIAP/getAllSIAPInfo",
    "dss": "https://archive.eso.org/bin/dss_sia/dss.sia",
    "jplus": "https://archive.cefca.es/catalogues/vo/siap/jplus-dr4/reduced",
}
EXTRA_PROVIDERS = {
    "mast:targetpixels": {"name": "MAST Kepler/K2 target-pixel products", "modes": FIELD_MODES, "collections": ["Kepler", "K2"]},
    **{k: {"name": "NOIRLab / " + k.split(":")[1], "modes": FIELD_MODES, "collections": v}
       for k, v in NOIRLAB_INSTRUMENTS.items() if k != "noirlab"},
    "gemini": {"name": "Gemini direct image archive", "modes": FIELD_MODES, "collections": ["GMOS-N", "GMOS-S", "Gemini imagers"]},
    "koa": {"name": "Keck Observatory Archive SIA2", "modes": FIELD_MODES, "collections": ["Keck imaging instruments"]},
    "lco": {"name": "Las Cumbres image archive", "modes": FIELD_MODES, "collections": ["LCO network"]},
    "wfcam": {"name": "WFCAM Science Archive", "modes": FIELD_MODES, "collections": ["UKIDSS DR11PLUS", "UHS"]},
    "sdss": {"name": "SDSS DR18 image service", "modes": FIELD_MODES, "collections": ["SDSS imaging"]},
    "dss": {"name": "ESO Digitized Sky Survey", "modes": FIELD_MODES, "collections": ["DSS1", "DSS2"]},
    "dasch": {"name": "DASCH DR7 exposure index", "modes": FIELD_MODES, "collections": ["Harvard photographic plates"]},
    "applause": {"name": "APPLAUSE DR4 plate archive", "modes": FIELD_MODES, "collections": ["APPLAUSE plates"]},
    "iphas": {"name": "IGAPS INT/WFC image metadata", "modes": FIELD_MODES, "collections": ["INT/WFC IPHAS", "INT/WFC UVEX"]},
    "euclid": {"name": "ESA Euclid ObsCore", "modes": FIELD_MODES, "collections": ["Euclid public releases"]},
    "herschel": {"name": "ESA Herschel observation archive", "modes": FIELD_MODES, "collections": ["PACS", "SPIRE"]},
    "rubin": {"name": "Rubin Science Platform SIA2", "modes": FIELD_MODES, "collections": ["Authorized RSP image collections"], "credentials": ["RSP_API_TOKEN"]},
    "jplus": {"name": "J-PLUS DR4 individual reduced images", "modes": FIELD_MODES, "collections": ["J-PLUS DR4"]},
    "splus": {"name": "S-PLUS image collections", "modes": FIELD_MODES, "collections": ["Authorized S-PLUS releases"], "credentials": ["SPLUS_USERNAME", "SPLUS_PASSWORD", "SPLUS_API_TOKEN"]},
    "scanpi": {"name": "IRAS SCANPI scan metadata", "modes": FIELD_MODES, "collections": ["IRAS calibrated scans"]},
    "akari": {"name": "AKARI pointed observation archive", "modes": FIELD_MODES, "collections": ["AKARI IRC pointed images"]},
    "lasco": {"name": "SOHO LASCO / VSO", "modes": FIELD_MODES, "collections": ["LASCO C1", "LASCO C2", "LASCO C3"]},
}

KOA_INSTRUMENTS = ["DEIMOS", "ESI", "LRIS", "LWS", "MOSFIRE", "NIRC", "NIRC2", "NIRSPEC", "OSIRIS"]

# Protocol profiles are also usable as examples for user-supplied services.
VO_PROFILES = {
    "alma": {"name": "ALMA Science Archive", "protocol": "obscore", "endpoint": "https://almascience.eso.org/tap/sync",
             "identityColumns": ["obs_id"], "combinedProducts": True,
             "scopeWarning": "ALMA rows describe science products / spectral windows. Resolve DataLink for files; member exposure times and moving-target tracking require inspection."},
    "heasarc": {"name": "HEASARC multi-mission images", "protocol": "obscore", "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync",
                "includeUnknownAxes": False,
                "scopeWarning": "Image products with indexed footprints and times in HEASARC ObsCore; excludes unknown axes, event lists, and raw observations. The null-inclusive query timed out in live validation."},
    "casda": {"name": "CSIRO ASKAP Science Data Archive", "protocol": "obscore", "endpoint": "https://casda.csiro.au/casda_vo_tools/tap/sync",
              "combinedProducts": True, "scopeWarning": "Released ASKAP images and cubes; some cubelets omit time metadata. DataLink may require an archive staging request."},
    "nrao": {"name": "NRAO images and visibility scans", "protocol": "obscore", "endpoint": "https://data-query.nrao.edu/tap/sync",
             "table": "tap_schema.obscore", "spatial": "pointing", "productTypes": ["image", "cube", "visibility"],
             "rowIdentityColumns": ["obs_id", "obs_publisher_did", "scan_num", "t_min", "freq_min", "freq_max", "pol_states"],
             "combinedProducts": True, "linkKind": "archive query",
             "scopeWarning": "NRAO includes visibilities requiring calibration/imaging and VLASS image products. FOV can be inaccurate for non-VLA data; select files through the archive portal."},
    "gavo": {"name": "GAVO heterogeneous image collections", "protocol": "obscore", "endpoint": "https://dc.g-vo.org/tap/sync",
             "scopeWarning": "Published GAVO image/cube collections only; consult native metadata for calibration and time systems."},
    "chandra": {"name": "Chandra direct archive images", "protocol": "sia1", "endpoint": "https://cda.harvard.edu/cxcsiap/queryImages",
                "scopeWarning": "Chandra archive image products; event files and detector masks require archive follow-up."},
    "xmm": {"name": "XMM-Newton pointed observation images", "protocol": "sia1", "endpoint": "https://nxsa.esac.esa.int/nxsa-sl/servlet/siap",
            "scopeWarning": "Pointed-observation image bundles; slew observations are a separate service."},
    "xmm:slew": {"name": "XMM-Newton slew images", "protocol": "sia1", "endpoint": "https://nxsa.esac.esa.int/nxsa-sl/servlet/siap-slew",
                 "scopeWarning": "Slew image products; constituent exposure times and scan geometry require inspection."},
}
EXTRA_PROVIDERS.update({k: {"name": v["name"], "modes": FIELD_MODES, "collections": [v["name"]], "protocol": v["protocol"], "endpoint": v["endpoint"]}
                        for k, v in VO_PROFILES.items()})
EXTRA_PROVIDERS.update({
    "kmtnet": {"name": "KASI KMTNet image archive", "modes": FIELD_MODES, "collections": ["Released KMTNet images"]},
    "miris": {"name": "KASI MIRIS image archive", "modes": FIELD_MODES, "collections": ["MIRIS"]},
})
