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
