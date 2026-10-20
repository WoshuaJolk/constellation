"""Every imaging dataset in the constellation inventory, including explicit gaps.

An adapter being callable does not certify that every exposure in its parent
archive is indexed, released, accessible, or detectable at a requested position.
"""

# dataset, provider IDs, scope / remaining limitation
DATASETS = [
    ("DECam", ["noirlab", "ssois", "cadc"], "NOIRLab instcal images; SSOIS and CADC indexes provide cross-checks."),
    ("Mosaic, Mosaic II, Mosaic3, 90Prime", ["noirlab:mosaic", "noirlab:90prime"], "All five Mosaic archive instrument names plus 90Prime, across image processing levels."),
    ("NEWFIRM", ["noirlab:newfirm"], "Public NOIRLab image records; conservative pointing envelope."),
    ("Pan-STARRS1 warps", ["catch", "mast", "ssois"], "Released/indexed PS1 subsets; not a certification of every warp or later PS1/PS2 observations."),
    ("ZTF", ["ztf", "most:ztf", "ssois", "irsa"], "Public science quadrants and moving-object indexes."),
    ("PTF, iPTF", ["most:ptf", "irsa"], "Released IRSA products and MOST PTF index."),
    ("SDSS imaging", ["sdss", "ssois"], "Direct SIA1 FITS discovery omits observation dates; date coverage remains unverified unless another index supplies it."),
    ("Hyper Suprime-Cam", ["smoka", "hsc", "ssois"], "SMOKA direct CCDs plus HSC-SSP PDR3 frame tables; HSC CAS requires an account."),
    ("Suprime-Cam", ["smoka", "ssois"], "Direct SMOKA CCDs and SSOIS cross-check."),
    ("MegaCam, CFH12K, WIRCam", ["cadc", "ssois"], "Released CADC collections and SSOIS indexed holdings."),
    ("OmegaCAM", ["eso"], "Public ESO ObsCore image products; inspect processing level."),
    ("VIRCAM", ["eso"], "Public ESO ObsCore image products; inspect processing level."),
    ("WFI", ["eso"], "Public ESO ObsCore image products; inspect processing level."),
    ("SkyMapper", ["catch", "ssois"], "CATCH DR4 and SSOIS indexed subsets."),
    ("Catalina Sky Survey", ["catch"], "CATCH/PDS indexed site and release coverage only."),
    ("Spacewatch", ["catch"], "CATCH/PDS released subset only."),
    ("LONEOS", ["catch"], "CATCH/PDS released subset only."),
    ("NEAT", ["catch"], "CATCH/PDS released subset only."),
    ("INT WFC", ["iphas"], "527,736 IGAPS IPHAS/UVEX CCD records via the public HTTP metadata index, with UTC epochs. Does not cover every ING program."),
    ("Las Cumbres", ["lco"], "Public EXPOSE records with cursor pagination and archive polygons; optional token includes authorized proprietary holdings."),
    ("GMOS", ["gemini"], "Canonical Gemini imaging OBJECT files; conservative camera envelope."),
    ("Keck imagers", ["koa"], "Nine imaging instruments queried separately through KOA SIA2; most spatial matches use archive-estimated footprints."),
    ("WFCAM", ["wfcam"], "UKIDSS DR11PLUS and UHS DR3 full SQL result files; public HTTP access only."),
    ("2MASS", ["irsa", "most:2mass"], "Released IRSA images and MOST index; inspect atlas versus individual exposure products."),
    ("DSS plates", ["dss"], "DSS1/2 cutout descriptors, not a complete plate index; no observation epochs in SIA response; maximum 40 arcmin per axis."),
    ("DASCH plates", ["dasch"], "DR7 queryexps point lookup; nonzero fields and tracks are explicitly partial; native plate/exposure/solution identities retained."),
    ("WISE", ["irsa", "most:wise_merge"], "IRSA image products and merged WISE/NEOWISE moving-object index."),
    ("NEOWISE", ["irsa", "most:wise_merge"], "IRSA image products and merged WISE/NEOWISE moving-object index."),
    ("SPHEREx", ["irsa", "most:spherex"], "Released IRSA and MOST products, subject to current release coverage."),
    ("TESS FFIs", ["mast"], "MAST CAOM image index/product manifests; cadence-level FFI completeness is not independently certified."),
    ("Kepler, K2 FFIs", ["mast"], "MAST CAOM image index/product manifests; released FFI subset."),
    ("GALEX", ["mast"], "MAST CAOM image products, including combined products."),
    ("Spitzer IRAC", ["irsa", "most:spitzer_bcd"], "IRSA products and MOST BCD moving-object index."),
    ("Hubble imagers", ["mast"], "MAST CAOM observations/product lists; proprietary rights and detector masks apply."),
    ("JWST imagers", ["mast"], "MAST CAOM observations/product lists; proprietary rights and detector masks apply."),
    ("Euclid Q1", ["euclid"], "ESA ObsCore includes individual calibrated VIS/NISP files; native file names avoid merging filters within a visit."),
    ("Rubin data previews", ["rubin"], "RSP SIA2 requires a token with read:image scope and appropriate data rights; no authenticated live query validated."),
    ("APPLAUSE plates", ["applause"], "DR4 plate solutions joined to exposure dates; multiple solutions/exposures remain association candidates."),
    ("WIYN ODI", ["noirlab:odi"], "Both odi and (p)odi instrument names; all public image processing levels."),
    ("NGTS", ["eso"], "ESO public products include combined survey images; individual cadence exposure access is not certified."),
    ("FORS2, HAWK-I", ["eso"], "Public ESO ObsCore imaging products."),
    ("ATLAS", ["catch"], "PDS/CATCH released sites and dates; excludes unpublished operational holdings."),
    ("NEOSSat", ["cadc"], "CADC public image collections; use observer-specific apparent positions."),
    ("Kepler, K2 target pixels", ["mast:targetpixels"], "Kepler/K2 target-pixel FITS products, not light curves; cubes require cadence and actual pixel-mask inspection."),
    ("IRAS scans", ["scanpi"], "SCANPI detector scan tables and UTC timestamps for the 1983 mission; point-only sampling, not image footprints."),
    ("AKARI", ["akari", "irsa"], "JAXA phase 1/2 IRC pointed observation bundles plus IRSA released maps; phase 3 exposures are not covered by the direct adapter."),
    ("Herschel imaging", ["herschel"], "HSA observation bundles with epoch ranges and archive polygons; bundles may also contain spectra."),
    ("SOFIA imaging", ["irsa", "most:sofia"], "Released IRSA imagery and MOST index."),
    ("J-PLUS", ["jplus"], "DR4 reduced individual images via SIA1, with observation MJD; full detector footprints require FITS WCS."),
    ("S-PLUS", ["splus"], "Public ADSS image-collection discovery, optional credentials for access; stacks lack constituent epochs, so time coverage remains partial."),
    ("SOHO LASCO", ["lasco"], "VSO frame IDs and UTC times, conservative solar envelope; coronagraph masks and SOHO observer geometry remain unverified."),
]


def coverage(checks=None):
    from .model import PROVIDERS
    states = {}
    for check in checks or []:
        states.setdefault(check["provider"], set()).add(check["status"])
    rows = []
    for name, providers, scope in DATASETS:
        row = {"dataset": name, "providers": providers, "scope": scope}
        if checks is not None:
            row["providerQueryStates"] = {p: sorted(states.get(p, {"not queried"})) for p in providers}
        rows.append(row)
    return {"inventoryDate": "2026-10-20", "datasetCount": len(rows), "providerCount": len(PROVIDERS),
            "allAstronomicalObservationsCovered": False,
            "interpretation": "A provider query state describes its index, not proof of complete dataset coverage. Read each scope and each query receipt.",
            "datasets": rows}
