export type Step = 0 | 1 | 2; // empty, half, full

export type Dataset = {
  name: string;
  instrument: string;
  years: string;
  sky: string;
  depth: string;
  exposures: string;
  volume: string;
  tb: number;
  where: string;
  url: string;
  step: Step;
  note?: string;
};

export type Record = {
  name: string;
  holds: string;
  size: string;
  years: string;
  where: string;
  url: string;
  step: Step;
  note?: string;
};


// Depth is a single exposure at 5 sigma; counts and volumes are rounded from archive pages (2026-10).
export const DATASETS: Dataset[] = [
  { name: "DECam", instrument: "Blanco 4 m", years: "2012–", sky: "south + equator, 1 sq deg", depth: "r 23.5", exposures: "7.1M", volume: "1.7 PB", tb: 1700, where: "NOIRLab · Data Lab", url: "https://astroarchive.noirlab.edu/", step: 0, note: "DES, DECaLS, DELVE, DEEP and every PI program; 93% of the NOIRLab archive" },
  { name: "Mosaic, Mosaic II, Mosaic3, 90Prime", instrument: "Mayall 4 m, Blanco 4 m, Bok 2.3 m", years: "2003–2019", sky: "both hemispheres", depth: "r 23", exposures: "~0.5M", volume: "~50 TB", tb: 50, where: "NOIRLab · Data Lab", url: "https://astroarchive.noirlab.edu/", step: 0, note: "BASS and MzLS single epochs" },
  { name: "NEWFIRM", instrument: "Mayall, Blanco", years: "2007–", sky: "pointed, NIR", depth: "Ks 20", exposures: "~1M", volume: "~20 TB", tb: 20, where: "NOIRLab · Data Lab", url: "https://astroarchive.noirlab.edu/", step: 0 },
  { name: "Pan-STARRS1 warps", instrument: "PS1 1.8 m", years: "2010–2014", sky: "dec > −30, 3π", depth: "r 22", exposures: "24M", volume: "1.6 PB", tb: 1600, where: "MAST", url: "https://outerspace.stsci.edu/display/PANSTARRS/", step: 0, note: "DR2 single-epoch warps; later PS1 and all PS2 data are not released" },
  { name: "ZTF", instrument: "P48", years: "2018–", sky: "dec > −31, 47 sq deg", depth: "r 20.5", exposures: "70M quadrants", volume: "~1 PB", tb: 1000, where: "IRSA · Fornax", url: "https://irsa.ipac.caltech.edu/Missions/ztf.html", step: 0, note: "DR24; alerts carry difference cutouts from day one" },
  { name: "PTF, iPTF", instrument: "P48", years: "2009–2016", sky: "dec > −30", depth: "R 21", exposures: "~3M chips", volume: "~100 TB", tb: 100, where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/ptf.html", step: 0 },
  { name: "SDSS imaging", instrument: "APO 2.5 m", years: "1998–2009", sky: "14,500 sq deg, Stripe 82 repeats", depth: "r 22.5", exposures: "~1M fields", volume: "16 TB", tb: 16, where: "SAS", url: "https://www.sdss.org/dr18/imaging/", step: 0 },
  { name: "Hyper Suprime-Cam", instrument: "Subaru 8 m", years: "2014–", sky: "pointed, 1.8 sq deg", depth: "r 26", exposures: "~5M CCDs", volume: "~300 TB", tb: 300, where: "SMOKA, hscla", url: "https://smoka.nao.ac.jp/", step: 0, note: "18-month proprietary period; SSP PDR3 warps and coadds at hscla" },
  { name: "Suprime-Cam", instrument: "Subaru 8 m", years: "1999–2017", sky: "pointed, 0.25 sq deg", depth: "r 25.5", exposures: "1.2M", volume: "19 TB", tb: 19, where: "SMOKA", url: "https://smoka.nao.ac.jp/", step: 0 },
  { name: "MegaCam, CFH12K, WIRCam", instrument: "CFHT 3.6 m", years: "1999–", sky: "north, 1 sq deg", depth: "r 24.5", exposures: "~0.5M", volume: "~350 TB", tb: 350, where: "CADC · CANFAR", url: "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/cfht/", step: 0, note: "CFHTLS, CFIS, OSSOS, NGVS" },
  { name: "OmegaCAM", instrument: "VST 2.6 m", years: "2011–", sky: "south, 1 sq deg", depth: "r 24", exposures: "~0.5M", volume: "~400 TB", tb: 400, where: "ESO archive", url: "https://archive.eso.org/", step: 0, note: "KiDS, ATLAS, VPHAS+" },
  { name: "VIRCAM", instrument: "VISTA 4 m", years: "2009–", sky: "south, NIR", depth: "J 21", exposures: "~0.5M", volume: "~600 TB", tb: 600, where: "ESO archive", url: "https://archive.eso.org/", step: 0, note: "VHS, VVV, VIKING" },
  { name: "WFI", instrument: "MPG/ESO 2.2 m", years: "1999–", sky: "south, 0.3 sq deg", depth: "R 23", exposures: "~0.2M", volume: "~30 TB", tb: 30, where: "ESO archive", url: "https://archive.eso.org/", step: 0 },
  { name: "SkyMapper", instrument: "1.35 m", years: "2014–2021", sky: "south, 5.7 sq deg", depth: "r 21.5", exposures: "~0.6M", volume: "~100 TB", tb: 100, where: "SkyMapper DR4", url: "https://skymapper.anu.edu.au/", step: 0 },
  { name: "Catalina Sky Survey", instrument: "703, G96, I52", years: "2003–", sky: "ecliptic to ±60", depth: "V 19.5 to 21.5", exposures: "~3M", volume: "~100 TB", tb: 100, where: "PDS SBN", url: "https://sbnarchive.psi.edu/pds4/surveys/gbo.ast.catalina.survey/", step: 0, note: "NEO survey frames from 2003, delivered nightly" },
  { name: "Spacewatch", instrument: "0.9 m mosaic", years: "2003–2016", sky: "ecliptic", depth: "V 21.7", exposures: "~0.3M", volume: "~20 TB", tb: 20, where: "PDS SBN", url: "https://sbnarchive.psi.edu/pds4/surveys/gbo.ast.spacewatch.survey/", step: 0 },
  { name: "LONEOS", instrument: "0.6 m Schmidt", years: "1998–2011", sky: "north", depth: "V 19.3", exposures: "~0.2M", volume: "~10 TB", tb: 10, where: "PDS SBN", url: "https://sbnarchive.psi.edu/pds4/surveys/gbo.ast.loneos.survey/", step: 0 },
  { name: "NEAT", instrument: "Palomar 1.2 m, Maui GEODSS", years: "1995–2007", sky: "ecliptic", depth: "R 20.5", exposures: "~0.2M", volume: "~10 TB", tb: 10, where: "PDS SBN", url: "https://sbnarchive.psi.edu/pds4/surveys/", step: 0 },
  { name: "INT WFC", instrument: "INT 2.5 m", years: "1998–", sky: "north, 0.3 sq deg", depth: "r 22", exposures: "~0.3M", volume: "~20 TB", tb: 20, where: "ING, CASU", url: "https://casu.ast.cam.ac.uk/casuadc/ingarch/query", step: 0, note: "IPHAS, UVEX" },
  { name: "Las Cumbres", instrument: "0.4 m, 1 m, 2 m network", years: "2014–", sky: "pointed, both hemispheres", depth: "r 21", exposures: "~10M", volume: "~100 TB", tb: 100, where: "LCO archive", url: "https://archive.lco.global/", step: 0, note: "one-year proprietary period; follow-up, not survey" },
  { name: "GMOS", instrument: "Gemini 8 m", years: "2001–", sky: "pointed", depth: "r 25", exposures: "~0.3M", volume: "~100 TB", tb: 100, where: "Gemini archive", url: "https://archive.gemini.edu/", step: 0 },
  { name: "Keck imagers", instrument: "Keck 10 m", years: "1994–", sky: "pointed", depth: "r 26", exposures: "~0.5M", volume: "~50 TB", tb: 50, where: "KOA", url: "https://koa.ipac.caltech.edu/", step: 0 },
  { name: "WFCAM", instrument: "UKIRT 3.8 m", years: "2005–", sky: "north, NIR", depth: "K 18.3", exposures: "~1M", volume: "~100 TB", tb: 100, where: "WSA", url: "http://wsa.roe.ac.uk/", step: 0, note: "UKIDSS, UHS" },
  { name: "2MASS", instrument: "two 1.3 m", years: "1997–2001", sky: "all sky, NIR", depth: "Ks 14.3", exposures: "~60k scans", volume: "10 TB", tb: 10, where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/2mass.html", step: 0 },
  { name: "DSS plates", instrument: "Palomar and UK Schmidt", years: "1949–2002", sky: "all sky", depth: "R 21", exposures: "~7k plates", volume: "3 TB", tb: 3, where: "STScI, SuperCOSMOS", url: "https://archive.stsci.edu/cgi-bin/dss_form", step: 0, note: "POSS-I, POSS-II, SERC, AAO-R" },
  { name: "DASCH plates", instrument: "Harvard collection", years: "1885–1992", sky: "all sky", depth: "B 15", exposures: "429k plates", volume: "~320 TB", tb: 320, where: "DASCH DR7", url: "https://dasch.cfa.harvard.edu/", step: 0, note: "scanning finished 2024" },
  { name: "WISE", instrument: "0.4 m, 4 bands", years: "2010–2011", sky: "all sky, IR", depth: "W1 17", exposures: "~1.6M sets", volume: "~30 TB", tb: 30, where: "IRSA · AWS Open Data", url: "https://irsa.ipac.caltech.edu/Missions/wise.html", step: 0 },
  { name: "NEOWISE", instrument: "0.4 m, 2 bands", years: "2013–2024", sky: "all sky, IR", depth: "W1 16.5", exposures: "26.9M sets", volume: "~330 TB", tb: 330, where: "IRSA · AWS Open Data", url: "https://irsa.ipac.caltech.edu/data/WISE/docs/release/NEOWISE/expsup/", step: 0, note: "final release 2024; every sky position every six months" },
  { name: "SPHEREx", instrument: "0.2 m, 102 bands", years: "2025–", sky: "all sky, IR", depth: "19 AB", exposures: "~1M", volume: "~100 TB", tb: 100, where: "IRSA · AWS Open Data", url: "https://irsa.ipac.caltech.edu/Missions/spherex.html", step: 0, note: "quick releases weekly" },
  { name: "TESS FFIs", instrument: "4 × 10 cm", years: "2018–", sky: "all sky, 24 × 96 deg", depth: "T 17 in 30 min", exposures: "~2M", volume: "~100 TB", tb: 100, where: "MAST · AWS Open Data", url: "https://archive.stsci.edu/missions-and-data/tess", step: 0, note: "the only all-sky dataset with nightly cadence for months; used for Planet Nine" },
  { name: "Kepler, K2 FFIs", instrument: "0.95 m", years: "2009–2018", sky: "fields near ecliptic (K2)", depth: "Kp 20", exposures: "~1k", volume: "~1 TB", tb: 1, where: "MAST", url: "https://archive.stsci.edu/missions-and-data/kepler", step: 0 },
  { name: "GALEX", instrument: "0.5 m UV", years: "2003–2013", sky: "most sky, UV", depth: "NUV 20.5", exposures: "~100k", volume: "30 TB", tb: 30, where: "MAST · AWS Open Data", url: "https://archive.stsci.edu/missions-and-data/galex", step: 0 },
  { name: "Spitzer IRAC", instrument: "0.85 m IR", years: "2003–2020", sky: "pointed", depth: "3.6 µm 21 AB", exposures: "~10M BCDs", volume: "~100 TB", tb: 100, where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/spitzer.html", step: 0 },
  { name: "Hubble imagers", instrument: "2.4 m", years: "1990–", sky: "pointed", depth: "V 27", exposures: "~1M", volume: "~500 TB", tb: 500, where: "MAST · AWS Open Data", url: "https://archive.stsci.edu/missions-and-data/hst", step: 0 },
  { name: "JWST imagers", instrument: "6.5 m IR", years: "2022–", sky: "pointed", depth: "29 AB", exposures: "~0.2M", volume: "~300 TB", tb: 300, where: "MAST · AWS Open Data", url: "https://archive.stsci.edu/missions-and-data/jwst", step: 0 },
  { name: "Euclid", instrument: "1.2 m, VIS + NISP", years: "2024–", sky: "wide survey, 63 sq deg in Q1", depth: "VIS 25.5", exposures: "~50k", volume: "~1 PB at DR1", tb: 1000, where: "ESA SAS, IRSA", url: "https://eas.esac.esa.int/sas/", step: 0, note: "DR1 due 2026-10-21" },
  { name: "Rubin data previews", instrument: "Simonyi 8.4 m, ComCam and LSSTCam", years: "2024–2026", sky: "a few fields", depth: "r 24", exposures: "~5k", volume: "~30 TB", tb: 30, where: "RSP", url: "https://data.lsst.cloud/", step: 0, note: "DP1 and EDP2 are for data-rights holders; the survey images stay proprietary for two years, the alerts are public" },
  { name: "APPLAUSE plates", instrument: "German plate archives", years: "1893–1998", sky: "mostly north", depth: "B 15", exposures: "~100k plates", volume: "~50 TB", tb: 50, where: "plate-archive.org", url: "https://www.plate-archive.org/", step: 0 },
  { name: "WIYN ODI", instrument: "WIYN 3.5 m", years: "2012–", sky: "pointed, 0.3 sq deg", depth: "r 24", exposures: "~0.1M", volume: "~30 TB", tb: 30, where: "ODI-PPA", url: "https://portal.odi.iu.edu/", step: 0 },
  { name: "NGTS", instrument: "12 × 20 cm", years: "2015–", sky: "south, 96 sq deg", depth: "I 16", exposures: "~10M", volume: "~100 TB", tb: 100, where: "ESO archive", url: "https://archive.eso.org/", step: 0 },
  { name: "FORS2, HAWK-I", instrument: "VLT 8 m", years: "1999–", sky: "pointed", depth: "R 26", exposures: "~0.5M", volume: "~50 TB", tb: 50, where: "ESO archive", url: "https://archive.eso.org/", step: 0 },
];

// Pixels nobody outside the project can get.
export const PRIVATE = [
  "Rubin LSST survey images (two-year proprietary period, data-rights holders only)",
  "Pan-STARRS after 2014 and all of PS2",
  "ATLAS (forced photometry only)",
  "ASAS-SN (light curves only)",
  "LINEAR, KMTNet, OGLE, WFST, BlackGEM, GOTO",
];

export const RECORDS: Record[] = [
  { name: "MPC observations", holds: "all published astrometry", size: "521M obs", years: "1801–", where: "MPC, PDS SBN mirror", url: "https://minorplanetcenter.net/iau/ECS/MPCAT-OBS/MPCAT-OBS.html", step: 0 },
  { name: "MPC orbits", holds: "all designated orbits", size: "1.57M orbits", years: "", where: "MPCORB", url: "https://minorplanetcenter.net/iau/MPCORB.html", step: 0 },
  { name: "MPC isolated tracklets", holds: "unlinked observations", size: "9.5M obs", years: "", where: "ITF", url: "https://minorplanetcenter.net/iau/ITF/ITF.txt.gz", step: 0 },
  { name: "NEOCP, PCCP", holds: "unconfirmed candidates", size: "~100 at a time", years: "", where: "MPC", url: "https://minorplanetcenter.net/iau/NEO/toconfirm_tabular.html", step: 0 },
  { name: "JPL SBDB, Horizons", holds: "orbits, covariances, ephemerides", size: "1.5M bodies", years: "", where: "ssd-api.jpl.nasa.gov", url: "https://ssd.jpl.nasa.gov/tools/sbdb_query.html", step: 0 },
  { name: "Lowell astorb", holds: "orbits with uncertainties", size: "1.5M", years: "", where: "asteroid.lowell.edu", url: "https://asteroid.lowell.edu/astorb/", step: 0 },
  { name: "ESA NEOCC", holds: "NEO orbits and risk list", size: "~40k NEOs", years: "", where: "neo.ssa.esa.int", url: "https://neo.ssa.esa.int/", step: 0 },
  { name: "Gaia DR3 solar system", holds: "epoch astrometry of asteroids", size: "158k objects, 23M transits", years: "2014–2017", where: "Gaia archive", url: "https://gea.esac.esa.int/archive/", step: 0 },
  { name: "Gaia FPR solar system", holds: "66 months of asteroid astrometry", size: "156k objects", years: "2014–2020", where: "Gaia archive", url: "https://www.cosmos.esa.int/web/gaia/fpr", step: 0 },
  { name: "Radar astrometry", holds: "delay and Doppler", size: "~1k objects", years: "1968–", where: "JPL", url: "https://ssd.jpl.nasa.gov/sb/radar.html", step: 0 },
  { name: "Stellar occultations", holds: "chords and sizes", size: "~1k events", years: "", where: "IOTA, Lucky Star", url: "https://occultations.org/", step: 0 },
  { name: "Lightcurves", holds: "rotation periods, photometry", size: "~40k objects", years: "", where: "LCDB, ALCDEF", url: "https://www.minorplanet.info/php/lcdb.php", step: 0 },
  { name: "SsODNet", holds: "one card per object", size: "1.5M", years: "", where: "IMCCE", url: "https://ssp.imcce.fr/webservices/ssodnet/", step: 0 },
  { name: "ZTF alerts", holds: "difference detections, matched", size: "~2B alerts", years: "2018–", where: "ALeRCE, Fink, Lasair, UW archive", url: "https://ztf.uw.edu/alerts/public/", step: 0 },
  { name: "Rubin alerts", holds: "difference detections, matched", size: "~10M per night", years: "2026–", where: "brokers", url: "https://rubinobservatory.org/for-scientists/data-products/alerts-and-brokers", step: 0 },
  { name: "NEOWISE known-object associations", holds: "IR detections of known bodies", size: "~50M", years: "2010–2024", where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/wise.html", step: 0 },
  { name: "SDSS Moving Object Catalog", holds: "asteroid detections, colors", size: "472k", years: "1998–2007", where: "SDSS, PDS", url: "https://faculty.washington.edu/ivezic/sdssmoc/sdssmoc.html", step: 0 },
  { name: "OSSOS", holds: "TNOs with efficiencies", size: "840", years: "2013–2017", where: "ossos-survey.org", url: "https://www.ossos-survey.org/", step: 0 },
  { name: "CFEPS", holds: "TNOs with efficiencies", size: "169", years: "2003–2007", where: "CADC", url: "https://ui.adsabs.harvard.edu/search/q=title%3A%22Canada-France%20Ecliptic%20Plane%20Survey%22", step: 0 },
  { name: "DES TNOs", holds: "TNOs with efficiencies", size: "817", years: "2013–2019", where: "DES", url: "https://des.ncsa.illinois.edu/releases", step: 0 },
  { name: "DEEP", holds: "TNOs with efficiencies", size: "~2k", years: "2019–2023", where: "DECam", url: "https://ui.adsabs.harvard.edu/search/q=title%3A%22DECam%20Ecliptic%20Exploration%20Project%22", step: 0 },
  { name: "MOVIS", holds: "VISTA movers", size: "39k", years: "2010–2013", where: "VizieR", url: "https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/591/A115", step: 0 },
  { name: "Euclid Q1 solar system", holds: "Euclid movers", size: "~2k", years: "2024–", where: "ESA SAS", url: "https://eas.esac.esa.int/sas/", step: 0 },
  { name: "Comets", holds: "orbits, activity, photometry", size: "~5k", years: "", where: "MPC, JPL, COBS", url: "https://cobs.si/", step: 0 },
  { name: "Natural satellite astrometry", holds: "moon observations", size: "~1M obs", years: "1600–", where: "NSDB, IMCCE", url: "https://nsdb.imcce.fr/", step: 0 },
  { name: "Planets and natural satellites", holds: "moon ephemerides", size: "~300", years: "", where: "JPL Horizons", url: "https://ssd.jpl.nasa.gov/horizons/", step: 0 },
  { name: "Artificial satellites", holds: "tracked-object elements", size: "~30k", years: "1957–", where: "Space-Track, CelesTrak", url: "https://www.space-track.org/", step: 0 },
  { name: "NOIRLab Source Catalog DR2", holds: "5 sigma detections, every exposure", size: "68B measurements", years: "2004–2019", where: "Data Lab", url: "https://datalab.noirlab.edu/nscdr2/index.php", step: 0 },
  { name: "Pan-STARRS DR2 detections", holds: "single-epoch detections", size: "~10B", years: "2010–2014", where: "MAST", url: "https://outerspace.stsci.edu/display/PANSTARRS/PS1+Database+object+and+detection+tables", step: 0 },
  { name: "ZTF DR24 sources", holds: "single-epoch detections", size: "1,043B", years: "2018–2025", where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/ztf.html", step: 0 },
  { name: "unTimely", holds: "unWISE epoch detections", size: "23B", years: "2010–2020", where: "NERSC", url: "https://portal.nersc.gov/project/cosmo/data/unwise/neo7/untimely-catalog/", step: 0 },
  { name: "NEOWISE single-exposure sources", holds: "L1b detections", size: "~150B", years: "2013–2024", where: "IRSA", url: "https://irsa.ipac.caltech.edu/data/WISE/docs/release/NEOWISE/expsup/", step: 0 },
  { name: "SkyMapper DR4 photometry", holds: "per-image detections", size: "~10B", years: "2014–2021", where: "SkyMapper", url: "https://skymapper.anu.edu.au/", step: 0 },
  { name: "Rubin DP1 sources", holds: "difference and forced detections", size: "~100M", years: "2024–2025", where: "RSP", url: "https://dp1.lsst.io/", step: 0 },
  { name: "Euclid Q1 MER", holds: "sources in Q1 fields", size: "~30M", years: "2024–", where: "ESA SAS", url: "https://eas.esac.esa.int/sas/", step: 0 },
  { name: "Gaia DR4 epochs", holds: "epoch astrometry and photometry", size: "1.8B sources", years: "2014–2020", where: "Gaia archive", url: "https://www.cosmos.esa.int/web/gaia/dr4", step: 0, note: "due 2026-12" },
  { name: "Gaia DR3 sources", holds: "every bright star", size: "1.8B", years: "2014–2017", where: "Gaia archive", url: "https://gea.esac.esa.int/archive/", step: 0 },
  { name: "Legacy Surveys DR10", holds: "every fixed source, deep", size: "~2.5B", years: "", where: "legacysurvey.org", url: "https://www.legacysurvey.org/dr10/", step: 0 },
  { name: "Pan-STARRS DR2 objects", holds: "every fixed source, 3π", size: "3B", years: "", where: "MAST", url: "https://outerspace.stsci.edu/display/PANSTARRS/", step: 0 },
  { name: "DES DR2", holds: "every fixed source, south", size: "691M", years: "", where: "NCSA", url: "https://des.ncsa.illinois.edu/releases/dr2", step: 0 },
  { name: "HSC-SSP PDR3", holds: "every fixed source, deepest", size: "~600M", years: "", where: "hscla", url: "https://hsc-release.mtk.nao.ac.jp/doc/", step: 0 },
  { name: "SkyMapper DR4 objects", holds: "every fixed source, south", size: "~700M", years: "", where: "SkyMapper", url: "https://skymapper.anu.edu.au/", step: 0 },
  { name: "CatWISE2020", holds: "every fixed IR source", size: "1.9B", years: "", where: "IRSA", url: "https://catwise.github.io/", step: 0 },
  { name: "unWISE", holds: "every fixed IR source", size: "2B", years: "", where: "NERSC", url: "https://catalog.unwise.me/", step: 0 },
  { name: "2MASS PSC", holds: "every fixed NIR source", size: "471M", years: "", where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/2mass.html", step: 0 },
  { name: "VHS", holds: "every fixed NIR source, south", size: "~1B", years: "", where: "ESO", url: "https://archive.eso.org/", step: 0 },
  { name: "UKIDSS, UHS", holds: "every fixed NIR source, north", size: "~1B", years: "", where: "WSA", url: "http://wsa.roe.ac.uk/", step: 0 },
  { name: "Gaia DR3 variables", holds: "changes but does not move", size: "10M", years: "", where: "Gaia archive", url: "https://gea.esac.esa.int/archive/", step: 0 },
  { name: "ASAS-SN variables", holds: "changes but does not move", size: "~700k", years: "", where: "asas-sn.osu.edu", url: "https://asas-sn.osu.edu/variables", step: 0 },
  { name: "Transient Name Server", holds: "reported transients", size: "~200k", years: "", where: "wis-tns.org", url: "https://www.wis-tns.org/", step: 0 },
  { name: "CADC SSOIS", holds: "orbit to image lookup", size: "", years: "", where: "CADC", url: "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/ssois/", step: 0 },
  { name: "SkyBoT", holds: "known objects in a cone", size: "", years: "", where: "IMCCE", url: "https://ssp.imcce.fr/webservices/skybot/", step: 0 },
  { name: "JPL SBIdent", holds: "known objects in a field", size: "", years: "", where: "JPL", url: "https://ssd-api.jpl.nasa.gov/doc/sb_ident.html", step: 0 },
  { name: "ADAM precovery", holds: "orbit matches in NSC and ZTF", size: "", years: "", where: "B612", url: "https://adam.b612.ai/", step: 0 },
];
