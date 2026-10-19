export type Step = 0 | 1 | 2 | 3 | 4;

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
  step: Step;
  note?: string;
};

export type Record = {
  name: string;
  holds: string;
  size: string;
  years: string;
  where: string;
  step: Step;
  note?: string;
};

export const PIXEL_STEPS = ["listed", "fetched", "indexed", "passed"];
export const RECORD_STEPS = ["listed", "fetched", "mirrored", "merged"];

// Depth is a single exposure at 5 sigma; counts and volumes are rounded from archive pages (2026-10).
export const DATASETS: Dataset[] = [
  { name: "DECam", instrument: "Blanco 4 m", years: "2012–", sky: "south + equator, 1 sq deg", depth: "r 23.5", exposures: "7.1M", volume: "1.7 PB", tb: 1700, where: "NOIRLab · Data Lab", step: 2, note: "DES, DECaLS, DELVE, DEEP and every PI program; 93% of the NOIRLab archive" },
  { name: "Mosaic, Mosaic3, 90Prime", instrument: "Mayall 4 m, Bok 2.3 m", years: "2004–2019", sky: "north", depth: "r 23", exposures: "~0.5M", volume: "~50 TB", tb: 50, where: "NOIRLab · Data Lab", step: 1, note: "BASS and MzLS single epochs" },
  { name: "NEWFIRM", instrument: "Mayall, Blanco", years: "2007–", sky: "pointed, NIR", depth: "Ks 20", exposures: "~1M", volume: "~20 TB", tb: 20, where: "NOIRLab · Data Lab", step: 1 },
  { name: "Pan-STARRS1 warps", instrument: "PS1 1.8 m", years: "2010–2014", sky: "dec > −30, 3π", depth: "r 22", exposures: "24M", volume: "1.6 PB", tb: 1600, where: "MAST", step: 1, note: "DR2 single-epoch warps; later PS1 and all PS2 data are not released" },
  { name: "ZTF", instrument: "P48", years: "2018–", sky: "dec > −31, 47 sq deg", depth: "r 20.5", exposures: "70M quadrants", volume: "~1 PB", tb: 1000, where: "IRSA · Fornax", step: 2, note: "DR24; alerts carry difference cutouts from day one" },
  { name: "PTF, iPTF", instrument: "P48", years: "2009–2016", sky: "dec > −30", depth: "R 21", exposures: "~3M chips", volume: "~100 TB", tb: 100, where: "IRSA", step: 1 },
  { name: "SDSS imaging", instrument: "APO 2.5 m", years: "1998–2009", sky: "14,500 sq deg, Stripe 82 repeats", depth: "r 22.5", exposures: "~1M fields", volume: "16 TB", tb: 16, where: "SAS", step: 1 },
  { name: "Hyper Suprime-Cam", instrument: "Subaru 8 m", years: "2014–", sky: "pointed, 1.8 sq deg", depth: "r 26", exposures: "~5M CCDs", volume: "~300 TB", tb: 300, where: "SMOKA, hscla", step: 2, note: "18-month proprietary period; SSP PDR3 warps and coadds at hscla" },
  { name: "Suprime-Cam", instrument: "Subaru 8 m", years: "1999–2017", sky: "pointed, 0.25 sq deg", depth: "r 25.5", exposures: "1.2M", volume: "19 TB", tb: 19, where: "SMOKA", step: 1 },
  { name: "MegaCam, CFH12K, WIRCam", instrument: "CFHT 3.6 m", years: "1999–", sky: "north, 1 sq deg", depth: "r 24.5", exposures: "~0.5M", volume: "~350 TB", tb: 350, where: "CADC · CANFAR", step: 1, note: "CFHTLS, CFIS, OSSOS, NGVS" },
  { name: "OmegaCAM", instrument: "VST 2.6 m", years: "2011–", sky: "south, 1 sq deg", depth: "r 24", exposures: "~0.5M", volume: "~400 TB", tb: 400, where: "ESO archive", step: 1, note: "KiDS, ATLAS, VPHAS+" },
  { name: "VIRCAM", instrument: "VISTA 4 m", years: "2009–", sky: "south, NIR", depth: "J 21", exposures: "~0.5M", volume: "~600 TB", tb: 600, where: "ESO archive", step: 1, note: "VHS, VVV, VIKING" },
  { name: "WFI", instrument: "MPG/ESO 2.2 m", years: "1999–", sky: "south, 0.3 sq deg", depth: "R 23", exposures: "~0.2M", volume: "~30 TB", tb: 30, where: "ESO archive", step: 1 },
  { name: "SkyMapper", instrument: "1.35 m", years: "2014–2021", sky: "south, 5.7 sq deg", depth: "r 21.5", exposures: "~0.6M", volume: "~100 TB", tb: 100, where: "SkyMapper DR4", step: 1 },
  { name: "Catalina Sky Survey", instrument: "703, G96, I52", years: "2003–", sky: "ecliptic to ±60", depth: "V 19.5 to 21.5", exposures: "~3M", volume: "~100 TB", tb: 100, where: "PDS SBN", step: 1, note: "NEO survey frames from 2003, delivered nightly" },
  { name: "Spacewatch", instrument: "0.9 m mosaic", years: "2003–2016", sky: "ecliptic", depth: "V 21.7", exposures: "~0.3M", volume: "~20 TB", tb: 20, where: "PDS SBN", step: 1 },
  { name: "LONEOS", instrument: "0.6 m Schmidt", years: "1998–2011", sky: "north", depth: "V 19.3", exposures: "~0.2M", volume: "~10 TB", tb: 10, where: "PDS SBN", step: 1 },
  { name: "NEAT", instrument: "Palomar 1.2 m, Maui GEODSS", years: "1995–2007", sky: "ecliptic", depth: "R 20.5", exposures: "~0.2M", volume: "~10 TB", tb: 10, where: "PDS SBN", step: 1 },
  { name: "INT WFC", instrument: "INT 2.5 m", years: "1998–", sky: "north, 0.3 sq deg", depth: "r 22", exposures: "~0.3M", volume: "~20 TB", tb: 20, where: "ING, CASU", step: 1, note: "IPHAS, UVEX" },
  { name: "Las Cumbres", instrument: "0.4 m, 1 m, 2 m network", years: "2014–", sky: "pointed, both hemispheres", depth: "r 21", exposures: "~10M", volume: "~100 TB", tb: 100, where: "LCO archive", step: 1, note: "one-year proprietary period; follow-up, not survey" },
  { name: "GMOS", instrument: "Gemini 8 m", years: "2001–", sky: "pointed", depth: "r 25", exposures: "~0.3M", volume: "~100 TB", tb: 100, where: "Gemini archive", step: 1 },
  { name: "Keck imagers", instrument: "Keck 10 m", years: "1994–", sky: "pointed", depth: "r 26", exposures: "~0.5M", volume: "~50 TB", tb: 50, where: "KOA", step: 1 },
  { name: "WFCAM", instrument: "UKIRT 3.8 m", years: "2005–", sky: "north, NIR", depth: "K 18.3", exposures: "~1M", volume: "~100 TB", tb: 100, where: "WSA", step: 1, note: "UKIDSS, UHS" },
  { name: "2MASS", instrument: "two 1.3 m", years: "1997–2001", sky: "all sky, NIR", depth: "Ks 14.3", exposures: "~60k scans", volume: "10 TB", tb: 10, where: "IRSA", step: 1 },
  { name: "DSS plates", instrument: "Palomar and UK Schmidt", years: "1949–2002", sky: "all sky", depth: "R 21", exposures: "~7k plates", volume: "3 TB", tb: 3, where: "STScI, SuperCOSMOS", step: 1, note: "POSS-I, POSS-II, SERC, AAO-R" },
  { name: "DASCH plates", instrument: "Harvard collection", years: "1885–1992", sky: "all sky", depth: "B 15", exposures: "429k plates", volume: "~320 TB", tb: 320, where: "DASCH DR7", step: 1, note: "scanning finished 2024" },
  { name: "WISE", instrument: "0.4 m, 4 bands", years: "2010–2011", sky: "all sky, IR", depth: "W1 17", exposures: "~1.6M sets", volume: "~30 TB", tb: 30, where: "IRSA · AWS Open Data", step: 1 },
  { name: "NEOWISE", instrument: "0.4 m, 2 bands", years: "2013–2024", sky: "all sky, IR", depth: "W1 16.5", exposures: "26.9M sets", volume: "~330 TB", tb: 330, where: "IRSA · AWS Open Data", step: 1, note: "final release 2024; every sky position every six months" },
  { name: "SPHEREx", instrument: "0.2 m, 102 bands", years: "2025–", sky: "all sky, IR", depth: "19 AB", exposures: "~1M", volume: "~100 TB", tb: 100, where: "IRSA · AWS Open Data", step: 1, note: "quick releases weekly" },
  { name: "TESS FFIs", instrument: "4 × 10 cm", years: "2018–", sky: "all sky, 24 × 96 deg", depth: "T 17 in 30 min", exposures: "~2M", volume: "~100 TB", tb: 100, where: "MAST · AWS Open Data", step: 1, note: "the only all-sky dataset with nightly cadence for months; used for Planet Nine" },
  { name: "Kepler, K2 FFIs", instrument: "0.95 m", years: "2009–2018", sky: "fields near ecliptic (K2)", depth: "Kp 20", exposures: "~1k", volume: "~1 TB", tb: 1, where: "MAST", step: 1 },
  { name: "GALEX", instrument: "0.5 m UV", years: "2003–2013", sky: "most sky, UV", depth: "NUV 20.5", exposures: "~100k", volume: "30 TB", tb: 30, where: "MAST · AWS Open Data", step: 1 },
  { name: "Spitzer IRAC", instrument: "0.85 m IR", years: "2003–2020", sky: "pointed", depth: "3.6 µm 21 AB", exposures: "~10M BCDs", volume: "~100 TB", tb: 100, where: "IRSA", step: 1 },
  { name: "Hubble imagers", instrument: "2.4 m", years: "1990–", sky: "pointed", depth: "V 27", exposures: "~1M", volume: "~500 TB", tb: 500, where: "MAST · AWS Open Data", step: 1 },
  { name: "JWST imagers", instrument: "6.5 m IR", years: "2022–", sky: "pointed", depth: "29 AB", exposures: "~0.2M", volume: "~300 TB", tb: 300, where: "MAST · AWS Open Data", step: 1 },
  { name: "Euclid", instrument: "1.2 m, VIS + NISP", years: "2024–", sky: "wide survey, 63 sq deg in Q1", depth: "VIS 25.5", exposures: "~50k", volume: "~1 PB at DR1", tb: 1000, where: "ESA SAS, IRSA", step: 1, note: "DR1 due 2026-10-21" },
  { name: "Rubin data previews", instrument: "Simonyi 8.4 m, ComCam and LSSTCam", years: "2024–2026", sky: "a few fields", depth: "r 24", exposures: "~5k", volume: "~30 TB", tb: 30, where: "RSP", step: 1, note: "DP1 and EDP2 are for data-rights holders; the survey images stay proprietary for two years, the alerts are public" },
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
  { name: "MPC observations", holds: "every published astrometric observation of every designated object", size: "521M obs", years: "1801–", where: "MPC, PDS SBN mirror", step: 2, note: "80-column and ADES; parsed daily for the pinned orbits" },
  { name: "MPC orbits", holds: "orbits for every designated minor planet and comet", size: "1.57M orbits", years: "", where: "MPCORB, MPC Explorer", step: 2 },
  { name: "MPC isolated tracklets", holds: "observations nobody has linked", size: "9.5M obs", years: "", where: "ITF", step: 1, note: "already the unknown side of the difference, for surveys that report" },
  { name: "NEOCP, PCCP", holds: "candidates awaiting confirmation", size: "~100 at a time", years: "", where: "MPC", step: 1 },
  { name: "JPL SBDB, Horizons", holds: "orbits with covariances; ephemerides for anything", size: "1.5M bodies", years: "", where: "ssd-api.jpl.nasa.gov", step: 2, note: "used for every ephemeris check" },
  { name: "Lowell astorb", holds: "orbits with ephemeris uncertainties", size: "1.5M", years: "", where: "asteroid.lowell.edu", step: 1 },
  { name: "Gaia DR3 solar system", holds: "epoch astrometry of known asteroids", size: "158k objects, 23M transits", years: "2014–2017", where: "Gaia archive", step: 1, note: "DR4 adds 2017–2020, due 2026-12" },
  { name: "ZTF alerts", holds: "every 5 sigma difference detection with known-object match", size: "~2B alerts", years: "2018–", where: "ALeRCE, Fink, Lasair, UW archive", step: 2 },
  { name: "Rubin alerts", holds: "every difference detection with ssObject match", size: "~10M per night", years: "2026–", where: "brokers", step: 1, note: "public from day one, unlike the images" },
  { name: "NEOWISE known-object associations", holds: "every detection matched to a known body", size: "~50M", years: "2010–2024", where: "IRSA", step: 1 },
  { name: "SDSS Moving Object Catalog", holds: "asteroid detections with colors", size: "472k", years: "1998–2007", where: "SDSS, PDS", step: 1 },
  { name: "Characterized TNO surveys", holds: "detections with published efficiency functions", size: "~4k TNOs", years: "2003–", where: "OSSOS, CFEPS, DES, DEEP, DEEP ecliptic", step: 1, note: "the only records that say what was not seen" },
  { name: "NIR and Euclid SSO catalogs", holds: "moving sources from VISTA, KiDS and Euclid", size: "~100k", years: "2010–", where: "MOVIS, KiDS, Euclid Q1", step: 1 },
  { name: "Comets", holds: "orbits, activity, photometry", size: "~5k", years: "", where: "MPC, JPL, COBS", step: 1 },
  { name: "Planets and natural satellites", holds: "ephemerides of every known moon", size: "~300", years: "", where: "JPL Horizons", step: 1 },
  { name: "Artificial satellites", holds: "orbital elements of every tracked object", size: "~30k", years: "1957–", where: "Space-Track, CelesTrak", step: 1, note: "streak subtraction" },
  { name: "Gaia DR3 sources", holds: "position, motion, parallax of every bright star", size: "1.8B", years: "2014–2017", where: "Gaia archive", step: 2, note: "the static sky and the astrometric frame" },
  { name: "Deep static catalogs", holds: "every fixed source to the coadd depth", size: "~10B", years: "", where: "Legacy Surveys DR10, PS1 DR2, DES DR2, HSC PDR3, SkyMapper DR4", step: 1 },
  { name: "Infrared static catalogs", holds: "every fixed IR source", size: "~2.5B", years: "", where: "CatWISE2020, unWISE, 2MASS, VHS, UKIDSS", step: 1 },
  { name: "Variable and transient catalogs", holds: "sources that change but do not move", size: "~100M", years: "", where: "Gaia variables, ZTF, ASAS-SN, TNS", step: 1 },
  { name: "Coverage services", holds: "which image holds which orbit", size: "", years: "", where: "CADC SSOIS, SkyBoT, JPL SBIdent", step: 2, note: "exist today for single orbits; the archive pass replaces them" },
];
