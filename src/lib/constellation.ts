export type Step = 0 | 1 | 2; // empty, half, full

export type DatasetRole = "Core optical" | "Deep optical" | "Recovery" | "Infrared" | "Targeted" | "Specialized" | "Access-dependent" | "Evaluate";

export type Dataset = {
  role: DatasetRole;
  name: string;
  instrument: string;
  years: string;
  sky: string;
  depth: string;
  exposures: string;
  volume: string;
  tb: number | null; // null means the archive volume has not been verified
  where: string;
  url: string;
  step: Step;
  note?: string;
  reference?: string;
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
  reference?: string;
};


// Planning inventory reviewed 2026-10-20, not a live archive manifest.
// Depths mix bands and product types; never use these labels as exposure-level limits.
// Retained counts/volumes are rough planning estimates, not uniformly verified totals.
// Role describes proposed use, while step tracks ingestion progress.
export const DATASETS: Dataset[] = [
  { name: "DECam", role: "Core optical", instrument: "Blanco 4 m", years: "2012–", sky: "south + equator", depth: "r 23.5", exposures: "7.1M", volume: "1.7 PB", tb: 1700, where: "NOIRLab · Data Lab", url: "https://astroarchive.noirlab.edu/", step: 0, note: "Public DECam programs include DES, DECaLS, DELVE and DEEP; deduplicate shared exposures." },
  { name: "Mosaic, Mosaic II, Mosaic3, 90Prime", role: "Core optical", instrument: "Mayall 4 m, Blanco 4 m, Bok 2.3 m", years: "2003–2019", sky: "both hemispheres", depth: "r 23", exposures: "~0.5M", volume: "~50 TB", tb: 50, where: "NOIRLab · Data Lab", url: "https://astroarchive.noirlab.edu/", step: 0, note: "BASS and MzLS single epochs" },
  { name: "NEWFIRM", role: "Infrared", instrument: "Mayall, Blanco", years: "2007–", sky: "pointed, NIR", depth: "Ks 20", exposures: "~1M", volume: "~20 TB", tb: 20, where: "NOIRLab · Data Lab", url: "https://astroarchive.noirlab.edu/", step: 0 },
  { name: "Pan-STARRS1 warps", role: "Core optical", instrument: "PS1 1.8 m", years: "2010–2014", sky: "dec > −30, 3π", depth: "r 22", exposures: "24M", volume: "1.6 PB", tb: 1600, where: "MAST", url: "https://outerspace.stsci.edu/display/PANSTARRS/", step: 0, note: "DR2 single-epoch warps; later PS1 and PS2 observations need separate release and access checks." },
  { name: "ZTF", role: "Core optical", instrument: "P48", years: "2018–", sky: "dec > −31, 47 sq deg", depth: "r 20.5", exposures: "70M quadrants", volume: "~1 PB", tb: 1000, where: "IRSA · Fornax", url: "https://irsa.ipac.caltech.edu/Missions/ztf.html", step: 0, note: "DR24; alerts carry difference cutouts from day one" },
  { name: "PTF, iPTF", role: "Core optical", instrument: "P48", years: "2009–2016", sky: "dec > −30", depth: "R 21", exposures: "~3M chips", volume: "~100 TB", tb: 100, where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/ptf.html", step: 0 },
  { name: "SDSS imaging", role: "Core optical", instrument: "APO 2.5 m", years: "1998–2009", sky: "14,500 sq deg, Stripe 82 repeats", depth: "r 22.5", exposures: "~1M fields", volume: "16 TB", tb: 16, where: "SAS", url: "https://www.sdss.org/dr18/imaging/", step: 0 },
  { name: "Hyper Suprime-Cam", role: "Deep optical", instrument: "Subaru 8 m", years: "2014–", sky: "pointed, 1.8 sq deg", depth: "r ~26 (survey stack)", exposures: "~5M CCDs", volume: "~300 TB", tb: 300, where: "SMOKA, HSC-SSP PDR3", url: "https://smoka.nao.ac.jp/", step: 0, note: "Single-visit calibrated images and warps are distinct from coadds. The quoted survey depth is not a per-exposure limit; use visit metadata for orbit searches.", reference: "https://hsc-release.mtk.nao.ac.jp/doc/index.php/available-data__pdr3/" },
  { name: "Suprime-Cam", role: "Deep optical", instrument: "Subaru 8 m", years: "1999–2017", sky: "pointed, 0.25 sq deg", depth: "r 25.5", exposures: "1.2M", volume: "19 TB", tb: 19, where: "SMOKA", url: "https://smoka.nao.ac.jp/", step: 0 },
  { name: "MegaCam, CFH12K, WIRCam", role: "Deep optical", instrument: "CFHT 3.6 m", years: "1999–", sky: "north, 1 sq deg", depth: "r 24.5", exposures: "~0.5M", volume: "~350 TB", tb: 350, where: "CADC · CANFAR", url: "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/cfht/", step: 0, note: "CFHTLS, CFIS, OSSOS, NGVS; WIRCam is near-infrared and needs its own band-dependent sensitivity, not the optical r depth." },
  { name: "OmegaCAM", role: "Deep optical", instrument: "VST 2.6 m", years: "2011–", sky: "south, 1 sq deg", depth: "r 24", exposures: "~0.5M", volume: "~400 TB", tb: 400, where: "ESO archive", url: "https://archive.eso.org/", step: 0, note: "KiDS, VST ATLAS, VPHAS+; VST ATLAS is distinct from the ATLAS asteroid survey." },
  { name: "VIRCAM", role: "Infrared", instrument: "VISTA 4 m", years: "2009–", sky: "south, NIR", depth: "J 21", exposures: "~0.5M", volume: "~600 TB", tb: 600, where: "ESO archive", url: "https://archive.eso.org/", step: 0, note: "VHS, VVV, VIKING" },
  { name: "WFI", role: "Deep optical", instrument: "MPG/ESO 2.2 m", years: "1999–", sky: "south, 0.3 sq deg", depth: "R 23", exposures: "~0.2M", volume: "~30 TB", tb: 30, where: "ESO archive", url: "https://archive.eso.org/", step: 0 },
  { name: "SkyMapper", role: "Core optical", instrument: "1.35 m", years: "2014–2021", sky: "south, 5.7 sq deg", depth: "r 21.5", exposures: "~0.6M", volume: "~100 TB", tb: 100, where: "SkyMapper DR4", url: "https://skymapper.anu.edu.au/", step: 0 },
  { name: "Catalina Sky Survey", role: "Recovery", instrument: "703, G96, I52", years: "2003–", sky: "ecliptic to ±60", depth: "V 19.5 to 21.5", exposures: "~3M", volume: "~100 TB", tb: 100, where: "PDS SBN", url: "https://sbnarchive.psi.edu/pds4/surveys/gbo.ast.catalina.survey/", step: 0, note: "NEO survey frames; query the released date range and archive delivery lag." },
  { name: "Spacewatch", role: "Recovery", instrument: "0.9 m mosaic", years: "2003–2016", sky: "ecliptic", depth: "V 21.7", exposures: "~0.3M", volume: "~20 TB", tb: 20, where: "PDS SBN", url: "https://sbnarchive.psi.edu/pds4/surveys/gbo.ast.spacewatch.survey/", step: 0 },
  { name: "LONEOS", role: "Recovery", instrument: "0.6 m Schmidt", years: "Archived subset; see bundle", sky: "north", depth: "V 19.3", exposures: "~0.2M", volume: "~10 TB", tb: 10, where: "PDS SBN", url: "https://sbnarchive.psi.edu/pds4/surveys/gbo.ast.loneos.survey/", step: 0, note: "Archived dates have gaps; derive coverage from the selected PDS bundle, not the survey operating span.", reference: "https://sbn.psi.edu/pds/resource/loneos.html" },
  { name: "NEAT", role: "Recovery", instrument: "Palomar 1.2 m, Maui GEODSS", years: "1995–2007", sky: "ecliptic", depth: "R 20.5", exposures: "~0.2M", volume: "~10 TB", tb: 10, where: "PDS SBN", url: "https://sbnarchive.psi.edu/pds4/surveys/", step: 0 },
  { name: "INT WFC", role: "Deep optical", instrument: "INT 2.5 m", years: "1998–", sky: "north, 0.3 sq deg", depth: "r 22", exposures: "~0.3M", volume: "~20 TB", tb: 20, where: "ING, CASU", url: "https://casu.ast.cam.ac.uk/casuadc/ingarch/query", step: 0, note: "IPHAS, UVEX" },
  { name: "Las Cumbres", role: "Targeted", instrument: "0.4 m, 1 m, 2 m network", years: "2014–", sky: "pointed, both hemispheres", depth: "r 21", exposures: "~10M", volume: "~100 TB", tb: 100, where: "LCO archive", url: "https://archive.lco.global/", step: 0, note: "one-year proprietary period; follow-up, not survey" },
  { name: "GMOS", role: "Targeted", instrument: "Gemini 8 m", years: "2001–", sky: "pointed", depth: "r 25", exposures: "~0.3M", volume: "~100 TB", tb: 100, where: "Gemini archive", url: "https://archive.gemini.edu/", step: 0 },
  { name: "Keck imagers", role: "Targeted", instrument: "Keck 10 m", years: "1994–", sky: "pointed", depth: "r 26", exposures: "~0.5M", volume: "~50 TB", tb: 50, where: "KOA", url: "https://koa.ipac.caltech.edu/", step: 0 },
  { name: "WFCAM", role: "Infrared", instrument: "UKIRT 3.8 m", years: "2005–", sky: "north, NIR", depth: "K 18.3", exposures: "~1M", volume: "~100 TB", tb: 100, where: "WSA", url: "http://wsa.roe.ac.uk/", step: 0, note: "UKIDSS, UHS" },
  { name: "2MASS", role: "Specialized", instrument: "two 1.3 m", years: "1997–2001", sky: "all sky, NIR", depth: "Ks 14.3", exposures: "~60k scans", volume: "10 TB", tb: 10, where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/2mass.html", step: 0 },
  { name: "DSS plates", role: "Specialized", instrument: "Palomar and UK Schmidt", years: "1949–2002", sky: "all sky", depth: "R 21", exposures: "~7k plates", volume: "3 TB", tb: 3, where: "STScI, SuperCOSMOS", url: "https://archive.stsci.edu/cgi-bin/dss_form", step: 0, note: "POSS-I, POSS-II, SERC, AAO-R" },
  { name: "DASCH plates", role: "Specialized", instrument: "Harvard collection", years: "1885–1992", sky: "all sky", depth: "B 15", exposures: "429k plates", volume: "~320 TB", tb: 320, where: "DASCH DR7", url: "https://dasch.cfa.harvard.edu/", step: 0, note: "scanning finished 2024" },
  { name: "WISE", role: "Infrared", instrument: "0.4 m, 4 bands", years: "2010–2011", sky: "all sky, IR", depth: "W1 17", exposures: "~1.6M sets", volume: "~30 TB", tb: 30, where: "IRSA · AWS Open Data", url: "https://irsa.ipac.caltech.edu/Missions/wise.html", step: 0 },
  { name: "NEOWISE", role: "Infrared", instrument: "0.4 m, 2 bands", years: "2013–2024", sky: "all sky, IR", depth: "W1 16.5", exposures: "26.9M sets", volume: "~330 TB", tb: 330, where: "IRSA · AWS Open Data", url: "https://irsa.ipac.caltech.edu/data/WISE/docs/release/NEOWISE/expsup/", step: 0, note: "Use released single-exposure products and their actual date coverage; mission end is not the data-release date." },
  { name: "SPHEREx", role: "Infrared", instrument: "0.2 m, 102 bands", years: "2025–", sky: "all sky, IR", depth: "Wavelength-dependent", exposures: "Release-dependent", volume: "Not verified", tb: null, where: "IRSA · AWS Open Data", url: "https://irsa.ipac.caltech.edu/Missions/spherex.html", step: 0, note: "Public quick-release spectral images; QR3 began in September 2026. Preserve release, wavelength and exposure metadata.", reference: "https://irsa.ipac.caltech.edu/Missions/spherex.html" },
  { name: "TESS FFIs", role: "Specialized", instrument: "4 × 10 cm", years: "2018–", sky: "all sky, 24 × 96 deg", depth: "T 17 in 30 min", exposures: "~2M", volume: "~100 TB", tb: 100, where: "MAST · AWS Open Data", url: "https://archive.stsci.edu/missions-and-data/tess", step: 0, note: "High-cadence sector imaging. Candidate-orbit shift-and-stack searches can reach below single-frame limits; assess background, motion and spacecraft parallax.", reference: "https://arxiv.org/abs/1910.06383" },
  { name: "Kepler, K2 FFIs", role: "Specialized", instrument: "0.95 m", years: "2009–2018", sky: "fields near ecliptic (K2)", depth: "Kp 20", exposures: "~1k", volume: "~1 TB", tb: 1, where: "MAST", url: "https://archive.stsci.edu/missions-and-data/kepler", step: 0, note: "Sparse full-frame snapshots: K2 usually has two FFIs per campaign. Repeated target-pixel observations are listed separately.", reference: "https://archive.stsci.edu/k2/data_products.html" },
  { name: "GALEX", role: "Specialized", instrument: "0.5 m UV", years: "2003–2013", sky: "most sky, UV", depth: "NUV 20.5", exposures: "~100k", volume: "30 TB", tb: 30, where: "MAST · AWS Open Data", url: "https://archive.stsci.edu/missions-and-data/galex", step: 0 },
  { name: "Spitzer IRAC", role: "Infrared", instrument: "0.85 m IR", years: "2003–2020", sky: "pointed", depth: "3.6 µm 21 AB", exposures: "~10M BCDs", volume: "~100 TB", tb: 100, where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/spitzer.html", step: 0 },
  { name: "Hubble imagers", role: "Targeted", instrument: "2.4 m", years: "1990–", sky: "pointed", depth: "V 27", exposures: "~1M", volume: "~500 TB", tb: 500, where: "MAST · AWS Open Data", url: "https://archive.stsci.edu/missions-and-data/hst", step: 0 },
  { name: "JWST imagers", role: "Targeted", instrument: "6.5 m IR", years: "2022–", sky: "pointed", depth: "29 AB", exposures: "~0.2M", volume: "~300 TB", tb: 300, where: "MAST · AWS Open Data", url: "https://archive.stsci.edu/missions-and-data/jwst", step: 0 },
  { name: "Euclid Q1", role: "Targeted", instrument: "1.2 m, VIS + NISP", years: "2024 (Q1)", sky: "wide survey, 63 sq deg in Q1", depth: "Product-dependent", exposures: "See release", volume: "Not verified", tb: null, where: "ESA SAS, IRSA", url: "https://eas.esac.esa.int/sas/", step: 0, note: "Q1 includes single calibrated VIS/NIR exposures over 63.1 sq deg. Index released products; do not count projected DR1 volume as available data.", reference: "https://www.cosmos.esa.int/web/euclid/q1-contents" },
  { name: "Rubin data previews", role: "Access-dependent", instrument: "Simonyi 8.4 m, ComCam and LSSTCam", years: "2024–2026", sky: "a few fields", depth: "r 24", exposures: "~5k", volume: "~30 TB", tb: 30, where: "RSP", url: "https://data.lsst.cloud/", step: 0, note: "DP1 requires Rubin data rights. Record authorization and release separately from public alert availability.", reference: "https://dp1.lsst.io/access/index.html" },
  { name: "APPLAUSE plates", role: "Specialized", instrument: "German plate archives", years: "1893–1998", sky: "mostly north", depth: "B 15", exposures: "~100k plates", volume: "~50 TB", tb: 50, where: "plate-archive.org", url: "https://www.plate-archive.org/", step: 0 },
  { name: "WIYN ODI", role: "Deep optical", instrument: "WIYN 3.5 m", years: "2012–", sky: "pointed, 0.3 sq deg", depth: "r 24", exposures: "~0.1M", volume: "~30 TB", tb: 30, where: "ODI-PPA", url: "https://portal.odi.iu.edu/", step: 0 },
  { name: "NGTS", role: "Evaluate", instrument: "12 × 20 cm", years: "2015–", sky: "south, 96 sq deg", depth: "Product-dependent", exposures: "See release", volume: "Not verified", tb: null, where: "ESO archive", url: "https://archive.eso.org/", step: 0, note: "Documented public release includes light curves and stacked dithered images. Individual science-exposure access must be verified before enabling orbit-to-frame searches.", reference: "https://eso.org/cms/eso-archive-news/second-data-release-of-the-next-generation-transit-survey.html" },
  { name: "FORS2, HAWK-I", role: "Targeted", instrument: "VLT 8 m", years: "1999–", sky: "pointed", depth: "R 26", exposures: "~0.5M", volume: "~50 TB", tb: 50, where: "ESO archive", url: "https://archive.eso.org/", step: 0, note: "FORS2 optical and HAWK-I near-infrared imaging require separate filters and sensitivity estimates; the R label does not describe HAWK-I." },
  { name: "ATLAS", role: "Recovery", instrument: "ATLAS telescope network", years: "Archived subset; site-dependent", sky: "both hemispheres", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "PDS SBN · CATCH", url: "https://pdssbn.astro.umd.edu/data_sb/missions/atlas/index.shtml", step: 0, note: "Public reduced images and source detection catalogs are archived at PDS. Search actual site/date coverage; forced photometry is a separate service." },
  { name: "NEOSSat", role: "Recovery", instrument: "NEOSSat", years: "See archive", sky: "pointed, space-based", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "CSA · CADC", url: "https://www.asc-csa.gc.ca/eng/funding-programs/funding-opportunities/ao/2026-neossat-science-guest-observation-program-cycle-11.asp", step: 0, note: "Public FITS astronomy images. Use spacecraft observer positions when predicting apparent coordinates." },
  { name: "Kepler, K2 target pixels", role: "Specialized", instrument: "Kepler", years: "2009–2018", sky: "selected pixel regions", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "MAST", url: "https://archive.stsci.edu/k2/data_products.html", step: 0, note: "Repeated target-pixel observations complement sparse FFIs. Match actual downloaded pixel regions, not the entire camera field." },
  { name: "IRAS scans", role: "Infrared", instrument: "IRAS", years: "1983", sky: "all sky, infrared", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "IRSA · SCANPI", url: "https://irsa.ipac.caltech.edu/applications/Scanpi/", step: 0, note: "Calibrated survey scans, not ordinary image frames. Requires a time-aware scan adapter and a thermal-emission model." },
  { name: "AKARI", role: "Infrared", instrument: "AKARI", years: "See archive", sky: "survey and pointed infrared", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "JAXA DARTS · IRSA", url: "https://www.ir.isas.jaxa.jp/AKARI/Archive/", step: 0, note: "Maps, catalogs and pointed observations are different products. Verify time-resolved access before treating all-sky maps as orbit evidence." },
  { name: "Herschel imaging", role: "Targeted", instrument: "Herschel PACS, SPIRE", years: "See archive", sky: "pointed, far-infrared", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "Herschel Science Archive", url: "https://archives.esac.esa.int/hsa/whsa/", step: 0, note: "Public archive; prioritize orbit intersections. Retain observation times, observer positions and moving-target processing provenance." },
  { name: "SOFIA imaging", role: "Targeted", instrument: "SOFIA", years: "See archive", sky: "pointed, infrared", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "IRSA · MOST", url: "https://irsa.ipac.caltech.edu/applications/MOST/instructions.html", step: 0, note: "MOST supports archive searches. Select imaging products and evaluate sensitivity in the relevant band." },
  { name: "J-PLUS", role: "Evaluate", instrument: "JAST80", years: "Release-dependent", sky: "north, optical", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "CEFCA", url: "https://jplus.cefca.es/datareleases/releases", step: 0, note: "Public combined images and catalogs; verify individual-exposure access and constituent timestamps for the selected release before orbit matching." },
  { name: "S-PLUS", role: "Evaluate", instrument: "T80-South", years: "Release-dependent", sky: "south, optical", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "S-PLUS Cloud · Data Lab", url: "https://datalab.noirlab.edu/data/splus", step: 0, note: "Public images and catalogs; verify exposure-level access and timestamps. Single-mode photometry does not by itself mean single-exposure imagery." },
  { name: "SOHO LASCO", role: "Specialized", instrument: "SOHO LASCO", years: "See archive", sky: "near the Sun", depth: "Product-dependent", exposures: "See archive", volume: "Not verified", tb: null, where: "SOHO archive", url: "https://soho.nascom.nasa.gov/data/archive/", step: 0, note: "Useful for sungrazing-comet searches; low priority for distant-planet searches. Account for coronagraph masks and observer geometry." },
];

// Keep unknown-volume candidates available without diluting the planning table.
// Numeric volumes are planning estimates, not newly verified measurements.
export const QUANTIFIED_DATASETS = DATASETS.filter((dataset) => dataset.tb !== null);
export const DATASET_CANDIDATES = DATASETS.filter((dataset) => dataset.tb === null);

// Access limits to check per product; this is not a complete list of private data.
export const PRIVATE = [
  "Rubin LSST survey images (two-year proprietary period, data-rights holders only)",
  "Pan-STARRS after 2014 and all of PS2",
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
  { name: "Rubin alerts", holds: "difference detections, matched", size: "Design: ~10M/night", years: "2026–", where: "brokers", url: "https://rubinobservatory.org/for-scientists/data-products/alerts-and-brokers", step: 0, note: "Design-scale rate, not a measured nightly count. Public alerts do not imply public access to full survey images." },
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
  { name: "NOIRLab Source Catalog DR2", holds: "per-exposure detections in DR2 coverage", size: "68B measurements", years: "2004–2019", where: "Data Lab", url: "https://datalab.noirlab.edu/nscdr2/index.php", step: 0 },
  { name: "Pan-STARRS DR2 detections", holds: "single-epoch detections", size: "~10B", years: "2010–2014", where: "MAST", url: "https://outerspace.stsci.edu/display/PANSTARRS/PS1+Database+object+and+detection+tables", step: 0 },
  { name: "ZTF DR24 sources", holds: "single-epoch detections", size: "1,043B", years: "2018–2025", where: "IRSA", url: "https://irsa.ipac.caltech.edu/Missions/ztf.html", step: 0 },
  { name: "unTimely", holds: "unWISE epoch detections", size: "23B", years: "2010–2020", where: "NERSC", url: "https://portal.nersc.gov/project/cosmo/data/unwise/neo7/untimely-catalog/", step: 0 },
  { name: "NEOWISE single-exposure sources", holds: "L1b detections", size: "~150B", years: "2013–2024", where: "IRSA", url: "https://irsa.ipac.caltech.edu/data/WISE/docs/release/NEOWISE/expsup/", step: 0 },
  { name: "SkyMapper DR4 photometry", holds: "per-image detections", size: "~10B", years: "2014–2021", where: "SkyMapper", url: "https://skymapper.anu.edu.au/", step: 0 },
  { name: "Rubin DP1 sources", holds: "difference and forced detections", size: "~100M", years: "2024–2025", where: "RSP", url: "https://dp1.lsst.io/", step: 0, note: "Requires Rubin data rights; this is not an unrestricted public catalog.", reference: "https://dp1.lsst.io/access/index.html" },
  { name: "Euclid Q1 MER", holds: "sources in Q1 fields", size: "~30M", years: "2024–", where: "ESA SAS", url: "https://eas.esac.esa.int/sas/", step: 0 },
  { name: "Gaia DR4 epochs (planned)", holds: "planned epoch astrometry and photometry", size: "Pending release", years: "2014–2020", where: "Gaia archive", url: "https://www.cosmos.esa.int/web/gaia/dr4", step: 0, note: "Not available as of the 2026-10-20 review; ESA expects release in December 2026. Counts must be checked after release.", reference: "https://www.esa.int/Science_Exploration/Space_Science/Gaia" },
  { name: "Gaia DR3 sources", holds: "every bright star", size: "1.8B", years: "2014–2017", where: "Gaia archive", url: "https://gea.esac.esa.int/archive/", step: 0 },
  { name: "Legacy Surveys DR10", holds: "every fixed source, deep", size: "~2.5B", years: "", where: "legacysurvey.org", url: "https://www.legacysurvey.org/dr10/", step: 0, note: "Useful for background-source checks and exposure metadata. Deduplicate underlying DECam, BASS and MzLS observations against instrument archives.", reference: "https://www.legacysurvey.org/dr10/files/" },
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
  { name: "ATLAS source catalogs", holds: "per-exposure source detections", size: "See archive", years: "Archived subset", where: "PDS SBN", url: "https://pdssbn.astro.umd.edu/data_sb/missions/atlas/index.shtml", step: 0, note: "Linked to archived reduced images; distinct from known-object associations and forced photometry." },
  { name: "IRSA MOST", holds: "orbit to image lookup", size: "", years: "", where: "IRSA", url: "https://irsa.ipac.caltech.edu/applications/MOST/instructions.html", step: 0, note: "Accepts supplied orbital elements; searches WISE/NEOWISE, ZTF, PTF, Spitzer, SPHEREx, 2MASS and SOFIA." },
  { name: "CATCH", holds: "moving-target and sky-region image lookup", size: "", years: "", where: "PDS SBN", url: "https://catch.astro.umd.edu/about", step: 0, note: "Public API for survey coverage including ATLAS and other PDS holdings. Check supported orbit inputs and indexed date ranges for each integration." },
];
