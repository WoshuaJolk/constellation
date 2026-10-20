# Archive frames for agents

`magictools frames` searches image indexes by field, designation, orbit family,
MPC80 arc, or supplied ephemeris. It writes native product IDs, retrieval links,
query receipts and reusable CSV/JSON manifests. It does not download images or
infer detectability from a survey's nominal depth.

This is a local Python CLI, independent of the constellation website. It adds
no runtime dependencies to `magictools-astro`. Python 3.10+ and POSIX advisory
locking are required (Linux/macOS). Install from this checkout:

```sh
python -m pip install -e .
magictools frames capabilities
magictools frames credentials status
magictools frames search examples/frames/field.json --run-dir runs/field
```

## Agent workflow

1. Inspect `capabilities`; choose `providers: "all"` or named providers.
2. Run `search ... --plan` for a validated, network-free query plan.
3. Run the search in a new run directory. JSON goes to stdout; progress events
   go to stderr. `--format ndjson` streams progress/receipts to stdout; final
   frames remain in the run artifacts. `--format csv` prints the final CSV.
4. Inspect **both** `checks` and `unqueried`. A zero count with `error`, `blocked`,
   `limited`, or an unqueried service does not establish absence of observations.
   Read `coverageAssessment` or run `audit` to assess the evidence behind matches.
5. Retrieve the desired native products using `url`, `linkKind` and `products`.
   Some archives supply a product manifest or registered request page instead
   of an immediately downloadable FITS file.

```sh
magictools frames status runs/field
magictools frames audit runs/field --checksums
magictools frames search examples/frames/field.json --run-dir runs/field --resume --timeout 900
magictools frames merge runs/field/manifest.json runs/orbits/manifest.json -o combined.json
```

Exit codes: **0** all scheduled checks completed; **3** incomplete archive
checks; **2** invalid input/configuration; **4** artifact-integrity audit failure;
**130** interrupted. Unqueried providers make the search incomplete. Successful
execution is never a claim about all public surveys or detectability.

There is no client row cap, date-span cap, or orbit-family size cap. The default
per-request timeout is 600 seconds; set `timeoutSeconds` or `--timeout`.
`concurrency` defaults to 3 (maximum 16), with one in-flight request per archive
host. Archives retain their own limits and may reject large requests.

## Inputs and units

Every manifest has `mode`, `start`, and `end` (UTC `YYYY-MM-DD`, end exclusive).
RA, Dec and `radius` are degrees in ICRS. `eellipse` is **arcseconds**, passed as
SSOIS's box parameter; the same angular value pads direct ephemeris searches.
It is not an uncertainty probability or a full covariance ellipse.

| Mode | Required additional fields |
| --- | --- |
| `field` | `ra`, `dec`; optional `radius` (default point query) |
| `object` | `object`; optional SSOIS `resolver` |
| `orbits` | nonempty `orbits` array, unique `label` per sample |
| `arc` | intact `arc` text or `arcFile`, at least two MPC80 rows |
| `ephemeris` | `ephemeris` array or `ephemerisFile`, plus `observer` |

File references are relative to the input manifest. `providers` defaults to
`"all"`. `"most"` in a provider array expands all seven MOST catalogs; use
`mostCatalogs` to select a subset, or explicit IDs such as `most:wise_merge`.

**Names:** `resolver` accepts `bynameCADC` (default), `bynameMPC`,
`bynameHorizons`, and `bynameLowell`. This selection applies to SSOIS; MOST and
CATCH use their own name services. Newly designated objects may resolve in MPC
before Horizons.

**Orbits:** provide `epochMjd` (or `epochJd`), `e`, `i`, `node`, `peri`, and
`label`. Angles are degrees, distances au. Elliptic samples additionally have
`a` and `meanAnomaly`. Cometary samples have `kind: "cometary"`, `q`, and
`tpJd` (**JD**, not MJD), including `e >= 1`. MOST sends these as `obj_type=Comet`.
`ephemStepDays` controls MOST sampling (default 0.25 days).

SSOIS's orbital endpoint requires semimajor axis and mean anomaly. Cometary
elements are converted for `e != 1`; exact parabolic inputs are reported
blocked there. In live checks, SSOIS rejected hyperbolic ephemeris generation
while MOST accepted the same cometary input. Acceptance with zero matches is
not a validation of the predicted orbit.

**Arcs:** `arcFitter: "bern"` (default) or `"mpc"` selects SSOIS's fitter.
These are the actual CGI `search` values; the CLI mode is named `arc`.

**Ephemerides:** JSON/CSV rows contain `date` or UTC `mjd`, decimal-degree `ra`,
`dec`, optional `sigmaArcsec`, and optional track `label`. The reader also
accepts find_orb's dated sexagesimal output with trailing sigma and PA columns;
sigma suffixes `m`, `u`, `d` mean milliarcseconds, microarcseconds and degrees.
Unrecognized dated lines fail rather than silently removing epochs. Use the
CSV interchange format for other find_orb output layouts.

```csv
date,ra,dec,sigmaArcsec,label
2025-01-01,359.8,2,60,nominal
2025-01-02,0.1,2.1,90,nominal
```

Declare `observer`, e.g. `T09` or `geocenter`. The tool does **not** convert
between observers or time scales. Supply separate appropriate tracks/runs for
different observers, especially spacecraft and nearby objects. All the
providers in one run receive the supplied track; observer suitability remains
the caller's responsibility.

Ephemeris samples are interpolated on great-circle segments with linearly
interpolated sigma. Sampling must resolve the motion. The search uses
conservative monthly cones and then encloses the predicted trail throughout
each exposure before testing its footprint or pointing envelope. The midpoint
prediction remains in the output for reference. No extrapolation is performed.
Broad combined products and missing geometry remain labeled candidates. Trail
envelopes retain possible crossings; they do not integrate flux along a trail.

`sigmaMultiplier` (default 1) scales the supplied angular envelope. Direct
matching adds `radius + (sigmaMultiplier * sigmaArcsec + eellipse)/3600`.
The legacy SSOIS uploaded-ephemeris endpoint receives a conservative maximum
sigma per window; live requests returned HTTP 500 during validation. Direct
field adapters provide the working per-date uncertainty path.

You can attach `ephemeris`/`ephemerisFile` and `observer` to an orbit or object
manifest to cross-check direct archives alongside native moving-object
services. Without a supplied track, direct field services are explicitly
unqueried; the tool does not silently propagate orbits itself.

## Connections

The registry contains **49 built-in provider IDs**, plus arbitrary configured
SIA1, SIA2, and ObsCore services. All **51 imaging entries** in the
constellation inventory are mapped to adapters and explicit coverage limits:

```sh
magictools frames coverage
```

The report is also included in `capabilities` and each run manifest, where it
shows which providers were queried. This is not a claim that all exposures are
available: HSC and Rubin need accounts,
and several services supply only points, stacks, cutouts, or released subsets.

| Provider ID | Search coverage | Inputs |
| --- | --- | --- |
| `ssois` | CADC SSOIS indexed instruments | object, orbit family, arc, ephemeris |
| `smoka` | direct HSC and Suprime-Cam CCD records | field, ephemeris |
| `noirlab` | direct DECam instcal image products, including public survey programs | field, ephemeris |
| `hsc` | HSC-SSP PDR3 wide and revised deep/ultradeep frame tables | field, ephemeris; account required |
| `cadc` | CADC ObsCore image collections | field, ephemeris |
| `eso` | ESO ObsCore image collections | field, ephemeris |
| `irsa` | released image collections exposed by IRSA SIA2 | field, ephemeris |
| `mast` | MAST CAOM image collections and product lists | field, ephemeris |
| `catch` | all 13 indexed PDS sources: ATLAS sites, Catalina sites, LONEOS, NEAT, PS1 DR2, SkyMapper DR4, Spacewatch | field, ephemeris, named object |
| `ztf` | direct ZTF IBE science quadrants | field, ephemeris |
| `most:ztf`, `most:ptf` | ZTF and PTF moving-object indexes | object, orbit family |
| `most:wise_merge` | WISE/NEOWISE merged exposure index | object, orbit family |
| `most:spitzer_bcd`, `most:2mass` | Spitzer BCD and 2MASS | object, orbit family |
| `most:sofia`, `most:spherex` | SOFIA and SPHEREx | object, orbit family |
| `noirlab:mosaic`, `noirlab:90prime`, `noirlab:newfirm`, `noirlab:odi` | Nine instrument names covering Mosaic generations, 90Prime, NEWFIRM, ODI/pODI | field, ephemeris |
| `gemini` | Canonical Gemini imaging OBJECT records, including GMOS | field, ephemeris |
| `koa` | Nine Keck imaging instruments, separate resumable queries | field, ephemeris |
| `lco` | LCO EXPOSE records, full cursor pagination, polygon intersections | field, ephemeris |
| `wfcam` | UKIDSS DR11PLUS and UHS DR3, full SQL result files | field, ephemeris |
| `euclid` | ESA ObsCore, including individual Q1 calibrated frames | field, ephemeris |
| `herschel` | Public PACS/SPIRE observation bundles with epoch ranges | field, ephemeris |
| `jplus` | J-PLUS DR4 individual reduced images and observation MJD | field, ephemeris |
| `splus` | Public ADSS image collections; stack constituent epochs unavailable | field, ephemeris candidates |
| `sdss`, `dss` | SDSS FITS images / DSS cutout descriptors; SIA1 omits epochs | field, ephemeris candidates |
| `applause` | DR4 plate solutions and exposure dates; ambiguous associations retained | field, ephemeris candidates |
| `dasch` | DR7 exposure table, point-only API | point; region/track queries explicitly partial |
| `scanpi` | IRAS detector scans and UTC times, not image frames | point; region/track queries explicitly partial |
| `akari` | JAXA phase 1/2 IRC pointed observation bundles | field, ephemeris candidates |
| `lasco` | VSO native LASCO frame IDs/times, conservative solar envelope | field, ephemeris candidates |
| `mast:targetpixels` | Kepler/K2 target-pixel FITS cubes; light curves excluded | field, ephemeris candidates |
| `rubin` | RSP SIA2 authorized image collections | field, ephemeris; token/data rights required |
| `iphas` | IGAPS IPHAS/UVEX FITS metadata index | field, ephemeris; public HTTP index |
| `alma` | Science Archive images/cubes; native spectral-window identities, DataLink products | field, ephemeris candidates |
| `nrao` | VLA/VLASS and other NRAO image/visibility records; portal retrieval | field, ephemeris candidates |
| `casda` | Released ASKAP images and cubes, including undated products | field, ephemeris candidates |
| `heasarc` | Multi-mission ObsCore images with indexed footprints and times | field, ephemeris |
| `chandra` | Chandra direct image products, exposure times converted from ks | field, ephemeris |
| `xmm`, `xmm:slew` | XMM-Newton pointed bundles and slew products, with public retrieval URLs | field, ephemeris candidates |
| `gavo` | Heterogeneous published GAVO image/cube collections | field, ephemeris candidates |
| `kmtnet`, `miris` | KASI public FITS metadata, UTC times, conservative camera envelopes | field, ephemeris candidates; exhaustion unverified |


SMOKA's encoder coordinates and NOIRLab's pointing metadata give conservative
camera envelopes, **not exact CCD footprints**. SMOKA returns per-detector
native frame IDs; thousands of CCD records need not mean thousands of
exposures. HSC frame corners, ObsCore regions, ZTF corners and CATCH footprints
enable tighter geometric filtering when available.

MAST returns observation/product groups, including stacks; inspect their
product manifests for individual files. Its position selection depends on
MAST's own coverage index. IRSA SIA2 is not a guarantee that every IRSA mission
has time-resolved frames. CATCH has a 2-degree radius limit; wider query windows
are explicitly blocked. MOST has no equivalent to SSOIS's `eellipse` parameter;
receipts warn when it cannot apply that padding.

NOIRLab and MAST paginate. TAP/SIA overflow flags and SMOKA's 20,000-row
boundary trigger persisted date subdivision down to one day. Remaining
overflow stays `limited`. Splitting cannot fix every archive limit; critically,
SSOIS results are never reduced to an arbitrary first 300 rows.
SIA1 and KASI do not accept the temporal constraint; their overflow is retained
without issuing identical queries recursively.

## Bring an archive endpoint

Custom providers live in the input manifest, with IDs prefixed `vo:`. No code
change or global registry mutation is required. `providers: "all"` includes
these services alongside the built-ins. Endpoint configuration is part of each
persisted job's identity, so resuming cannot silently switch archives.

```json
{
  "mode": "field", "ra": 10, "dec": 20, "radius": 0.1,
  "start": "2000-01-01", "end": "2026-01-01",
  "providers": ["vo:my-archive"],
  "services": {
    "vo:my-archive": {
      "protocol": "obscore",
      "endpoint": "https://example.org/tap/sync",
      "table": "ivoa.obscore",
      "productTypes": ["image", "cube"],
      "includeUnknownAxes": true
    }
  }
}
```

Use the actual query endpoint: TAP `/sync`, SIA1 query URL, or SIA2 query URL.
Existing URL query parameters, such as a collection selector, are preserved.
Options include `collections`, `productTypes` (`image`, `cube`, `visibility`),
`maxrec` (explicit server boundary; overflow never means complete), `spatial`
(`footprint` or `pointing`), `timeScale` (`UTC` or `archive-defined`), and
`credentialProfile`. Table identifiers are validated; collection literals are
escaped. The implementation uses synchronous TAP with local resumable jobs and
date subdivision on actual overflow; it does not yet submit remote TAP/UWS jobs.

ObsCore queries normally include records with missing spatial/time axes as
unresolved candidates. This can be expensive on large indexes.
`includeUnknownAxes: false` explicitly excludes them and records that scope in
the receipt. HEASARC uses this setting because its null-inclusive query timed
out in validation. SIA2 applies the archive's TIME filter, which can likewise
omit undated products. SIA1 filters returned observation dates locally.

Find services and inspect their schema before querying:

```sh
magictools frames discover --text Chandra --output-dir runs/discovery
magictools frames discover --protocol sia2 --text infrared --output-dir runs/infrared-services
magictools frames inspect alma --output-dir runs/alma-schema
magictools frames inspect my-service.json --output-dir runs/custom-schema
magictools frames search examples/frames/custom-services.json --run-dir runs/custom --plan
magictools frames search examples/frames/multiwavelength.json --run-dir runs/crab
```

`discover` queries RegTAP and returns candidate service configurations. A TAP
registration does not guarantee an ObsCore table or image content; `inspect`
checks the schema and never claims coverage. Inspection/discovery save native
responses, SHA-256 receipts, and `report.json` in a new output directory.

To resolve a public DataLink URL returned by an archive:

```sh
magictools frames links 'https://archive.example/datalink?ID=native-id' --output-dir runs/product-links
```

This returns native links, semantics, errors, and service references without
downloading images or executing staging/cutout services. Service descriptors
remain in the saved native response. Authenticated DataLink staging is not yet
automated; use the archive's authenticated retrieval interface.

## Coverage evidence and audits

Every new receipt includes `coverageAssessment`: query success, index-exhaustion
evidence, product-kind counts, UTC intervals, parseable footprints, retrieval
references, and unresolved gaps. A valid empty table differs from an error,
an invalid schema, or a response that lacks a completion marker.

- `evidence-found`: at least one returned product has a bounded UTC interval
  and an intersecting footprint. It is not proof of a detection.
- `candidates-only`: potentially useful products need time/geometry inspection.
- `no-matches-in-queried-index`: the exhausted, scoped index returned no matches.
- `unverified`: exhaustion or successful execution was not established.

`entireRequestedRegionCovered` remains null: this CLI does not integrate sky/time
coverage masks. `allObservationsCovered` and `detectabilityAssessed` remain false.
The assessment's counts describe query windows and can overlap; final frame
lists are deduplicated separately. Legacy receipts without evidence fields
remain unverified. `audit --checksums` additionally verifies the latest saved
responses against their SHA-256 receipts and exits 4 on missing/corrupt artifacts.

Moving-target matching encloses the full predicted trail during each exposure,
including per-date uncertainty, rather than discarding a crossing because its
midpoint misses the detector. These envelope intersections are candidates until
the actual trail, chip masks, and observing location are checked. Products with
unknown constituent times are retained without assuming an exposure midpoint.

## Credentials

Public queries need no configuration. To prepare account-dependent archives:

```sh
magictools frames credentials init
magictools frames credentials status
```

`init` creates `~/.config/magictools/credentials.json` with mode `0600` and
environment references. It refuses to overwrite an existing file. Set secrets
through your environment/secret manager; do not put them in search manifests.

| Profile | Environment variables |
| --- | --- |
| HSC-SSP | `HSC_SSP_CAS_USER`, `HSC_SSP_CAS_PASSWORD` |
| MAST | `MAST_API_TOKEN` |
| NOIRLab | `NOIRLAB_API_TOKEN` |
| IRSA | `IRSA_USER`, `IRSA_PASSWORD` |
| LCO | `LCO_API_TOKEN` (optional for public metadata) |
| Gemini | `GEMINI_ARCHIVE_SESSION` (optional archive session cookie) |
| Rubin | `RSP_API_TOKEN` (`read:image` scope and data rights required) |
| DASCH | `DASCHLAB_API_KEY` (optional; selects the full API tier) |
| S-PLUS | `SPLUS_API_TOKEN`, or `SPLUS_USERNAME` / `SPLUS_PASSWORD` |

Custom environment names or mounted secret files are supported:

```json
{
  "providers": {
    "hsc": {
      "username": {"env": "MY_HSC_USER"},
      "password": {"file": "/run/secrets/hsc-password"}
    }
  }
}
```

Use `--credentials /path/to/config.json` on `search` or `serve`.
Custom services also accept that option on `inspect`. Their credentials require
an explicit origin grant in the private credential file, separate from the
search manifest:

```json
{
  "providers": {
    "my-archive": {
      "token": {"env": "MY_ARCHIVE_TOKEN"},
      "allowedOrigins": ["https://example.org"]
    }
  }
}
```

Set `credentialProfile: "my-archive"` in the service definition. HTTPS origins
must match exactly, including an explicit port. Username/password profiles are
also supported. Anonymous custom endpoints receive no ambient credentials.
Credential config and secret files must be owner-only on POSIX. Literal
credential strings are accepted in that private config, but environment/file
references make rotation easier. Status reports availability, never values,
and explicitly does not claim the archive has validated them.

Authentication headers and HSC credential bodies are omitted from receipts.
Known secret values are redacted from errors and recorded text responses;
redacted response hashes describe the saved bytes. Authorization, cookies, and API keys are removed on cross-origin redirects.
S-PLUS login responses stay in memory and are never written to receipt artifacts.
Credentials are never transmitted over HTTP; WFCAM's legacy HTTP service is
queried anonymously, as is the IGAPS metadata host. No HSC or Rubin account was available during development,
so authenticated execution remains unverified; missing-account behavior is tested.

## Durable results and provenance

Each run contains:

- `run.sqlite`: checkpoints and normalized records; one writer per directory.
- `queries/<job>/<attempt>/`: native responses, SHA-256 hashes, public request
  parameters, `frames.ndjson`, and `receipt.json`. Retrying preserves attempts.
- `manifest.json`: input, frames, checks, unqueried providers and limitations.
- `frames.csv`: compact request list; JSON preserves all native metadata,
  product variants, hypotheses, and provenance.

Downloads stream to disk; parsing and final JSON assembly use memory. This is
not a constant-memory pipeline for billion-row all-sky requests. Completed
queries are checkpointed; an interrupted individual query restarts. Resume
skips successful queries, retries errors/blocked/limited queries, and retries
malformed-row partial results only with `--retry-partial`. Timeout/concurrency
may change on resume; scientific input must match.

Bad rows are counted without discarding good rows. A failed later page or
retry preserves earlier valid records. `rejectedRows` differs from
`excludedRows` (valid rows outside the requested time/geometry). Dedupe uses
normalized instrument plus native product identity, never just time/position.
Recognized ZTF filenames and CFHT exposure IDs merge cross-service records.
Different detector/products remain distinct; unresolved identifiers are not
fuzzily collapsed. Product links and provenance are retained when keys merge.

## Optional local job API

The CLI can expose the request contract to another agent process:

```sh
magictools frames serve --directory runs --port 8765
curl -H 'Content-Type: application/json' --data-binary @examples/frames/field.json \
  http://127.0.0.1:8765/api/frames
```

`POST /api/frames` returns **202** with `pollUrl` and a CLI `resumeCommand`.
`GET /api/frames` returns capabilities; `GET <pollUrl>` returns checks/results
as they are checkpointed. Jobs survive as files after restart; resume an
interrupted job through the CLI. The API binds only to loopback, rejects
browser-origin requests, and accepts inline ephemeris/arc data rather than
arbitrary local file references. There is no website or hosted service.

## Validation and prior art

The full S60 fixture from the web repo's pinned SSOIS audit has **12,218 data
rows plus a header** (12,219 lines); the regression retains every row. Tests
also cover 64 orbit samples over 26 years, hyperbolic/parabolic MOST parameters,
uncertainty and RA wrap, exposure midpoints, pagination, overflow subdivision,
resume, malformed rows, dedupe and credential handling.

Live checks on 2026-10-20 returned 3,472 SMOKA CCD candidates, 39 DECam
products, 144 MAST products, 770 CATCH products, ESO records and WISE matches
in selected fields/date ranges. CATCH also returned four products for named
object `12` over one day; combined strings such as `12 Victoria` were rejected
by its resolver, with the failure retained in the receipt.
These are protocol checks, not completeness
benchmarks. Some MOST catalog/date combinations returned empty bodies
(reported as errors); CADC timed out in a short 40-second check. Never turn
these failures into a claim of no coverage.

This tool consolidates the existing pinned SSOIS/DECam/HSC/SMOKA workflows and
adds durable provenance and direct archive cross-checks. Federated search is
prior art: [SSOIS](https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/ssois/documentation.html),
[MOST](https://irsa.ipac.caltech.edu/applications/MOST/MOSTProgramInterface.html)
and [CATCH](https://catch.astro.umd.edu/about) already provide archive searches.
Adapter references: [NOIRLab API](https://astroarchive.noirlab.edu/api/docs/),
[SMOKA](https://smoka.nao.ac.jp/),
[HSC official client](https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/raw/master/pdr3/hscReleaseQuery/hscReleaseQuery.py),
[HSC frame schema](https://hsc-release.mtk.nao.ac.jp/schema/schemata/pdr3.pdr3_wide.frame.json),
[MAST API](https://mast.stsci.edu/api/v0/),
[IRSA SIA](https://irsa.ipac.caltech.edu/SIA),
[CATCH API](https://catch-api.astro.umd.edu/).


### Expanded adapter validation (2026-10-20)

Live public queries returned KOA frames (5), Euclid calibrated images (32),
Gemini science images (3), LCO exposures (4), J-PLUS individual images (72),
WFCAM records (305), APPLAUSE candidates (2,312 after local footprint filtering),
DASCH exposures (15,717 retained), S-PLUS stacks (205), LASCO frames (4),
Kepler target-pixel products (49), IGAPS CCD frames (129 from its 527,736-row
index), and IRAS detector scans (20). Counts describe
specific validation queries, not archive volumes or independent exposures.
AKARI returned 730 candidates in the chosen region and preserved a receipt
count for 64 index rows lacking a bundle link. Herschel's public observation
query also returned native records; the adapter restricts instruments to PACS
and SPIRE, but bundles can contain spectroscopy as well as images.

SDSS and DSS image discovery works, but their SIA1 responses omit epochs.
S-PLUS constituent exposure dates are also unavailable. Such queries remain
`partial`, even when spatial discovery succeeds. Missing dates do not trigger
pointless recursive date subdivision. Combined bundles and pixel cubes are
never rejected on the assumption that one timestamp represents every member.

IGAPS/IPHAS serves its 527,736-row metadata index through documented HTTP;
HTTPS returns HTTP 503. The live FITS schema and UTC `utstart` epochs were
validated. Immutable metadata is shared within a run with per-query receipts,
and the parsed table is reused to avoid repeatedly loading the 40 MB download.
HSC and Rubin await account-backed validation. `magictools frames coverage` retains these limitations alongside
NGTS cadence, PS1 release, AKARI phase 3, and other collection-level gaps.

Protocol references:
[Gemini](https://archive.gemini.edu/help/api.html),
[KOA](https://koa.ipac.caltech.edu/UserGuide/program_interface.html),
[LCO](https://developers.lco.global/),
[WFCAM SQL](http://wsa.roe.ac.uk:8080/wsa/SQL_form.jsp),
[DASCH DR7](https://dasch.cfa.harvard.edu/dr7/web-apis/),
[APPLAUSE TAP](https://www.plate-archive.org/tap/tables),
[Euclid TAP](https://eas.esac.esa.int/tap-server/tap/tables),
[Herschel TAP](https://archives.esac.esa.int/hsa/whsa-tap-server/tap/tables),
[Rubin SIA](https://rsp.lsst.io/guides/api/sia.html),
[J-PLUS reduced images](https://archive.cefca.es/catalogues/vo/siap/jplus-dr4/reduced),
[S-PLUS ADSS client](https://github.com/schwarzam/adss/blob/main/adss/endpoints/images.py),
[SCANPI](https://irsa.ipac.caltech.edu/applications/Scanpi/docs/ScanpiProgramInterface.html),
[AKARI native index](https://data.darts.isas.jaxa.jp/pub/akari/AKARI-IRC_Image_Pointed_POIphase1-2_1.0.1/browse/dataList.html),
[VSO schema](https://sdac.virtualsolar.org/API/VSOi_rpc_literal.wsdl).

### Configurable services and multiwavelength validation (2026-10-20)

Live searches returned ALMA science products (10), ASKAP/CASDA products (134),
GAVO candidates (9 after local filtering), Chandra images (6), XMM pointed image
bundles (4), XMM slew products (12), a NRAO/VLASS image (1), KMTNet images (185),
and MIRIS images (11). ALMA schema inspection returned 73 columns; its DataLink
resolver returned seven product/service entries. These counts describe the
selected test fields/date windows, not independent exposures or completeness.

HEASARC's null-inclusive ObsCore query timed out at 240 seconds. A query requiring
indexed footprints and times returned four image records; this restriction is
explicit in the built-in profile and receipts. NRAO publishes invalid zero end
times for some images; these remain candidates with unknown temporal extent.
Some ASKAP cubelets lack epochs. KASI has no independent total/overflow marker,
so its queries remain partial even when the returned table is below the request
boundary. KMTNet/MIRIS image dimensions are degrees; exact footprints need FITS WCS.

KMTNet exposes actual FITS images and was added. OGLE publishes reference FITS
images, but a verified per-exposure search interface was not established here.
ASAS-SN's documented Sky Patrol client returns photometry/light curves, not an
image-frame index. MOA also remains an investigation candidate. None is advertised
as a working per-exposure adapter merely because a catalog or web page exists.

Sources: [ALMA notebooks](https://almascience.nrao.edu/alma-data/archive/archive-notebooks),
[NRAO scripted access](https://science.nrao.edu/facilities/vla/archive/scripted-access-to-the-nrao-archive),
[CASDA TAP](https://casda.csiro.au/casda_vo_tools/tap),
[Chandra archive](https://cxc.harvard.edu/cda/),
[XSA](https://www.cosmos.esa.int/web/xmm-newton/xsa),
[HEASARC VO resources](https://heasarc.gsfc.nasa.gov/navo/summary/resources.html),
[KASI API documentation](https://data.kasi.re.kr/confluence/display/KSDA/KASI+Science+Data+Archive+APIs),
[KASI releases](https://data.kasi.re.kr/notice),
[GAVO TAP](https://dc.g-vo.org/tap),
[OGLE reference images](https://ftp.astrouw.edu.pl/ogle/ogle3/maps/blg/),
[ASAS-SN client](https://github.com/asas-sn/skypatrol).
