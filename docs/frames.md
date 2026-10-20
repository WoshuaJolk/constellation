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
5. Retrieve the desired native products using `url`, `linkKind` and `products`.
   Some archives supply a product manifest or registered request page instead
   of an immediately downloadable FITS file.

```sh
magictools frames status runs/field
magictools frames search examples/frames/field.json --run-dir runs/field --resume --timeout 900
magictools frames merge runs/field/manifest.json runs/orbits/manifest.json -o combined.json
```

Exit codes: **0** all scheduled checks completed; **3** incomplete archive
checks; **2** invalid input/configuration; **130** interrupted. A completed run
can still contain `unqueried` providers: completion is relative to the scheduled
checks, never a claim about all public surveys.

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
conservative monthly cones and then compares each exposure midpoint with its
footprint or pointing envelope. No extrapolation is performed. Broad combined
products and missing geometry remain labeled candidates. This does not
integrate a fast target's trail throughout an exposure.

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

The registry currently contains **17 provider IDs**, covering multiple surveys
through each archive. The constellation inventory is larger than the connected
set; listing a survey in that inventory does not make it queryable here.

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
Credential config and secret files must be owner-only on POSIX. Literal
credential strings are accepted in that private config, but environment/file
references make rotation easier. Status reports availability, never values,
and explicitly does not claim the archive has validated them.

Authentication headers and HSC credential bodies are omitted from receipts.
Known secret values are redacted from errors and recorded text responses;
redacted response hashes describe the saved bytes. Authorization is removed
on cross-origin redirects. No HSC account was available during development,
so HSC execution remains unverified against authenticated service responses;
the public schema/client protocol and missing-account behavior are tested.

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
