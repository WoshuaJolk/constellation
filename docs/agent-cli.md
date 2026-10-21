# Kepler and FitSky for agents

`constellation` provides discovery, JSON Schemas, runnable examples, validation,
execution and inspection through one noninteractive command. Scientific engines
retain their existing model assumptions and accuracy scopes. Use Python 3.11 or later for the complete research installation below.

The Python distribution is named `const-ellation`; the executable is `constellation`. The
Python import package remains `magictools` for compatibility. If you installed
the former `magictools-astro` distribution, uninstall it before installing the
renamed package so that both distributions do not own the same files.

From this checkout:

```sh
python -m pip install '.[research,agent]'
constellation doctor
constellation list
```

Install directly from GitHub:

```sh
python -m pip install 'const-ellation[research,agent] @ git+https://github.com/WoshuaJolk/constellation.git@main'
```

Use the GitHub URL above until a registry release is published. The distribution
name `const-ellation` is currently unregistered on PyPI and npm; this repository
has not published a package to either registry. `pip install constellation`
refers to an unrelated project.

The equivalent module invocation is `python -m magictools.agent_cli`. No server,
login, detector catalogue or interactive prompts are required. Science inputs
are local files. The generated examples are synthetic, not discoveries.

The same executable provides [archive discovery](frames.md) through
`constellation find_frames`, `constellation capabilities`, and `constellation frames`.
Archive schema inspection uses `constellation frames inspect`; the scientific
`constellation inspect` command and its JSON contract remain unchanged.

## One command for every tool

Run `constellation list` for a JSON catalogue of scientific tools and all
command families, or `constellation --help` for command help.

| Family | Examples |
|---|---|
| Archive discovery | `constellation find_frames query.json --run-dir runs/search`, `constellation capabilities` |
| Scientific agent contracts | `constellation run fit_sky input.json --output runs/fit`, `constellation schema kepler-propagate` |
| Manifest tools | `constellation force_track input.json`, `constellation fold_kepler input.json`, `constellation ztf_search query.json` |
| Synthetic demonstrations | `constellation demo demo-output`, `constellation orbit_demo orbit-output` |
| Research interfaces | `constellation fold-chart --help`, `constellation fit_sky --help` |

The former `magictools` and `magictools-*` executables remain compatibility
aliases. Use `constellation` for new integrations. Each delegated tool keeps
its documented JSON, streaming and exit-code contract; the single-envelope
contract below applies to the scientific agent operations. Research commands
require the optional research dependencies.

## Canonical names and compatibility

Use `find_frames` for archive lookup and `check_frames` for checking saved
coverage evidence and receipts. `search` and `audit` remain aliases, including
under `constellation frames`. They operate on the same manifests and run directories.

FitSky is presented as one tool, `fit_sky`, with explicit engine selection:

```sh
constellation fit_sky input.json new-output --engine sequential
constellation fit_sky examples/fit-sky.json new-bounded-output --engine bounded --verify
constellation run fit_sky input.json --output new-agent-output
```

The direct sequential interface and the agent interface share initialization,
resume validation, assimilation, checkpointing, and report writing. The direct
interface streams summaries; the agent interface adds schema validation,
provenance, and a final JSON envelope. Bounded inference uses a different model
contract and does not share sequential checkpoints or accuracy guarantees.
Use `constellation fit_sky --engine bounded --help` for its options.

Compatibility spellings retain their original routing: `fit-sky` selects the
bounded engine, `fit-sky-stream` selects sequential inference, and the scientific
tool ID `fitsky` resolves to `fit_sky`. Checkpoint schema IDs remain unchanged.
`run kepler-fold` wraps `fold_kepler`; `run kepler-chart` wraps `fold-chart`.
These are alternate interfaces, not additional transform engines. `list` reports
aliases separately from canonical command names.

## Choose the scientific operation

| Tool ID | Input | Result | Scientific scope |
|---|---|---|---|
| `fit_sky` | JSON scene with pixels, source/count priors, instrument response, queries | Joint sequential inference report and resumable checkpoint | Approximate, uncertified posterior; retains blends, source identity, continuous orbital parameters and shared calibration |
| `kepler-propagate` | Cartesian states and times; optional reception observers | Positions/velocities and optional light-time-corrected RA/Dec | Numerical all-conic two-body propagation, no search/inference |
| `kepler-fold` | FITS exposure manifest plus finite elliptic-orbit bank or pixel trajectories | Scored/ranked orbit hypotheses and detection bounds | Existing finite-bank folding engine; **not** the PFKT time-compression research engine |
| `kepler-chart` | Bounded observer-chart domain and NPZ Fourier fields/query coordinates | Complex scores, geometry certificates and audits | Existing chart research transform; no raw-pixel adapter or whole-archive/full-score guarantee |

The complete FKT transform/inverse objective remains open. The newer PFKT
time-compression experiments remain in their research interface; this CLI does
not relabel the finite-bank or chart engine as that construction.

## Runnable FitSky workflow

```sh
constellation example fit_sky --output fit-example
constellation schema fit_sky
constellation validate fit_sky fit-example/input.json
constellation run fit_sky fit-example/input.json --output fit-first \
  --particles 256 --sweeps 8 --seed 42 --stop-after 2
constellation inspect fit-first
constellation inspect fit-first/checkpoint.npz
constellation run fit_sky fit-example/input.json --output fit-rest \
  --resume fit-first/checkpoint.npz --catalogues
```

The example has four exposures over 24 days, two instruments, star/orbit
ambiguity, continuous orbital velocity, overlapping source possibilities and
shared calibration. Its small finite count prior is declared explicitly.
Replacing this example with real data requires a scientifically justified
model/prior; a successful run is not evidence that the model fits the survey.

FitSky options are `--particles`, `--sweeps`, `--seed`, `--pixel-block`,
`--max-stages`, `--gibbs-flux`, `--split-merge`, `--catalogues`, `--stop-after`,
`--exposure-batch` and `--resume`. Omitted sampler settings use the existing
engine defaults (2,048 particles, 20 sweeps). `--max-stages` bounds tempering
stages, not wall-clock time. Memory/runtime still depend on the scene.

Resume preserves checkpoint settings; conflicting overrides fail. An input
manifest may append exposures while retaining the old observations, priors
and calibration definitions. A new output directory is required for every run.
The original checkpoint remains untouched. Legacy v1 non-orbital checkpoints
load under the integration's documented migration rules.

## Runnable Kepler workflows

```sh
constellation example kepler-propagate --output orbit-example
constellation run kepler-propagate orbit-example/input.json --output orbit-result

constellation example kepler-fold --output fold-example
constellation validate kepler-fold fold-example/input.json
constellation run kepler-fold fold-example/input.json --output fold-result

constellation example kepler-chart --output chart-example
constellation validate kepler-chart chart-example/input.json
constellation run kepler-chart chart-example/input.json --output chart-result
```

The fold example creates actual small FITS images with celestial WCS and an
elliptic orbit bank, then compares folded results with direct evaluation.
The chart example supplies synthetic Fourier coefficients and queries; its
scores are not a posterior or evidence that an object exists. Full result
vectors are in `engine/scores.npz`; inspect `report.json` for accuracy and
direct-fallback details. Fallback is disabled in the generated example.

## JSON protocol

Every operational invocation emits exactly **one JSON object on stdout**.
FitSky progress events go to stderr as JSON lines; native libraries may also
emit diagnostics there. `--help` is conventional human-readable text. Schema
and tool discovery need no scientific dependency imports. `doctor` reports
which dependencies are available; `ok=true` means the inspection itself worked,
not that every dependency is installed.

For example, a completed FitSky run returns an envelope shaped like:

```json
{
  "schema_version": 1,
  "ok": true,
  "command": "run",
  "tool": "fit_sky",
  "scientific_scope": "approximate_uncertified",
  "scientific_status": "approximate_uncertified",
  "full_objective_complete": false,
  "result": {"assimilated_exposures": 4},
  "output": "/absolute/path/to/run",
  "artifacts": {
    "report.json": "/absolute/path/to/run/report.json",
    "checkpoint.npz": "/absolute/path/to/run/checkpoint.npz"
  }
}
```

This illustration omits other summary/artifact fields. The complete result is
in `report.json`; stdout stays compact, even with `--catalogues`. A schema is
returned in the `result` member; extract it with
`constellation schema fit_sky | jq '.result'` when a validator needs the raw
JSON Schema. Unknown keys, duplicate JSON keys and nonfinite numeric literals
are rejected. JSON Schemas use draft 2020-12 and contain no remote references.

| Exit | Meaning |
|---|---|
| 0 | The requested operation completed; inspect scientific status separately |
| 2 | Usage, manifest, preflight or explicit settings error |
| 3 | Required dependency missing; JSON error includes installation guidance |
| 4 | Scientific execution failed or refused evaluation |
| 130 | Interrupted with Ctrl-C/SIGINT |

Errors return `ok=false` and `error: {code, message, exception}`. When a new run
directory exists, its `failure.json` and `run.json` retain that failure. The
checkpoint path and `resume_available` flag identify recoverable FitSky work.
An initial checkpoint is saved before assimilation and replaced after each
completed exposure batch. Hard process termination is not guaranteed to emit
JSON; the last completed checkpoint remains the recovery point.

An agent should distinguish **operation success** from **scientific accuracy**:
FitSky remains uncertified, and chart geometry certificates are not full
score/posterior certificates. `inspect` reads saved output without rerunning
inference; checkpoint inspection is metadata inspection, not full validation.

## Input conventions and validation boundaries

- `kepler-propagate`: `[rx,ry,rz,vx,vy,vz]`, heliocentric ICRS-equatorial AU and
  AU/day. Times are TDB-day offsets from the common state epoch. Optional
  observer rows are reception positions in the same frame. Returned array
  axes are orbit, time, state-coordinate; RA/Dec axes are orbit, time, angle.
  Constants are reported explicitly. They are the Kepler propagator defaults,
  not FitSky's slightly different model constants.
- `fit_sky`: source Cartesian orbits use the scene's zero epoch and consistent
  heliocentric frame. Exposure times are offsets, not absolute MJD. Tangent
  bases, observers and angular pixel scales are required for orbital classes.
  See [FitSky's complete model contract](fit-sky.md).
- `kepler-fold`: absolute `epoch_mjd_tdb` / `mjd_tdb` for orbital input; finite
  elliptic elements in AU/degrees. Pixel tracks use zero-based detector centers.
  See [fold input conventions](fold-kepler.md).
- `kepler-chart`: NPZ shapes/units are in its JSON Schema. The observer-chart
  coordinates are not Cartesian states or elliptic elements. Input `data` is
  complex Fourier coefficients; `parameters` contains the query coordinates.

`validate` performs schema/model/resource admission, and returns its exact
`validation_scope`. It does not run a search, prove that the whole orbital prior
is numerically supported, establish astrophysical completeness, or certify
posterior accuracy. Fold preflight checks resource existence and science FITS
headers; the engine checks pixel/noise/PSF contents during execution. Chart
certificates are built during execution, not preflight.

Resources resolve relative to the input manifest, regardless of the working
directory. The run's `input.json` resolves external resources to absolute paths;
`provenance.json` hashes the input, resources and optional resume checkpoint.
Large resources are referenced, not copied. To move/replay a run elsewhere,
transfer its external files and update paths in a new manifest. Do not modify
input files concurrently with a run.

No command overwrites an existing output directory. No candidate shortlist or
posterior truncation is added by this interface. Kepler and FitSky research
objectives, engines and archived experiment hashes remain unchanged.

Validation: 141 CLI/Kepler/FitSky regression tests passed, including 14 agent
protocol tests. An installed wheel ran all four workflows outside the checkout
and resumed FitSky through four exposures (18 successful workflow commands).
The full repository test suite was not rerun. See the
[validation record](agent-cli-validation.json) for source hashes and commands.
