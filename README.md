# Constellation

Astronomy tools for finding archival observations, testing moving-object
hypotheses in images, and analyzing what detections or non-detections imply.
Everything runs locally under one command: **`constellation`**.

## Install

Python 3.11+ is recommended for the full toolset:

```sh
pipx install 'const-ellation[research,agent] @ git+https://github.com/WoshuaJolk/constellation.git@main'
constellation doctor
constellation list
```

Omit `[research,agent]` for the core tools and archive search. Install from
GitHub; this project has no PyPI or npm release. The distribution is
`const-ellation`, the command is `constellation`, and the Python import namespace
remains `magictools`. See the [installation and agent guide](docs/agent-cli.md)
for upgrades and alternative installation methods.

## Find surveys and frames

Search by sky field, object name, orbit family, MPC80 arc, or your own ephemeris.
Constellation queries 49 built-in archive providers and supports additional
SIA/ObsCore endpoints. Results include native frame IDs, retrieval links,
per-provider failures, and auditable query receipts.

Save your coordinates and dates as `query.json`:

```json
{
  "mode": "field",
  "ra": 150,
  "dec": 2,
  "radius": 0.05,
  "start": "2020-01-01",
  "end": "2020-01-03",
  "providers": "all"
}
```

```sh
# Fast pass: live progress, candidate survey list, and available frames.
constellation find_frames query.json --run-dir runs/my-field

# Continue the same search without the fast-pass budget.
constellation find_frames query.json --run-dir runs/my-field --deep --resume

# Inspect evidence and verify saved response checksums.
constellation check_frames runs/my-field --checksums
```

The default fast pass targets 60 seconds, with a 10-second allowance per query.
Use `--budget 120` to change the pass budget or `--deep` for a thorough first
run. Unfinished work stays resumable. JSON goes to stdout; progress goes to
stderr. Each run saves `manifest.json`, `frames.csv`, and raw response receipts.
Exit code **3** means some checks remain incomplete, including a fast pass
that reached its budget.

A candidate survey is a place to look, not confirmed coverage. Returned frames
still need footprint, epoch, access, and detectability checks. Image retrieval
is a separate step. See the [archive guide](docs/frames.md) for input formats,
credentials, provider scope, streaming output, and timeout behavior.

## Tools

Each row is one tool. Alternate interfaces and old names are aliases, not
additional scientific capabilities. Click a name for its input contract.

| Tool | What it does |
| --- | --- |
| [`find_frames`](docs/frames.md) | Finds candidate surveys and frame references by field, object, orbit, arc, or ephemeris; saves resumable receipts. |
| [`check_frames`](docs/frames.md) | Checks a saved archive run's coverage evidence and optional response checksums. |
| [`kepler-propagate`](docs/agent-cli.md) | Propagates supplied Cartesian orbits with optional observer-dependent light time; invoke through `constellation run`. |
| [`map_cover`](docs/ambitious-tools.md) | Projects orbit samples onto local FITS images and checks usable pixels, masks, trails, and family coverage. |
| [`force_track`](docs/formats.md) | Measures signed flux and uncertainty at supplied track positions using Gaussian or measured PSFs. |
| [`dig_floor`](docs/formats.md) | Scores a finite trajectory bank with a joint pixel likelihood and stated Gaussian trial corrections. |
| [`fold_kepler`](docs/fold-kepler.md) | Accelerates repeated finite-bank searches with shared sums and precise candidate refinement. |
| [`inject_tracks`](docs/archive-workflow.md) | Injects tracks into supplied images and reruns the search to measure recovery. |
| [`judge_line`](docs/archive-workflow.md) | Tests candidate consistency with temporal splits, leave-one-out fits, residuals, and control tracks. |
| [`weigh_stamp`](docs/ambitious-tools.md) | Compares moving-source, stationary-source, and supplied artifact models on the same pixels. |
| [`learn_null`](docs/formats.md) | Updates supplied hypotheses after non-detections and ranks follow-up using calibrated detection efficiencies. |
| [`map_complete`](docs/selection-and-assignment.md) | Calculates exact finite-population cadence selection and survey-union probabilities. |
| [`inject_all`](docs/selection-and-assignment.md) | Simulates finite-population selection and reports Monte Carlo uncertainty and conditional count likelihoods. |
| [`match_all`](docs/selection-and-assignment.md) | Solves a finite weighted candidate-assignment problem and produces checkable certificates. |
| [`verify_match`](docs/selection-and-assignment.md) | Independently checks a saved assignment certificate. |
| [`prove_floor`](docs/prove-floor.md) | Bounds prior-average detection power for a supplied template family and noise model. |
| [`find_ejector`](docs/find-ejector.md) | Searches supplied stellar phase-space samples for possible past convergence. |
| [`ejector_prepare`](docs/find-ejector-catalogue.md) | Prepares Gaia astrometry and supplied distance/RV assumptions for stellar traceback analysis. |
| [`ejector_fit`](docs/ejector-fit.md) | Fits possible shared stellar origins in measured astrometry and radial velocity. |
| [`ejector_search`](docs/ejector-search.md) | Fits groups in a supplied stellar catalogue, evaluates search controls, and checks source counterparts. |
| [`ejector_bound`](docs/ejector-search.md) | Certifies or excludes encounters within a declared continuous model domain, or reports unresolved cases. |
| [`ztf_search`](docs/archive-workflow.md) | Queries ZTF exposures for the ZTF-specific cutout-preparation workflow; use `find_frames` for general discovery. |
| [`ztf_fetch`](docs/archive-workflow.md) | Downloads selected ZTF cutouts and prepares image, variance, mask, and PSF inputs. |
| [`fold-chart`](research/kepler_transform/chart_construction.md) | Applies the experimental bounded observer-chart transform to supplied Fourier fields. |
| [`fit_sky`](docs/fit-sky.md) | Infers interacting sources and instrument uncertainty jointly; selects approximate sequential or bounded research inference with `--engine`. |

`find_frames` and `check_frames` replace the names `search` and `audit`.
`fit_sky` is one tool with two distinct engines: `--engine sequential` (default,
resumable and approximate) and `--engine bounded` (small-scene research inference).
The agent interface `constellation run fit_sky` uses the same sequential workflow.
Old spellings remain compatibility aliases; see the [migration guide](docs/agent-cli.md#canonical-names-and-compatibility).
Setup, inspection, archive-service utilities, and synthetic demos are listed by
`constellation --help`.

`constellation list` lists command families and scientific tool IDs. Agent
operations support schema discovery, examples, validation and execution:

```sh
constellation example kepler-propagate --output orbit-example
constellation validate kepler-propagate orbit-example/input.json
constellation run kepler-propagate orbit-example/input.json --output orbit-result
```

Use your own calibrated images, uncertainties, trajectories and justified
model assumptions; examples are synthetic. Read the
[scientific scope and Python API](docs/scientific-scope.md) before interpreting
scores or limits. A successful command does not establish a discovery,
survey-wide completeness, or a calibrated detection probability.

## Research and development

Experimental work is documented separately: [Kepler transforms](research/kepler_transform/README.md),
[time compression](research/kepler_transform/temporal/README.md),
[finite-detector transforms](research/kepler_transform/detector_time/README.md),
[detection limits](research/detection_limit/README.md), and
[FitSky](research/fit_sky/README.md). The general all-orbit transform and
full-scene inference goals remain open; these experiments are not interchangeable
with the finite-bank tools above.

For a development checkout, run `python -m pip install -e '.[test,research,agent]'`
and `python -m pytest -q`. The informational Next.js website uses Node.js 24:
`npm ci`, then `npm run dev`; checks are `npm run lint` and `npm run build`.
See the [roadmap](docs/roadmap.md) for remaining work.
