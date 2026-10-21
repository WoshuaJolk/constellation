# Constellation

Local astronomy tools for finding archival frames, measuring moving sources,
and making explicit, testable inferences from detections and non-detections.

The tools below have CLI acceptance tests demonstrating a concrete improvement
over a stated baseline. See the [results and limits](docs/release-readiness.md).
They support bounded tasks with supplied data and models; they do not certify
astronomical discoveries or complete survey coverage.

## Install

Python 3.10+ on Linux/macOS:

```sh
pipx install 'const-ellation @ git+https://github.com/WoshuaJolk/constellation.git@main'
constellation doctor
constellation list --ready
```

The package is **`const-ellation`**; the command is **`constellation`**.
Install from GitHub until a registry release is published. Existing installations
can update with `pipx upgrade const-ellation`. Research dependencies are optional.
See the [installation guide](docs/agent-cli.md) for virtual environments and aliases.

## Tools

| Tool | What it does |
| --- | --- |
| [`find_frames`](docs/frames.md) | Finds candidate surveys and native frame references by field, object, orbit, arc, or ephemeris; deduplicates results and preserves failures and receipts. |
| [`check_frames`](docs/frames.md) | Checks saved coverage evidence and detects altered archive responses using checksums. |
| [`kepler-propagate`](docs/agent-cli.md) | Propagates supplied Cartesian states with two-body dynamics and optional observer-dependent light time. |
| [`map_cover`](docs/ambitious-tools.md) | Projects orbit samples into local images and checks usable coverage, detector masks, and off-chip positions. |
| [`force_track`](docs/formats.md) | Measures signed flux and uncertainty along a supplied track using Gaussian or measured PSFs. |
| [`dig_floor`](docs/formats.md) | Combines exposures to score a supplied finite trajectory bank, including individually subthreshold signals. |
| [`inject_tracks`](docs/archive-workflow.md) | Injects tracks into supplied images and reruns the search to measure recovery without changing the originals. |
| [`judge_line`](docs/archive-workflow.md) | Checks temporal consistency, leave-one-out stability, residuals, and control tracks before accepting a candidate. |
| [`weigh_stamp`](docs/ambitious-tools.md) | Compares moving-source, stationary-source, and artifact models on the same pixels; weights are conditional on those models. |
| [`learn_null`](docs/formats.md) | Updates discrete hypotheses after non-detections and ranks follow-up using supplied, calibrated detection efficiencies. |
| [`map_complete`](docs/selection-and-assignment.md) | Calculates exact finite-population cadence selection while retaining dependence from shared exposures. |
| [`inject_all`](docs/selection-and-assignment.md) | Estimates finite-population selection beyond enumeration limits, with Monte Carlo uncertainty; this is not pixel injection. |
| [`match_all`](docs/selection-and-assignment.md) | Finds optimal or bounded compatible assignments over supplied scored candidates and produces certificates. |
| [`verify_match`](docs/selection-and-assignment.md) | Independently checks an assignment certificate and rejects inconsistent results. |
| [`prove_floor`](docs/prove-floor.md) | Bounds prior-average detection power for a specified finite Gaussian template family. |
| [`ejector_fit`](docs/ejector-fit.md) | Fits specified stellar-origin hypotheses in measured astrometry and radial velocity, including separate launches from one source. |

## Find frames

Save your own query as `query.json` (ICRS degrees, UTC dates; end exclusive):

```json
{
  "mode": "field", "ra": 150, "dec": 2, "radius": 0.05,
  "start": "2020-01-01", "end": "2020-01-03", "providers": "all"
}
```

```sh
constellation find_frames query.json -o runs/mission
constellation check_frames runs/mission --checksums
constellation find_frames query.json -o runs/mission --deep --resume
```

`-o` is optional for a first search: a unique directory under `runs/` is chosen
and reported. The fast pass targets 60 seconds; `--deep` continues without that
budget. Progress goes to stderr, JSON to stdout. Saved results include native
product references, `manifest.json`, `frames.csv`, and receipts. Exit code **3**
means checks remain incomplete; missing coverage is never treated as an absence
of observations. Searches query 49 built-in providers plus configurable VO
endpoints. Image downloads are a separate step. See the [archive guide](docs/frames.md)
for credentials, uncertainty inputs, budgets, and release/access limitations.

## Analyze your own data

```sh
constellation force_track tracks.json -o fluxes.json
constellation dig_floor tracks.json -o candidates.json
constellation kepler-propagate orbit.json
```

Manifest tools take `INPUT.json` and optionally `-o OUTPUT.json`.
Orbit propagation creates a run directory automatically, or accepts `-o DIRECTORY`.
Paths inside a manifest resolve relative to that manifest. Each tool's linked
guide describes its input schema, output, and assumptions; use `--help` for options.
Images need justified variance, mask, PSF, and flux calibration. Detection scores
and model weights do not by themselves establish a real object.

## Validation and development

[Release acceptance](docs/release-readiness.md) records the baseline, measured
benefit, and scope for every listed tool. CI tests the built wheel from outside
the checkout, verifies package metadata, and saves the results and distributions
as artifacts. Reproduce with `python -m pip install -e '.[test]'` and
`python -m pytest tests/test_release_cli.py -q`; full research tests require
`.[test,research,accelerate]`. Detailed [scientific assumptions](docs/scientific-scope.md)
and [methods](docs/methods.md) remain part of the tool contracts.

Experimental or specialist tools, outside the promoted release set:
`fold_kepler`, `fold-chart`, `fit_sky`, `find_ejector`, `ejector_prepare`,
`ejector_search`, `ejector_bound`, `ztf_search`, `ztf_fetch`.
They remain callable; [status and promotion gaps](docs/release-readiness.md#not-promoted)
explain why they are not in the main table. `constellation list` includes their
interfaces and compatibility aliases.

[`solve_itf`](docs/solve-itf.md) starts a separate full-ITF linkage objective with
an executable small-scene reference, orbital ambiguity, consistent assignments,
and explicit unresolved-mass accounting. The full-snapshot 180-second target
and continuous-orbit coverage remain open; it is not a promoted release tool.
