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
constellation search query.json --run-dir runs/my-field

# Continue the same search without the fast-pass budget.
constellation search query.json --run-dir runs/my-field --deep --resume

# Inspect evidence and verify saved response checksums.
constellation audit runs/my-field --checksums
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

## Work with your own data

| Task | Tools | Guide |
| --- | --- | --- |
| Propagate orbits and check usable image coverage | `kepler-propagate`, `map_cover` | [Agent interface](docs/agent-cli.md), [orbit coverage](docs/ambitious-tools.md) |
| Measure flux along a supplied track | `force_track` | [Input formats](docs/formats.md) |
| Search a finite trajectory bank and measure injection recovery | `dig_floor`, `fold_kepler`, `inject_tracks` | [Formats](docs/formats.md), [orbit folding](docs/fold-kepler.md) |
| Examine candidate consistency and artifact alternatives | `judge_line`, `weigh_stamp` | [Methods](docs/methods.md), [stamp models](docs/ambitious-tools.md) |
| Update hypotheses after non-detections and rank follow-up | `learn_null` | [Formats and calibration inputs](docs/formats.md) |
| Calculate finite survey selection, population recovery, or assignments | `map_complete`, `inject_all`, `match_all` | [Methods](docs/methods.md) |
| Bound detection power under a specified model | `prove_floor` | [Detection bounds](docs/prove-floor.md) |
| Fit and assess possible common stellar origins | `find_ejector`, `ejector_fit`, `ejector_search`, `ejector_bound` | [Fits](docs/ejector-fit.md), [search and bounds](docs/ejector-search.md) |
| Infer interacting sources across images (experimental) | `fitsky`, `fit-sky`, `fit-sky-stream` | [Agent interface](docs/agent-cli.md), [FitSky](docs/fit-sky.md) |

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
