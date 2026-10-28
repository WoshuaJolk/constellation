# `solve_itf`: full-catalogue ITF candidate linkage

`solve_itf` downloads or accepts the **entire MPC Isolated Tracklet File**, builds
an approximately 59 MB portable numerical archive, scans the catalogue for
cross-night links, fits physical orbits to the original observations, and grows
accepted triples into longer chains. Chains have no fixed maximum tracklet count.
The ranked text output includes identifiers, stations, arc length, observation
count, angular RMS, ecliptic inclination, and a signed BIC comparison score.

**This is a bounded candidate finder, not an exhaustive orbit solver.** The
five-minute budget covers candidate scanning, fitting, and chain extension.
Finite proposal gates can miss links even if all retained seeds are fitted.
Legacy JSON BIC percentages remain for compatibility and are explicitly
uncalibrated; console and dashboard output use the signed comparison score.
Neither is discovery confidence or a complete catalogue posterior.

## Run on the whole ITF

From the repository root:

```sh
pip install -e '.[itf]'
solve_itf fetch --directory .cache/solve_itf
OPENBLAS_NUM_THREADS=1 solve_itf run \
  --input .cache/solve_itf/itf.txt.gz \
  --cache .cache/solve_itf/catalogue \
  --observatories .cache/solve_itf/ObsCodes.html \
  --output runs/solve-itf-first \
  --seconds 280
```

The repository-local equivalent is `python solve_itf ...`. The installed command
is also `solve_itf`. No raw-image or research-transform dependencies are needed.
Numba is used for parsing, tracklet reduction, all-conic propagation, analytic
astrometric derivatives, and the scaled six-parameter optimizer. `--workers 4`
is the default: day partitions and orbit fits run in separate processes, in
successive phases. Linux fitting workers inherit prepared arrays through
copy-on-write; spawn platforms reopen memory maps and prepare their observer
geometry locally. This avoids Python's global interpreter lock around the
compiled fitting calls. Use `--workers 1` to restrict CPU use.
For a fresh data cache with multiple workers, a separate process compiles the
fitting kernels while ingestion runs. It exits before the parallel scan starts;
`compiler_warmup` records its outcome and any wait. This overlaps startup work
without reducing the fitted seed bank or relying on a preinstalled JIT cache.

The first `run --input` includes cache creation in its time budget. A subsequent
run can use the cache alone:

```sh
OPENBLAS_NUM_THREADS=1 solve_itf run \
  --cache .cache/solve_itf/catalogue \
  --observatories .cache/solve_itf/ObsCodes.html \
  --output runs/solve-itf-next --seconds 280
```

There is no tracklet sample size, chosen station, sky patch, magnitude cut, or
chosen date interval in either command. Every supported observation is ingested.
`fetch` downloads the complete current MPC file and the observatory table; it
keeps an existing snapshot. Use a new directory to fetch a newer snapshot.
Downloads are separate from the search budget and depend on the connection.
Existing caches and output directories are not overwritten. Reusing `--input`
checks the source SHA-256 against the cache.

The limit is checked between search batches and inside optimizer evaluations.
Ingestion, JIT compilation, an individual library call, and final serialization
can overrun a very short requested budget. `280` leaves startup/output headroom;
it is not a hardware-independent runtime guarantee. The result identifies a
budget exhaustion even when it contains useful candidates.

To reproduce a fresh-data, fresh-JIT benchmark on a POSIX host, use
`python research/solve_itf/benchmark.py --input PATH --observatories PATH
--cache NEW_CACHE --output NEW_OUTPUT`. It refuses existing cache/output paths
and saves process wall time, resource usage, and the solver's full results.

## Broader coverage and larger seed banks

When more runtime is available, use the full-catalogue broad profile:

```sh
OPENBLAS_NUM_THREADS=1 solve_itf run --profile broad \
  --cache .cache/solve_itf/catalogue \
  --observatories .cache/solve_itf/ObsCodes.html \
  --output runs/solve-itf-broad
```

| Setting | `fast` (default) | `broad` |
| --- | ---: | ---: |
| Triple-seed budget | 10,000 | 70,000 |
| Pair separation limit | 14 days | 28 days |
| Neighbors per day/query | 8 | 16 |
| Retained edge budget | 300,000 | 1,000,000 |
| Neighbors combined into triple seeds | 5 | All retained neighbors |
| Orbit starting grid, plus Gauss roots | 24 | 44 |
| Runtime budget | 280 seconds | 3,600 seconds |

Explicit flags override profile defaults, for example `--profile broad
--max-seeds 100000 --seconds 7200`. `--seed-neighbors 0` considers every retained
neighbor when forming seeds, independently of the chain-growth `--branches`
limit. This does not remove the per-query neighbor or total edge caps.

The broad profile retains the linear-motion proposal graph and adds a second
search that projects short arcs to a common epoch using trial topocentric
distance, radial motion, solar acceleration, Earth reflex motion, and terrestrial
rotation. It searches distances 0.03, 0.1, 0.3, 1, 3, 10, 50, and 250 AU;
distances through 0.3 AU also try radial rates -0.02 and +0.02 AU/day. Nearby
trials use gaps up to seven days below 0.3 AU and fourteen days below 1 AU,
because the approximate propagation deteriorates faster there. Other trials
use the configured pair window. The additional angular gate is
`position_gate + 10 * curvature_gate * gap_days²` arcseconds. The fast-search
edges are protected when merging the added proposals; half the remaining edge
capacity is reserved for novel curved-motion proposals, with unused capacity
available to the expanded linear search. The original fast-profile seed bank
(up to 10,000 seeds) is also protected within the larger seed budget. Counts
report both discarded expanded-linear edges and discarded curved proposals.

Proposal geometry uses daily Hermite interpolation of builtin Earth positions
and velocities, mean-precession site rotation, and UT1 approximately UTC. Solar
motion uses a second-order Taylor approximation. These are inexpensive candidate
generators, not the accepted orbital model. Every candidate still passes the
same full topocentric, light-time-corrected fit and unchanged residual limits.
The broad fitter adds starts at 0.01, 0.03, 300, and 1,000 AU, and radial velocity
offsets ±0.05 AU/day at selected distances. Final distances and velocities can
leave the starting grid; no prograde or bound-orbit constraint is imposed.

Independent ODE-generated controls verify recovery of a nearby curved-motion
association and retrograde, hyperbolic, and distant associations with 18-day
gaps that the fast graph excludes. These controls establish specific recoveries,
not population completeness. The finite grid, angular gates, neighbor caps,
seed budget, and two-body dynamics still exclude possibilities. Planetary
encounters, unsupported observers, rate-less arcs, and reused identifiers remain
limitations. Increasing compute does not by itself remove those limitations.

## Storage

The source MPC file remains untouched. The cache contains:

- `observations.npy`: memory-mapped MJD UTC times and integer-milliarcsecond RA/Dec.
- `tracklets.npy`: identifiers, station codes, observation offsets/counts,
  three-dimensional directions/rates, linear RMS, and duration.
- `catalogue.npz`: delta-coded observation times/angles plus identifiers,
  station codes and counts. Redundant motion summaries are reconstructed.
- `manifest.json`: source identity, checksums, format, row counts, quantization,
  byte counts, and build timing.

On the 2026-10-21 downloaded snapshot, the portable numerical archive is about
59 MB versus 132 MB for the MPC gzip. The memory-mapped working representation
is about 375 MB. These are decimal MB; RAM usage during ingestion is higher.
The compressed archive is an astrometric search representation: it does not
preserve magnitudes, bands, note flags, or the exact source formatting. Keep the
original download if those fields or unsupported records are needed.

Angles are rounded by at most **0.0005 arcsecond per coordinate**. The portable
archive stores observation time at the native MPC six-decimal-day precision
(one microday); restoring can change float64 rounding by less than a microsecond.
Noncanonical finer timestamps are rounded to a microday in the portable archive
(up to 0.0432 second); the working arrays retain their original float64 values.
No angular observation is replaced by its fitted tracklet line during orbit fitting.

To transfer a cache, copy `catalogue.npz` and `manifest.json`, then:

```sh
solve_itf restore --archive catalogue.npz --manifest manifest.json \
  --cache .cache/solve_itf/restored
```

The archive checksum is verified before expansion. Tracklets split when the same
identifier/station has an observation gap exceeding half a day. Exact duplicate
measurements within a key are removed; cross-identifier duplicates are also
removed before fitting. Anonymous observations remain separate singletons.

Every source row is accounted for. Invalid dates/coordinates and unsupported
satellite, radar, and roving-observer records are counted separately. They are
not silently assigned terrestrial coordinates. Unknown station coordinates
prevent a candidate fit and are counted. Singletons and zero-baseline tracklets
are retained in the cache but cannot seed this rate-based linker; neither can
tracklets spanning more than 0.6 day. There is no claim to solve these exclusions.

## Search and fitting

1. Fit each tracklet once in Cartesian unit-vector coordinates. This avoids an
   RA=0 seam and a polar tangent-chart singularity in the candidate search.
2. Group tracklets by UTC midpoint day. For each day-pair separation through
   `--max-gap-days` (default 14), extrapolate both sets toward the same midpoint
   and perform a spherical spatial-index search. The symmetric extrapolation
   cancels the leading constant angular acceleration term locally.
3. Retain a bounded sparse edge graph. The default angular gate is
   `30 + gap_days²` arcseconds. Up to eight neighbors per day are retained per
   source tracklet, with detected truncations counted. At most 300,000 edges
   and 10,000 ranked triple seeds are retained. These are efficiency limits,
   not certified bounds on omitted probability. Priority includes an uncertainty
   penalty so short, poorly measured rates do not crowd out informative arcs.
4. Batch observer geometry by station, then fit all retained triple seeds before
   extending accepted chains. This prevents an early large family from consuming
   the budget ahead of the seed list. Fit each candidate using all deduplicated observations, known terrestrial
   observer positions, UTC-to-TDB conversion, solar two-body dynamics, and
   iterated light time. Initial hypotheses include all positive real Gauss roots
   and 24 range/range-rate seeds (0.1–150 AU topocentric range; initial radial
   offsets -0.01, 0, +0.01 AU/day). These are starting points, not prior integration
   cells or a complete search domain; optimization can leave those ranges.
5. Use analytic derivatives of the universal Kepler equation and implicit light
   time instead of six finite-difference propagation jobs per Jacobian. The
   compiled optimizer solves its six-dimensional damped normal equations directly,
   avoiding thousands of Python and tiny BLAS/LAPACK calls. It keeps the full
   range/range-rate start grid and Gauss roots, with 80 optimization iterations
   per start. Retain distinct converged orbit modes. The default global RMS limit is 0.6
   arcsecond. Every constituent tracklet must meet the same RMS limit, and no
   individual residual can exceed `max(3*sigma, 2*rms_limit)`. No outliers are
   silently clipped. Perihelia below 0.005 AU and degenerate angular-momentum
   solutions are not accepted.
6. Predict nearby graph neighbors with every retained mode. First jointly fit
   all compatible additions for each mode, deduplicating identical proposals
   across parents. When a joint fit passes, skip individual extensions contained
   in it; when it fails, fall back to fitting individual additions. This avoids
   enumerating thousands of subsets of a valid long chain. Keep parents and
   alternatives, extending to four, five, six, or more tracklets while time remains.
   The full Gauss/range grid is always tried, plus at most `--max-parent-seeds`
   (default 16) warm starts from the strongest parents. At most four extension
   branches continue from a parent. Seed tracklets must be at least 0.3 day apart;
   subsequent additions may be on the same night or at another station. They
   still have to pass the prediction gate and full joint fit. Different identifiers
   are required, so reused-identifier associations remain excluded. Chain spans may exceed the
   pair window, but growth searches only the retained graph; it is not an
   unrestricted archive ephemeris sweep.

Joint growth is a bounded search policy, not an equivalence proof: a subset
could support an alternative orbit that a longer chain excludes. Those
subset-only modes can be missed. `--single-addition-growth` retains the slower
individual-addition policy, which can enumerate many contained subsets and
exhaust the time budget in dense families. The report counts joint proposals
and individual extensions covered by accepted joint fits. Increase `--max-seeds`
for a wider initial bank, particularly when reusing a prepared cache.

The slower SciPy optimizers remain available internally for validation through
`fit_chain(..., jacobian_backend="analytic")` and `"finite_difference"`.
The compiled optimizer is checked against those references on frozen real-data
candidates; numerical fitting is still a local, finite multistart search.

The propagation kernel supports elliptic, parabolic, and hyperbolic trajectories.
That is a statement about the forward model, **not** population completeness.
Moving terrestrial sites use Astropy/IERS offline rotations and approximate
builtin Earth/Sun ephemerides. Planetary perturbations, relativistic corrections,
catalogue astrometric debiasing, and survey-dependent correlated uncertainties
are not fitted. There are no implicit network requests during search.

## Interpreting probabilities and output

`report.json` retains all accepted candidates and every discovered mode;
`candidates.txt` gives the full numbered list. `maximal_candidates.txt` omits
chains whose tracklet sets are contained in longer accepted chains, keeping
their original ranks for cross-reference. This is also the default console view;
`--include-subsets` shows all chains. `--top` limits only console printing.
Containment is a presentation filter, not proof that a longer chain or its
orbit is correct. Partially overlapping alternatives remain visible in both
views. Text comparisons are marked `BIC score`; they are not probabilities.

With `m` tracklets, `n` observations, and a per-coordinate uncertainty `sigma`
(default 0.3 arcsecond), the score compares:

- one shared six-parameter orbit;
- independent four-parameter local straight-line fits for each tracklet.

The reported approximation is

```text
log_BF_BIC = (chi2_independent - chi2_orbit + (4*m - 6)*log(2*n)) / 2
p_BIC = logistic(log_BF_BIC + log(prior / (1-prior)))
```

`--prior-link-probability` defaults to 0.001. It is an explicit illustrative
per-proposed-chain prior, **not** a prior calibrated for the ITF or for this
selection procedure. Rankings use log odds to avoid numerical saturation of
probabilities near one. RMS is the square root of the mean squared two-coordinate
angular separation per observation, not the smaller per-coordinate RMS.
Inclination is relative to the mean J2000 ecliptic, not the equatorial plane.

BIC is an asymptotic approximation and may be poor for short, multimodal,
weakly identified orbital arcs. Each mode has Jacobian diagnostics and
`relative_mode_likelihood`, normalized from its peak Gaussian likelihood. These
mode weights are **not** posterior masses: no orbital-prior-volume or Hessian
integration has been performed. The reported link comparison uses the best mode.

Each mode's `local_uncertainty` reports formal range and inclination standard
errors from a column-scaled SVD of the astrometric Jacobian, plus the scaled
condition number and a diagnostic of whether orbital energy differs from zero
by more than three local standard errors. Errors use the configured astrometric
sigma; a very small observed RMS does not shrink them. Rank-deficient solutions
have null errors rather than misleading pseudoinverse precision. Text inclination
errors refer to the best mode only.

`[weak orbit]` means the best mode is rank deficient, its local range error exceeds
half its range, or its inclination error is undefined or exceeds 10 degrees.
These are explicit diagnostic thresholds, not calibrated confidence levels.
Local linear errors can themselves be unreliable in these cases. They do not
include other modes, planetary perturbations, or astrometric systematics. A small
formal error does not establish a unique orbit or a correct link, and an unresolved
energy sign means the nominal elliptic/hyperbolic label is uncertain locally.

Different objects can all be real, so candidate probabilities should not sum to
100%. Rows sharing tracklets are competing explanations. `overlapping_ranks`
identifies shared membership; `contains_ranks`, `contained_by_ranks`, and
`partially_overlapping_ranks` distinguish containment from partial overlap.
The program does not yet infer a globally exclusive catalogue. Supersets and
subsets are deliberately retained in the machine-readable output; they are not
separate discoveries. A low RMS or a high
BIC percentage alone does not validate an association.

`counts` and `timings` expose coverage, exclusions, neighbor truncations, seed
limits, attempted/rejected/interrupted fits, unprocessed seeds, successful
extensions, and timing. `full_catalogue_scan_complete` concerns the eligible
tracklet candidate scan only. `exhaustive`, `probabilities_calibrated`, and
`full_objective_complete` remain false even if `bounded_search_complete` is true.
Source code and dependency versions are recorded for reproducibility.

## Relationship to `fold_kepler` and `fit_sky`

The existing work was evaluated before choosing this implementation:

- [`fold_kepler`](fold-kepler.md) scores a supplied finite image-trajectory bank
  using FFT-filtered images and shared partial sums. Its recorded 64,000-orbit
  benchmark is 3.60× faster for repeated evaluations, but 0.50× as fast for
  first-use total time after preprocessing. ITF provides sparse detections,
  not the image response maps that this method consumes.
- The [global transform campaign](../research/kepler_transform/global_results/README.md)
  and later [detector-time work](../research/kepler_transform/detector_time/README.md)
  demonstrate useful exposure reuse on specified domains. They retain substantial
  orbital-resolution and construction costs, without a full-ITF linkage benchmark
  or a general sublinear orbit enumeration result.
- [`fit_sky`](fit-sky.md) provides explicit generative priors, alternative catalogue
  hypotheses, and uncertainty accounting. Its pixel likelihood and current
  integration/particle algorithms do not establish a five-minute whole-ITF
  solution. Copying its probability labels would not transfer its assumptions.

The useful concept here is shared computation: reduce each tracklet once and
reuse geometry/index work. A dense orbital FFT is not added without evidence
that construction plus querying beats the sparse baseline for this workload.
The new compiled propagator is independently checked against numerical ODE
integration, including light time, instead of assuming a new implementation is
correct because it follows the existing universal-variable approach.

## Checkpointed campaign stages

`solve_itf campaign --help` exposes graph, seed, fit, and export stages for larger
seed banks. Run each stage against the same catalogue and observatory snapshot;
retain the generated manifests and checkpoint databases. Defaults are bounded.
See each subcommand's `--help` for stage-specific resource and coverage controls.
This command runs local processes; it does not provision cloud machines.

## Long-arc recovery

Use a local JPL DE440/DE440s SPK and an existing report from the same catalogue:

```sh
solve_itf recover --cache .cache/solve_itf/catalogue \
  --report runs/solve-itf-first/report.json \
  --observatories .cache/solve_itf/ObsCodes.html \
  --ephemeris /path/to/de440s.bsp \
  --output runs/solve-itf-recovery --seconds 300 --workers 4 --backend compiled
```

DE440s is available from the [JPL NAIF archive](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp).
The SPK is not bundled or downloaded implicitly. The recovery model includes the
Sun, eight planets, Moon and Pluto; it omits asteroid perturbers, relativity,
non-gravitational forces and systematic astrometric errors. States are
heliocentric equatorial ICRF, in AU and AU/day, with MJD TDB epochs. Observation
timestamps in the catalogue are MJD UTC. Report inclination is ecliptic.

Recovery propagates local covariance to retrieve possible earlier/later
tracklets, then jointly refits their original observations. The compiled backend
has a SciPy reference (`--backend scipy`). Local covariance is conditional on the
noise/force model, not calibrated coverage or exploration of all orbital branches.
Cone sizes, seed selection, candidate caps and budgets limit retrieval.

For shorter gaps, use `--min-gap-days 1 --max-gap-days 120 --link-arcs`.
`--max-gap-days 0` spans the full available date range. Add `--staged --iterations 3`
to revisit bounded children through short, seasonal and longer windows.
Repeat the same command/output directory to resume a compatible checkpoint;
changed settings, code or sources require a fresh checkpoint. Failure, deferred
and unattempted counters are not successful coverage or physical rejection.
`solve_itf replay-growth --help` describes replaying deferred fitting work.

`python research/solve_itf/make_recovery_bank.py --help` merges preceding reports
while preserving alternative modes and rejecting mixed catalogue snapshots.

## Verification and scientific limits

Install `pip install -e '.[itf,test]'`, then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_solve_itf*.py
```

The shipped tests cover MPC parsing/cache round trips, independent ODE dynamics,
light time and analytic derivatives, synthetic linkage and extension, all-conic
starts, graph limits, covariance/screening parity, checkpoint identity, and
alternative-mode preservation. Tests requiring a real JPL kernel are optional
and report skips when their documented input is absent.

Development experiments found mixed-catalogue chains that pass low RMS cuts.
A good numerical fit is therefore not sufficient evidence of a correct physical
association. No 99% reliability, exhaustive search, calibrated BIC probability,
or confirmed-discovery claim follows from running this software. Prior dashboards,
cloud fleet controllers, private observations, and experimental certification
pipelines are not dependencies of this linker release.

## Measured stage performance

The [performance note and original timing receipt](../research/solve_itf/README.md)
break down the historical 184-second, four-worker, 10,000-seed full-catalogue
run. These are historical measurements, not guarantees for this release.
