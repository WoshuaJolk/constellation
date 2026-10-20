# Fold Kepler: continuous observer-chart campaign

**The declared near, distant and widest stress domains now have complete
continuous geometry covers at 0.0005 arcsec per tangent coordinate. The full
all-orbit transform objective is still open.** This is a functional bounded-domain
research implementation, not a claimed mathematical breakthrough or discovery.

See [the fixed objectives](../chart_objective.md) and
[derivation, model and command](../chart_construction.md).

## Status against the catalogue objective

| Requirement | Current evidence | Missing for completion |
|---|---|---|
| Stack of exposures with astrometry and times | Finite Gaussian detector adapter, checked on real ZTF cutouts | General validated instrument likelihoods, realistic full-image scale |
| Summed likelihood over every Keplerian trajectory at once | Continuous geometry on explicit six-dimensional boxes; scores at supplied queries | An economical global likelihood representation/synthesis across the full requested orbit domain |
| Well below one evaluation per trajectory | Shared 3D transforms outperform compiled direct scoring on two structured workloads | Broad benefit for arbitrary six-dimensional searches, including construction; widest and small real workloads currently lose |
| Transform and inverse as a library `dig_floor` can call | Importable research API, executable command, finite-query adjoint and regularized inverse | Production `dig_floor` integration and the general transform/inverse; the finite inverse does not imply unique image recovery |
| Whole-archive all-orbit stacking affordable | No archive-scale demonstration | End-to-end archive measurements, recovery validation, and bounded total resource costs |

**Continuous geometry coverage is not the same achievement as an all-orbit
likelihood transform.** The central mathematical/computational requirement
remains unresolved. No comparison against KBMOD or an existing ITF linker
has been performed; compiled direct scoring is our measured reference.

## Measured speed and its scope

[Compiled comparison](compiled-comparison.json): three rotated-order trials,
single-thread FINUFFT and BLAS, serial Numba without fast math. The direct
baseline starts at the chart's emission epoch, avoiding an unnecessary extra
propagation. Both methods evaluate all the same queries. The comparison using
the same polynomial geometry also shares repeated outer-coordinate work.

| Domain | Queries per trial | Exposures / span | Compiled direct | Fast application | Geometry construction + certification | Fast application + geometry |
|---|---:|---:|---:|---:|---:|---:|
| Near | 2,097,152 | 8 / 8 days | 18.80 s | 5.37 s | 5.65 s | 11.02 s |
| Distant | 2,097,152 | 32 / 30 days | 81.09 s | 5.75 s | 33.46 s | 39.21 s |
| Widest, quartic | 7,340,032 | 8 / 8 days | 73.54 s | 21.30 s | 68.06 s | 89.36 s |

These are application medians plus the measured original construction cost.
The first two are **1.71× and 2.07×** faster including geometry; the widest
case **loses**, at 0.82×. Against the optimized same-geometry comparison the
ratios are 1.67×, 1.78× and 1.28×. The earlier Python-only comparison is retained
as history, not used as the headline result.

Phase-order selection is cached during application trials. Its separately
measured cost and totals charging that cost appear in
[verification.json](verification.json), under `planning_supplement`.
Including that planning, the totals are **11.03 s, 39.23 s and 89.41 s**;
the corresponding ratios remain 1.71×, 2.07× and 0.82×.
The table is not a cold end-to-end command benchmark. Input I/O, imports,
image retrieval/preparation and selected slow Arb audits are excluded.
The direct baseline's first-call compilation (1.24 s) is reported separately
and excluded from the ratios above. Geometry construction includes failed
parent cells and their certification attempts.

Queries deliberately share eight outer-coordinate points per certified cell,
with 131,072 inner queries per outer point. **These are not millions of
independent arbitrary six-dimensional outer points.** Arbitrary queries have
a working polynomial/per-exposure route, but no demonstrated global fast
outer-coordinate synthesis. The inputs here are synthetic finite Fourier fields,
not full detector images. Small validation jobs overlapped parts of the campaign
on the shared host; these measurements are not hardware performance guarantees.

The phase policy selects Q=0 at a declared analytic operator tolerance of
1e-4. Maximum sampled score differences from compiled direct evaluation are
2.88e-5, 4.18e-5 and 1.48e-5. These are absolute differences for the recorded
input normalization, not relative likelihood errors. Continuous geometry and
analytic truncation bounds are separate from sampled numerical agreement.
**The entire fast floating-point/NUFFT score is not interval-certified.**

## Three attempts on the same widest domain

The domains and tolerance were not shrunk after failures:

1. [Quadratic construction](quadratic/stress-cover.json): zero resolved leaves
   at the 16-leaf budget. All unresolved cells remain recorded.
2. [Quartic construction](quartic/cover-16.json): all seven leaves resolved;
   68.06 seconds. Independently random six-dimensional validation and selected
   full-Arb score audits are retained.
3. [Observer-relative interval coordinates](translated/complete.json): all five
   leaves resolved; 48.29 seconds. The physical model and accuracy requirement
   are unchanged. This is a proof-cost experiment; the large application
   benchmark above uses the seven-cell version. No five-cell speedup is inferred.

The single unsplit quartic cell has a sampled position error of 0.00147 arcsec,
above the 0.0005 arcsec target. Thus the unsplit polynomial itself fails;
interval-bound tightening alone does not solve that particular approximation.

## Real pixels

[Three real ZTF exposures](real/result.json), spanning 12.3 minutes, are
archived as three 9×9 cutouts with masks, estimated variances, survey PSFs,
WCS, provenance and observer states in [real_inputs](real_inputs/observations.json).
They are public survey inputs retained from the earlier campaign, **not image
data from the user's web repository**.

For 16,384 candidates, the finite-detector adapter scores both numerator and
position-dependent information, with independent profiled constant backgrounds
and calibrated relative flux scales. It uses pixel-integrated Gaussian PSFs;
the empirical survey PSFs are preserved but not used in this approximation.

* Maximum SNR difference against independent direct pixels with the same
  affine model: **5.54e-8**.
* Maximum SNR difference using the actual TPV WCS in direct evaluation:
  **3.99e-7**.
* Both comparisons select the same best query.
* Total fast path: **8.41 s**, versus **0.84 s** for direct physical geometry
  and pixel scoring. This small workload is about **10× slower**.

This establishes agreement under an explicit model, not complete real-survey
calibration. Exposure trailing, spatial PSF variation, exact general WCS,
correlated-noise inference, false-positive calibration and recovery completeness
remain open. No new-object claim is made.

## ITF example: what these timings do and do not mean

The MPC's [Isolated Tracklet File](https://docs.minorplanetcenter.net/mpc-ops-docs/identifications/)
contains unlinked astrometric tracklets. A possible application is to use a
tracklet to define candidate distance/velocity ranges, retrieve surrounding
images, and score those trajectories for additional evidence before fitting
and validating a linkage.

The distant benchmark shows an **81-second scoring stage becoming 39 seconds
including geometry**, or about **6 seconds with that geometry already built**.
This is a measured synthetic scoring example, not a measured ITF workflow.
Geometry reuse requires compatible exposure times, observers, tangent bases
and chart domain; phase-policy reuse also requires compatible frequencies.
We have not compared this implementation against established ITF linkers,
measured whole-file recovery, or established an archive-wide runtime.

## Run and verify

Install from the repository:

```sh
pip install '.[research,accelerate]'
constellation fold-chart research/kepler_transform/chart_results/demo/manifest.json /tmp/new-chart-output
```

The [demo](demo/manifest.json) contains prepared Fourier inputs and 256 queries.
The command preserves coverage reports, certificates, all scores and selected
independent audits. Unresolved queries are refused unless explicit direct
fallback is enabled. `constellation fold_kepler` retains its existing production
interface; this experimental command is separate.

From a research checkout:

```sh
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.verify_chart_campaign --output /tmp/chart-verification.json
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.compiled_comparison research/kepler_transform/chart_results/quadratic research/kepler_transform/chart_results/quartic /tmp/new-comparison.json
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.translated_experiment /tmp/new-translated
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.real_exposure_experiment --observations research/kepler_transform/chart_results/real_inputs/observations.json /tmp/new-real
```

The proof replay checks exact partition volume/disjointness, outward physical
cells, polynomial/input bindings and interval bounds. Historical proofs use
their archived interval expression trees: an algebraic refactor can widen an
interval without changing the physical function. Replaying the old bounds with
the latest expression trees was attempted and rejected a weaker bound; this
was not treated as successful verification.

[quadratic/source_snapshot](quadratic/source_snapshot/snapshot.json) and
[compiled_sources](compiled_sources/snapshot.json) preserve executed module
versions. Original end-of-run checkout hashes are retained separately where
later work changed files while a process was running. The quartic certificate
engine is preserved in `compiled_sources`; its original report also includes
hashes of intermediate application/CLI/replay files not preserved as a full
checkout. Timing reruns use current code and may differ. The source snapshots
are research artifacts, not separate installed package versions.

The tests cover independent high-precision derivatives, all-conic nodal
enclosures, failure paths, off-grid geometry, full-Arb selected score audits,
forward/adjoint consistency, regularized inversion with known nullspaces, finite
detector edges/backgrounds, and installed-command inputs.
The full repository suite passed **294 tests**. The wheel was installed in a
separate environment and its actual command evaluated all 256 demo queries,
including two independent score audits. See [validation.json](validation.json)
and [the installed demo report](demo/installed_output/report.json).

Remaining full-objective requirements include economical general outer-coordinate
synthesis, complete fast numerical error control, and accurate efficient
large real-image workloads. Completion percentages are not inferred from the
number of implemented components.
