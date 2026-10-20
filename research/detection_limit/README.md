# Full-objective `prove_floor` research attempts

**The full objective is not achieved. No breakthrough is claimed.** This work
implements and tests three routes toward matching detection limits and
detectors. It closes the requested gap in explicit continuous controls, but
fails on the broad continuous orbital families tested here.

The [acceptance criteria](objective.md) were committed before experiments. The
[derivation](derivation.md) separates established theory, interval arithmetic,
Monte Carlo confidence, physical assumptions and unresolved requirements.

## What is implemented

- A primal/dual detection certificate: an upper bound for every admissible
  detector and a lower bound achieved by an executable randomized detector.
- Alternative-distribution importance sampling using bounded quantities, so
  calibration does not depend on seeing extremely rare noise-only false alarms.
  It works with Gaussian and Poisson controls and handles discrete LR ties.
- Continuous Gaussian source-family certificates using justified mean-error
  balls, with separate prior-average and worst-case objectives.
- Candidate least-favorable mixture optimization, independently validated
  rather than treated as a proof because the optimizer stopped.
- A brightness-limit bracket with an explicit confidence budget across adaptive
  evaluations. An ambiguous power result never silently removes half the range.
- Arb-enclosed, all-conic two-body propagation of six-dimensional state boxes,
  observer-dependent light time, gnomonic coordinates and pixel-integrated PSFs.
- A geometric matched-filter route, including an exact convex control and a
  counterexample showing why convexifying orbit alternatives can fail.

These are research APIs, not a replacement for the existing production
`magictools prove_floor` command. No real-survey discovery/exclusion is claimed.

## Reproduce

From a repository checkout:

```bash
python -m pip install -e '.[test,research]'
OPENBLAS_NUM_THREADS=1 python -m research.detection_limit.experiments new-results.json
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_detection_limit_research.py
```

The experiment refuses to overwrite its output. It runs offline, records
software versions and source hashes, and includes construction time. The full
run takes a few minutes on the development machine; timing is not a portable
speed claim. NumPy/SciPy and `python-flint` are required for this research;
production runtime dependencies are unchanged.

[`first_results.json`](first_results.json) preserves the pilot, including its
failed one-percentage-point targets. The larger run in [`results.json`](results.json)
uses a fresh, fixed validation seed, chosen without searching for favorable
results. It increases simulation budgets and tests up to 4,096 physical orbit
cells. Individual stochastic certificates have 99% confidence; this is **not**
a 99% joint claim for every report in the file. The adaptive brightness bracket
has its own simultaneous confidence allocation.

## Results

For power, a gap of 0.01 means **one percentage point**, not 1% relative error.
All numbers below are conditional on the explicitly specified models.

| Experiment | Detector lower bound — universal upper bound | Meaning |
| --- | --- | --- |
| Known Gaussian source, false-alarm probability 0.01 | 0.74780 — 0.75056 | Contains the independently known optimal power; gap 0.00276. |
| Known Gaussian source, false-alarm probability 1e-7, mean norm 6 | 0.78747 — 0.78913 | Rare-false-alarm control without waiting for null-tail events. |
| Continuous circle of Gaussian means | 0.55008 — 0.55427 | Contains the independent exact answer, 0.5533312682248218. A classical symmetry example. |
| Known Poisson source, false-alarm probability 0.01 | 0.40410 — 0.40670 | Boundary randomization handles discrete observations. |
| Continuous neighborhoods of two unequal-strength signals, minimax | 0.43078 — 0.43605 | Optimized weights close the worst-case gap; uniform weights leave a gap of about 0.142. |
| 16 separated continuous Kepler orbit boxes | 0.52244 — 0.52658 | Within 0.00415 of optimal prior-average power, including the covering error. |
| Broad bound-orbit box, 4,096 cells | 0.01 — 1.00 | Failed: the certified covering error remains too large. |

For that same narrow orbital prior, the flux required for **90% average
detection power** is bracketed between **8.25 and 8.34375 reference counts** at
a 1% false-alarm probability, with 99% simultaneous model-conditional confidence.
The bracket is about 1.12% wide relative to its upper endpoint. These are
synthetic input units with unit noise standard deviation, not a limiting
astronomical magnitude or a real-survey depth.

The orbital success has an important limitation: alternatives are separated by
two arcseconds, but each mode has only about **0.000206 arcsecond** transverse
half-width. Every mode has nonzero uncertainty in all six initial coordinates,
yet this is still a very narrow multimodal prior, not a blind search over the
connecting orbital domain. There are four synthetic exposures with known
independent Gaussian noise and a one-pixel-sigma PSF. Instantaneous exposure
times, fixed observer vectors and a point-mass Sun define the tested model.

### Route A: a useful certificate, unsuccessful broad coverage

The detector and dual bound work once the continuous mixture is approximated
closely enough. Direct interval propagation establishes what each cell covers,
including light time; no gap between sampled orbits is assumed away.

But the broad bound-family mean TV error decreases only from approximately 1.0
at one cell to 0.928 at 64, 0.803 at 256, and **0.602 at 4,096**. Construction of
the last cover took about 87 seconds for four exposures. Increasing simulation
accuracy cannot repair that deterministic covering error. Other tested families
include near-parabolic, unbound and close-observer states; none meets the full
power-gap target at the tested budgets.

This is a failure of the proposed cellwise certificate at these budgets, not a
proof that those sources are undetectable or that every continuous method fails.

The brightness-bracket attempt exposed another failure: alternative importance
sampling can calibrate the null poorly when the source is extremely easy to
detect. The saved bright-source probe compares it with an independent null
order-statistic calibration, which repairs this case. The latter needs enough
null draws and cannot replace the importance approach at arbitrarily small
false-alarm probabilities. Both methods retain explicit confidence budgets.

### Route B: minimize worst-case loss, then check it

On a two-component test, a uniform-prior detector has a minimax gap of about
0.142. Optimized weights near (0.877, 0.123) emphasize the weaker component and
reduce the validated gap below 0.005 for the point family and below 0.006 for
continuous mean balls. The validation checks each cell separately. An average
over the optimized prior is not substituted for a worst-case guarantee.

This uses established least-favorable-distribution theory. Scaling both the
continuous coverage and all-cell power validation remains unresolved.

### Route C: geometry alone does not solve nonconvex families

For means (3,t), -2<=t<=2, a separating matched filter and a physical hardest
member give the same power, about 0.74973. For the two means (+3,0), (-3,0), the
convex hull includes zero and the geometric lower bound collapses to 0.01,
although the exact two-sided detector has power about 0.66428. Thus replacing a
nonconvex orbital family with its convex hull can lose the relevant information.

## Using the certificate engine

```python
import numpy as np
from research.detection_limit.certificates import (
    GaussianCover, certify, flux_bracket, rejection_probability,
)

# This cover is an INPUT mathematical assertion, not something this class proves.
# Every physical mean in cell i must lie within radii[i] in whitened pixel norm.
cover = GaussianCover(
    means=np.array([[2.2, 0.0], [0.0, 2.9]]),
    radii=np.array([0.001, 0.001]),
    weights=np.array([0.5, 0.5]),
)
report = certify(cover, samples=1_048_576, alpha=0.01)
probability = rejection_probability([1.4, 0.2], cover.means, report)
# The rule rejects with this probability; randomization is part of the detector.

# If these are unit-flux means and the noise stays fixed as flux changes:
floor = flux_bracket(cover, maximum_flux=4.0, target_power=0.9)
```

`orbit_cover()` supplies such balls for the implemented physical model. A
caller-supplied radius without a proof/enclosure is not a continuous certificate.
`certify_likelihood()` supports another supplied distribution only if its
callback genuinely samples that alternative and evaluates its likelihood ratio;
the interface is not a general continuous-integration solver.

For the saved orbital detector, multiply the whitened pixel vector by the
transpose of `compression.pixel_to_score_basis`, then call
`rejection_probability()` with `score_means` and that report. All input exposure
parameters, orbit boxes, means and the projection are saved.

## What is still required for the full objective

1. Tight, affordable **whole-mixture** bounds over broadly uncertain continuous
   orbit families, or a substantially better uniform certificate. The present
   cellwise TV bounds ignore cancellations when averaging a smooth population.
2. A scalable minimax calculation over the actual nonconvex source family,
   including any specified nuisance parameters.
3. Validated instrument response/noise inputs, and extensions beyond the
   implemented instantaneous Gaussian-PSF model. The Poisson control validates
   the statistical interface, not a Poisson orbital-image pipeline.
4. Complete numerical certification: Arb encloses the physical forward model,
   while score evaluation, statistical quantiles and SVD remain floating-point.
5. Independent mathematical/astronomical review and a substantive difference
   from existing methods before any breakthrough claim.

The full objective remains unmet. The strongest next mathematical target is a
tight bound on the continuous population mixture itself, rather than requiring
every individual orbital distribution to be almost identical to its cell's
representative.
