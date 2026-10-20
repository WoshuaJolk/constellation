# FitSky research interface

**The full catalogue objective remains open.** This interface exposes the
research implementation and its conditional guarantees, not an all-sky service.
See the [fixed contract](../research/fit_sky/OBJECTIVE.md),
[derivation](../research/fit_sky/derivation.md), and
[campaign report](../research/fit_sky/README.md).

```sh
pip install '.[research]'
magictools-fit-sky examples/fit-sky.json /tmp/new-fit-sky \
  --verify --samples 100 --statistical-samples 100000
```

The example is an independently integrable synthetic flux/count control. Its
scope is printed in the manifest. Output directories must be new.

For an **approximate, resumable** variable-count posterior across exposures:

```sh
magictools-fit-sky-stream input.json /tmp/fit-sky-first \
  --particles 2048 --sweeps 20 --gibbs-flux --split-merge \
  --stop-after 10 --catalogues
magictools-fit-sky-stream input-with-more-exposures.json /tmp/fit-sky-next \
  --resume /tmp/fit-sky-first/checkpoint.npz --catalogues
```

This separate command does not replace certified integration. It reports
`approximate_uncertified`, keeps the accuracy target unmet, and never presents
particle agreement or ESS as a confidence interval. Its point estimates can
still miss posterior modes. It supports star/galaxy populations, motion,
parallax, variability and shared instrument uncertainty, with unbounded Poisson
counts if specified. Kepler propagation is not implemented in this renderer.

`--exposure-batch` defaults to one new exposure per assimilation; all previous
images remain in the joint likelihood. `--pixel-block` bounds rendering
workspace, not the size of the retained input dataset. Shared Gaussian linear
nuisance is integrated jointly. Arbitrary dense pixel covariance is rejected.
The algorithm currently reevaluates old images during catalogue changes, so
bounded memory is not a claim of linear total runtime.

`--gibbs-flux` adds conditional Gaussian brightness updates retaining other
sources and shared covariance. `--split-merge` adds joint changes between a
source and two nearby sources of the same class, conserving flux and centroid.
These are optional mixing improvements with independent mathematical controls;
they provide no guarantee that a finite population finds every mode.

`checkpoint.npz` saves completed assimilation and the RNG state; `report.json`
contains approximate evidence/query estimates, count and calibration summaries,
stage diagnostics and optional weighted catalogues. Resume accepts appended
observations while requiring existing data, priors and instrument definitions
to remain unchanged. Duplicate exposure IDs are rejected. Conflicting resume
settings require a fresh run. A failed assimilation writes `failure.json` and
retains the last completed checkpoint.

See the [sequential derivation](../research/fit_sky/sequential_derivation.md)
and [fixed 30-exposure campaign](../research/fit_sky/SEQUENTIAL_CAMPAIGN.md).

The certified `magictools-fit-sky` command writes:

- `report.json`: model, prior, requested queries, global evidence/posterior
  bounds, full partition certificate, work, and unresolved status.
- `verification.json`, with `--verify`: partition and enclosure replay.
- `samples.json`, with `--samples`: variable-length catalogue draws, conventional
  RNG scope, completion of the requested draw count, and omitted-count mass bound.
- `statistical.json`, with `--statistical-samples`: fixed-budget simultaneous
  confidence bounds, reported separately from deterministic enclosures.

`--method range`, `gauss`, `hybrid`, or `auto` selects interval range bounds,
validated tensor Gauss quadrature, analytic integration of one uniform flux,
or intersection of the latter two routes. Defaults to `auto`.
`--max-leaves`, `--max-seconds` and `--tolerance` set explicit budgets and accuracy.
Limits are soft boundaries between cell evaluations. An unfinished run retains
all prior mass and returns `unresolved`. A `bounded_target_met` result applies
only to that declared model/domain; `full_objective_complete` remains false.

`--statistical-prior` samples the complete declared prior instead of the
deterministic cover, including unbounded Poisson source counts. This can avoid
a count-truncation error while still failing badly from high variance. It is
not permission to interpret unbounded confidence intervals as convergence.

For Gaussian star/galaxy models, `--noise-floor both` constructs and replays
global and conditional continuous-template support bounds before statistical
integration or catalogue sampling. It needs samples and a partition proposal.
`noise-floor.json` and `conditional-noise-floor.json` contain the additional
pointwise certificates. These preserve the original priors and covariance;
they bound the residual that no allowed catalogue can remove.
Use `global` or `conditional` instead of `both` to select one construction.
This route does not implement continuous orbital support bounds.

`--method noise-floor` instead constructs the original partition adaptively
using those pointwise bounds. Its certificates contain the proposed dual
vectors, so replay never needs to rerun an optimizer. Statistical integration
and catalogue sampling replay these embedded proofs before using them.
An unresolved deterministic integration report can still be a useful proposal
for a separately bounded statistical calculation.

`--marginalize-flux` analytically integrates one uniform flux per Gaussian
statistical draw. Other sources and shared covariance remain in the integral.
Queries involving that flux use a truncated integral and correctly include
other sources that already satisfy an `any:` event. This also works with the
complete-prior sampling route; Poisson likelihoods retain ordinary sampling.

`--fast-samples` selects vectorized float64 rejection sampling with
`--samples`; the default uses Arb acceptance comparisons. Both preserve the
full represented prior partition. `--sample-proposals` controls the proposal
budget, and the output reports whether all requested catalogues were obtained.
The float route explicitly assumes numerically accurate forward evaluation.
It does not claim an arithmetic certificate of the random draws.

## Manifest

The outer JSON has `scene` and optional `queries`. The scene contains:

| Field | Meaning |
|---|---|
| `source_types` | Named source classes, positive mixture weights, proper parameter boxes |
| `count_prior.probabilities` | Nonnegative weights for K=0,1,..., normalized once |
| `count_prior.poisson_mean`, `max_count` | Poisson prior and computational enumeration cutoff; remaining mass retained |
| `exposures` | Explicit unique IDs, instrument, time, pixel data, response and noise |
| `nuisance` | Shared instrument parameter boxes |
| `noise` | `gaussian` (default) or `poisson` |
| `covariance` | Optional complete Gaussian covariance over usable pixels |
| `linear_nuisance` | Proper Gaussian prior for shared linear offsets/calibrations |

Every parameter is `[lower, upper]`; equal bounds are fixed values. Parameters
within a class and nonlinear nuisance boxes are independent uniforms.
Class labels have prior probabilities proportional to `weight`, default one.
Source parameters are IID conditional on class and count; source labels are
exchangeable. These are actual priors, not numerical search bounds detached from
an unspecified population model.

Source kinds and parameters:

- `star`: `x`, `y`, `flux`; optional `vx`, `vy`, `parallax`, `variability`, `color`.
- `galaxy`: the same, plus intrinsic detector-axis `sigma_x`, `sigma_y`.
  This is an axis-aligned Gaussian extended-source model.
- `kepler`: `rx`, `ry`, `rz` in AU and `vx`, `vy`, `vz` in AU/day at the declared
  zero epoch; `flux`, optional `variability`, `color`. Uses the existing
  all-conic two-body interval propagation and observer-dependent light time.

For stars, positions/rates are reference tangent-plane pixels and pixels/day;
the exposure affine mapping converts them to detector pixels. Parallax factors
are explicit per-exposure `parallax_x/y` coefficients in compatible units.
Flux is multiplied by
`flux_scale * exp(color * band_coordinate + variability * time)`.
This is not a general stellar light-curve or SED model.

An exposure requires `id`, `time`, a finite 2D `data` array, and positive
`psf_sigma`. Optional `noise_sigma` is scalar or per pixel; `mask=true` excludes
a pixel. Pixel centers are relative to the geometric image center.
`affine` defaults to identity, `offset` to zero, `flux_scale` to one, and
`instrument` to `default`. `background` is a known per-pixel offset unless
superseded by a nuisance parameter.

Kepler exposures additionally specify `observer`, tangent `center`, `east`,
`north`, `pixel_scale_arcsec`, and optionally `light_time_max` in days.
Reference frames/units must be consistent. No ephemeris fitting or automatic
instrument ingestion is inferred from the input.

Shared nuisance keys are `<instrument>_dx`, `_dy`, `_log_gain`, `_log_psf`,
and `_background`. The same parameter affects every exposure of that instrument.
Unknown/unused nuisance and source parameter names are rejected.

Gaussian `covariance` is indexed by unmasked pixels in exposure order, then
row-major order, and must be exactly symmetric positive definite. It supersedes
the diagonal variance model. `linear_nuisance` contains a pixel-by-parameter
`design`, prior `mean`, and symmetric positive-definite `covariance`; these
variables are marginalized jointly, with the determinant normalization retained.
With diagonal measurement noise, a low-rank calculation avoids a dense pixel
covariance. Poisson inputs cannot use this Gaussian nuisance option.

The instrument model currently omits exposure trailing, general WCS distortion,
arbitrary spatial PSFs, blends with unmodeled morphology, empirical artifact
populations and unknown noise laws. The archived ZTF demonstration is conditional
on a Gaussian approximation; its original survey PSFs are not replaced by a
claim that this approximation is calibrated.

## Posterior queries and draws

Queries are permutation-invariant:

```json
["count=0", "has:moving", "any:moving:vx:0:0.6"]
```

The last event is that at least one source of class `moving` has its `vx`
parameter in the closed interval [0,0.6]. Cells crossing the event boundary
retain their ambiguous mass until refined. Posterior draws contain all represented
sources and uniform-box nuisance parameters. Linear Gaussian nuisance is
marginalized; its posterior draws are not included.

```python
from research.fit_sky.integrate import infer, verify
from research.fit_sky.sampling import sample

report = infer(scene, method="auto", queries=["count=0"], max_seconds=120)
verification = verify(report)
catalogues = sample(report, count=100)
```

`research.fit_sky.graph.infer` exposes finite binary-dictionary sum-product
inference separately. It retains source interactions and supports full known
pixel covariance, but its finite dictionary and float64 arithmetic must not be
confused with the continuous Arb certificates.
