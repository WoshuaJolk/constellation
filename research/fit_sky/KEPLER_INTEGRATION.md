# Kepler propagation inside FitSky's joint pixel inference

The sequential engine can now infer orbital sources alongside stars and
galaxies. It previously rejected every `kepler` class. The integration reuses
Kepler's all-conic universal-variable propagator and predicts each proposed
source's pixel contribution at each exposure time and instrument response.
It does not require detections or link an input detection catalogue.

Neither project's full objective is complete. This is a forward-model
integration, not a fast all-orbit transform, an inverse transform, a new
propagation theorem, or a certified posterior solver.

## Existing models and reusable findings

| Component | Detected catalogue required? | Role in joint pixel inference |
|---|---|---|
| FitSky `scene.py` + `interval_orbits.py` | No | Existing orbit-aware interval renderer, including light time; supports bounded integration but not arbitrary orbital directional jets or the star/galaxy noise-floor bounds. Preserved unchanged. |
| Kepler `universal.propagate` | No | Reused directly for numerical elliptic, parabolic and hyperbolic Cartesian states. Established universal-variable algorithm. |
| Kepler interval geometry/domain covers | No | Potential tools to bound responses over continuous orbit cells; not yet a complete FitSky orbital posterior bound. |
| PFKT pixel/Fourier likelihood fields | No | Potential conditional-likelihood accelerator when data, response and relevant nuisance are fixed. Current guarantees do not cover this general changing multi-instrument model. |
| Sparse detection-catalogue stacking and tracklet linking | Yes | Can suggest proposals, but discards subthreshold pixel information unless a corresponding observation/selection model is included. Not a replacement likelihood here. |
| FitSky finite template interaction graph | No detected catalogue inherently | Exact algebra over its declared finite dictionary; it is a different prior if used to replace continuous sources without bounds on the omitted domain. |

PFKT's useful lesson is to separate orbit geometry, image response, and
likelihood statistics. The integration follows that separation without
assuming that individually scored orbits are independent sources.

For a conditional Gaussian source update, let `r` be the image residual after
all other sources, `h(theta)` its unit-flux pixel response, and `C` the joint
noise/calibration covariance. The relevant statistics are

```
N(theta) = h(theta)^T C^-1 r
I(theta) = h(theta)^T C^-1 h(theta)
log L(f,theta) = constant + f N(theta) - f^2 I(theta)/2.
```

This is the natural location for a future PFKT adapter. Other sources change
`r`; gain, PSF and astrometric calibration change `h`; covariance changes must
also be respected. Ignoring these dependencies loses source interactions.
The existing sequential conditional-flux update already retains them through
joint Woodbury statistics and now works for orbital sources. We have **not**
substituted an approximate transform for its exact numerical pixel sums.

## Implementation and assumptions

- `orbital.py` calls the unchanged Kepler universal solver with FitSky's explicit
  solar constant, then iterates light time using FitSky's light-speed constant.
  This avoids a small but real mismatch with the propagator's default constants.
- State coordinates are `rx, ry, rz` in AU and `vx, vy, vz` in AU/day, in one
  heliocentric inertial frame at the zero epoch. Exposure times are TDB-day
  offsets. Observer positions must be in that frame at reception, including
  site offsets when needed. The caller supplies them; there is no ephemeris fit.
- Exposure tangent bases must be finite, right-handed and orthonormal. The
  tangent projection and pixel scale precede each instrument's affine mapping,
  shared position offset, gain, PSF and background response.
- The renderer sums all active sources before evaluating the Gaussian or
  Poisson likelihood. Masks, colour/variability, joint linear Gaussian nuisance,
  continuous orbit coordinates, count/class changes, flux updates and
  split/merge remain supported. For orbits the split/merge coordinate centroid
  is an initial-state centroid, not a conserved apparent-image centroid.
- Geometry is cached only within an evaluation/update, for the latest exposure
  of each source slot. Changed orbit populations start a fresh cache. No
  trajectory grid or data-selected orbit shortlist is introduced.
- A solver failure, too-small observer distance, excessive light time or
  tangent-horizon violation raises an error. It is never converted into zero
  likelihood. A failed assimilation retains the last completed checkpoint.
- New checkpoints are v2. Loading legacy v1 appends unused auxiliary
  coordinates, preserving physical sources/weights/evidence exactly. Subsequent
  random draws differ from v1 because proposals consume more coordinates.

The physical model still omits perturbations, exposure trailing, aberration,
general distorted WCS and arbitrary spatial PSFs. The universal solver is
float64 and has numerical domain guards; it is not an interval certificate.
An orbit prior must remain within supported geometry or the run stops.

## Independent validation

The tests compare the universal-variable predictions against DOP853 integration
of Newton's equations and Brent light-time roots. Elliptic, parabolic and
hyperbolic trajectories are checked at -30, 0 and +40 days, with a tolerance
of `2e-7` detector pixels at 0.2 arcsec/pixel. A separate check agrees with the
existing interval pixel model to `1e-12` in pixel contribution.

Mixed orbital/star scenes with continuous orbital coordinates, two overlapping
sources, two instruments and shared covariance match independent pixel rendering
and a dense normalized Gaussian likelihood within `2e-9` in log likelihood.
Additional tests cover masked Poisson pixels, source/exposure permutations,
cache invalidation, continuous orbit/flux updates, split/merge, invalid geometry,
v2 checkpoint reproducibility and legacy checkpoint migration. All **103 FitSky
tests passed** (the full repository suite was not rerun). The stream CLI also
completed a two-exposure checkpoint and resumed through all four exposures.
Loading the actual archived 10-exposure pilot checkpoint preserved its physical
particles, weights and evidence without modifying that archive.

Two retained campaigns use four synthetic 5x7 exposures over **24 days** and
two instrument responses. They are not real-survey validations or repeats of
Kepler's three ZTF cutouts. Inputs and fixed seeds are retained in
[`orbital_results/`](orbital_results/). Reference calculations use independent
ODE rendering, complete ordered count/class enumeration, dense covariance and
quadrature of shared calibration. No data-based pruning is used.

| Control | Independent posterior | Sequential comparison |
|---|---|---|
| Two fixed competing orbital classes; K=0/1/2; two continuous shared offsets | P(K=1)=0.362839; P(K=2)=0.636434; P(has orbit A)=0.516151; P(has orbit B)=0.492258 | Three 1,024-particle runs: maximum query errors 0.00850, 0.02413, 0.00532 |
| Continuous `vy` uniform in [-0.002,0.014] AU/day; star/orbit identity, K=0/1/2 and shared offset | P(K=2)=0.618662; P(has orbit)=0.848679; P(any orbit with vy<=0.006)=0.662211 | Three 2,048-particle runs: maximum query errors 0.01490, 0.00952, 0.01817 |

Orders 8 and 12 agree to about `1e-13` in the first control's query
probabilities and `2e-11` in the second. This is numerical convergence evidence,
not a rigorous quadrature enclosure. The continuous control frees one velocity
coordinate; arbitrary six-dimensional orbital posteriors are not validated by
that result. All six coordinates are available to proposals and independently
tested in forward evaluation where specified.

Each sequential report deliberately retains `approximate_uncertified`,
`accuracy_target_met=false`, and `[0,1]` certified probability intervals.
Agreement on these controls does not prove mode coverage in other scenes.
The full regression record and source/artifact hashes are in
[`orbital_results/validation.json`](orbital_results/validation.json).

## Shortlists and excluded possibilities

In the fixed-orbit control, deleting orbit B and renormalizing the remaining
class prior raises P(has orbit A) from **0.516151 to 0.897477**. Catalogues
containing the deleted orbit carry **0.492258** of the original posterior.
That deletion changes the model, including class/count relationships; it is
not an equivalent faster evaluation or simple conditioning of the old posterior.

A proposal shortcut may retain full prior support with correct proposal ratios;
finite-particle mode coverage would still be unresolved. A genuine domain
restriction requires a bound on **omitted evidence**, including excluded counts,
classes, continuous orbit regions, blends and shared nuisance configurations.
If retained evidence has lower bound `Z_R_lower > 0` and omitted evidence has
upper bound `U`, omitted posterior mass is at most
`U / (Z_R_lower + U)`. Bounding excluded prior mass alone is insufficient unless
combined with a valid likelihood bound. Existing interval-cover/count-tail
machinery is a potential starting point; this integration adds no shortcut and
claims no such new global bound.

## Remaining work and project separation

The immediate gain is correct orbital participation in FitSky's sequential
joint model. Broad six-dimensional mixing, full posterior accuracy, trailing,
physical perturbations and real multi-instrument validation remain unresolved.
There is no demonstrated scan-level or PFKT speedup from this integration.

Kepler keeps its original all-orbit transform/inverse objective. Its source,
benchmarks, result hashes and working branch are unchanged. This work starts
from FitSky commit `b650423`, in a separate worktree/branch, and targets
`research/fit-sky`. No merge of FitSky's older Kepler files onto `main` is
performed. The imported universal and interval modules are byte-identical to
Kepler main commit `92e359b`.

Reproduce from the integration checkout:

```sh
pip install '.[research,test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.fit_sky.orbital_experiments /tmp/orbital-fixed-new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.fit_sky.orbital_continuous_control /tmp/orbital-continuous-new
pytest -q tests/test_fit_sky.py tests/test_fit_sky_noise_floor.py tests/test_fit_sky_sequential.py tests/test_fit_sky_orbital.py
magictools-fit-sky-stream research/fit_sky/orbital_results/continuous/input.json /tmp/orbital-stream-new --particles 2048 --sweeps 16
```
