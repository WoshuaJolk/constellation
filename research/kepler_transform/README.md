# Full-objective Kepler transform attempts

**Latest:** the [segment-composition attempt](segment_theory.md) follows the
FaXT/FDMT literature leads without substituting straight tracks for Kepler
orbits. It implements exact reception-time state composition, certifies that
two angular endpoints leave two locally independent orbital degrees of freedom,
and derives a conditional short-segment cost mechanism. It also certifies a
whole-domain near-null direction at a nonzero spatial frequency: removing DC
does not supply a uniformly stable inverse. [Results and replay](segment_results/README.md)
include the original three orbit boxes and an identity check on real ZTF pixels
with their full WCS. **The full objective is still unresolved; no new fast
transform or breakthrough is claimed.**

The [global mathematical campaign](global_results/README.md) derives
a source-interpolating hierarchy over the entire six-coordinate output box,
certifies its exact-arithmetic kernel error using complex-time interval
enclosures, and executes the sufficient orders on all three declared domains.
Its [theory](global_theory.md) explicitly charges the six-dimensional leaf
count, construction and interpolation instability. It also records an exact
regularization identity and the obstruction to a common exposure clock.
**It does not establish sublinear cost in resolved trajectories, a general
inverse, or a mathematical breakthrough.**

Explicit output of N values necessarily costs at least O(N). The relevant
FFT-like target is favorable total input-plus-output cost compared with separate
orbit-by-exposure evaluation. Sublinear construction of a compressed continuous
representation is a stronger goal, not a property of the FFT itself.

The preceding [observer-chart campaign](chart_results/README.md) resolves
the declared broad geometry domains with adaptive quadratic/quartic atlases,
tests actual ZTF pixels, and ships an experimental installed command and
library. Its [derivation and usage](chart_construction.md) separate continuous
geometry proofs from the remaining numerical and global-transform gaps.
Compiled direct-physics comparisons are the performance reference.

**Update:** the [second construction campaign](construction_results/README.md)
implements a sampled butterfly and a stronger localized NUFFT construction
with a six-dimensional geometry atlas. It achieves useful one-shot gains and
validated continuous geometry bounds on bounded cells, while preserving broad
domain failures. See its [derivation](construction.md). The first campaign and
its negative results are preserved below; its rotation method is no longer the
strongest implementation in this repository.

**First-campaign result: the full objective was not achieved. No breakthrough
was claimed.** The original campaign below preserves three approaches and a
further attempt to repair the strongest one, including the negative results.
The [acceptance criteria](objective.md) were committed before these experiments.

Its strongest intermediate result was a transform over all orbital orientations
for specified radial dynamics and harmonic response fields. Its directional
extension handles orientation-dependent light time in a smooth test. However,
constructing its response coefficients still evaluates every orientation at
every exposure, and its dense representation is unaffordable at telescope
resolution. That fails the objective even though the algebra works.

## Reproduce

From the repository root:

```bash
python -m pip install -e '.[test,research]'
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.experiments new-results.json
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_kepler_research.py
```

Output files must be new. Inputs are generated with seed 2026. The
[saved results](results.json) record software versions and source hashes.
Timings vary with hardware and are not used to infer a breakthrough. The code
runs offline. FINUFFT is optional; the production tools do not require it.

## Attempts and outcomes

| Route | What was tried | Evidence | Outcome against full objective |
| --- | --- | --- | --- |
| A. Global low-rank operator | Noise-weighted Gaussian template Gram matrices along physical orbits, including light time | Nearby templates need rank 1 for 99% energy; 256 resolved orbits need rank 254. Minimum Gram eigenvalue is about 0.99999956. | Global low-rank compression cannot uniformly preserve arbitrary resolved tracks. This does not exclude full-rank fast transforms. |
| B. Complementary phase blocks | State/frequency trees, exact block SVDs, application and adjoint, including near-parabolic and close-observer cases | At physical frequencies the tested 128–512 matrices retained dense storage. Relaxing the tolerance to 1e-3 did not fix the tested case. Low-frequency and constant-kernel controls compress. | This proposed partition/factorization failed. It is not a proof against butterfly algorithms. Dense construction also fails the cost target. |
| C. Radial invariants + rotation harmonics | Universal-variable dynamics, nonuniform phase FFTs, Wigner synthesis, adjoint and coefficient reconstruction | Elliptic, parabolic and hyperbolic tests match direct harmonic evaluation at about 2e-13 relative error. | Conditional identity, not a complete image transform. The simple moving-observer/light-time extension fails physical tests. |
| C, directional repair | Retain velocity-direction/roll harmonics and repeat with fully orientation-dependent light time | Smooth 10-AU test: relative error falls from 3.76e-6 at spin 0 to about 8.2e-10 at spin 1. | Construction used 38,808 orbit/frame evaluations, the same count as direct evaluation. No asymptotic construction gain. |

### A: a necessary low-rank condition fails

The Gram matrix for separated tracks is nearly the identity. A rank-r
approximation captures at most roughly r/N of the energy of N orthogonal unit
templates. A rank-deficient compression also loses some input direction
entirely. The numerical eigenvalue bound makes this obstruction robust to the
small physical template overlaps. This is standard linear algebra, not a new
impossibility theorem for fast Kepler transforms: the FFT is full rank and fast.

The experiment uses continuous Gaussian templates on an idealized tangent
plane with independent uniform noise. It tests a necessary condition for the
compression proposal; it is not a discovery experiment or real-image pipeline.

### B: reject this partition, not all hierarchical transforms

The tested kernel is exp(i k dot x_j(state)), with spatial Fourier modes up to
one inverse PSF sigma. These modes retain significant Gaussian PSF response.
Rows are six-dimensional states; columns combine time and two frequencies.
Balanced median trees supply complementary blocks. In these tests their ranks
are too large for profitable storage reduction.

Better coordinates, phase factorizations, interpolation schemes or domains
could behave differently. The prototype is **not** an implementation or
refutation of Hu et al.'s butterfly algorithm. These small experiments also do
not prove an asymptotic lower bound. Construction work and storage are reported
explicitly so multiplication timing cannot hide an expensive setup.

### C: exact conditional identity, unsuccessful full algorithm

A nonradial Kepler state is specified by radius, radial velocity, tangential
speed and a rotation. Its radial history and in-plane angle are independent of
orientation. This includes all conic energy classes.

For fixed radial dynamics and bandlimited response fields, the rotation identity
turns time accumulation into nonuniform Fourier transforms. It produces a
continuous harmonic representation over all orientations; it does not first
integrate an orientation bank. Independent Cartesian rotation and spherical
harmonic evaluation validate the algebra. The universal-variable propagator
is separately checked against DOP853 Cartesian integration, forward/backward,
on both sides of parabolic energy. Saved position discrepancies are <4e-12 AU.

Moving observers require reprojection. In smooth test fields, degree 12 reaches
the 1e-8 score-error target at 30 AU, but gives relative errors about 6e-5 at
3 AU and 0.15 at 1.02 AU. Higher degree could improve these errors; the result
does not establish that reprojection is impossible.

Emission time also depends on orientation for a displaced observer. Reusing
central-observer emission times yields maximum sky errors about 0.08 arcseconds
at 40 AU, 5.9 arcseconds in the 3-AU case, 97 arcseconds near parabolic energy,
and 116 arcseconds for the hyperbolic case. These compare the **same physical
orbit**, without fitting a different orbit to absorb the error. The saved
Gaussian response diagnostic quantifies resulting sensitivity loss under a
one-arcsecond PSF model, not a survey detection probability.

The directional extension keeps local velocity-direction information. It
resolves the light-time coupling in the smooth physical test, but constructing
its local response coefficients still costs O(NT). Both construction and direct
evaluation used 38,808 orbit/frame pairs. A small timing difference is not a
change in scaling. The full three-dimensional radial-domain covering problem
also remains unsolved.

### Telescope resolution is not low harmonic degree

For an explicit spherical heat-kernel PSF with one-arcsecond sigma, 99% of
template energy requires degree **442,637**. Degree 12 retains only about
**0.000063** of the optimum matched-filter S/N for this point source.

The implemented dense orientation grid at that bandwidth would contain about
3.47e17 real values (2.78e18 bytes). This is a property of this representation,
**not a lower bound on every localized or sparse transform**. The experiment
calculates the count; it does not allocate the grid.

Nor does roll bandwidth remain uniformly small. With exact light time and a
one-arcsecond PSF, the tested response needs spin limits 0, 8, 120 and 1,307 to
retain 99% of roll-response energy at radii 40, 3, 1.2 and 1.02 AU respectively.
This deliberately difficult sightline lies near the observer. It rejects a
uniform low-spin shortcut, not the directional identity itself.

## The inverse question

Adjoints and an inverse Wigner transform are implemented and tested. They do
not reconstruct the original time-separated images from summed likelihoods.

A universal exact image inverse has a simple counterexample. Take independent
exposures at the same epoch with identical response and noise. For every source
template p and flux F, the image pair (c+g,c-g) satisfies

```
||c+g-Fp||^2 + ||c-g-Fp||^2 = 2||c-Fp||^2 + 2||g||^2.
```

Replace g with a different spatial pattern h of equal norm. Every **full**
Gaussian likelihood stays unchanged, even with nonzero common signal c. This
is not merely information loss from flux profiling or thresholding. The test
uses a different spatial pattern, not an exchange of frame labels.

This elementary identifiability result is **not a breakthrough claim**. It
rules out unrestricted exact image reconstruction on this allowed input class,
while leaving fast forward transforms, adjoints, identifiable-quotient inverses
and regularized reconstruction open. If “inverse” means image synthesis from
an orbit population, that is a different operator needing its own definition.

## Remaining research target

The next attempt should address the unresolved requirements directly:

1. Construct localized directional coefficients **from pixels** without
   evaluating every orientation at every exposure. Count observer, light-time,
   masks, PSFs, noise and background work.
2. Prove tail/error bounds at telescope resolution, including near-observer
   and unbound cases. Smooth-field convergence alone is insufficient.
3. Cover the three radial parameters with controlled continuous error, rather
   than supplying a few chosen shapes.
4. Establish total construction/application scaling against a competent image
   baseline, including input/output and storage costs.
5. Compare a successful construction with prior generalized Radon, directional
   spherical and nonlinear-stacking methods before making a novelty claim.

The [derivation](derivation.md) preserves the useful identities and their costs
so a continuation starts at the construction problem, not another toy demo.

## Prior work

- [Whidden et al. (2019)](https://arxiv.org/abs/1901.02492): maximum-likelihood GPU trajectory searches.
- [Napier et al. (2026), heliostack](https://arxiv.org/abs/2604.16731): nonlinear stacking using radial invariants and reprojection, with real-data results. Basic invariant separation is not new.
- [Hu et al. (2013)](https://www.ahay.org/RSF/book/tccs/butterfly/paper_html/paper.html): generalized Radon transforms using butterfly structure. Curved paths alone do not establish novelty.
- [Wandelt and Gorski (2001)](https://arxiv.org/abs/astro-ph/0008227): fast spherical convolution over rotations and transpose operators.
- [Barnett, Magland and af Klinteberg (2019)](https://arxiv.org/abs/1808.06736): FINUFFT, used for nonuniform Fourier sums and adjoints.

These are prior-art leads, not an exhaustive novelty review. The derivation
applies established identities; no novelty has been established.
