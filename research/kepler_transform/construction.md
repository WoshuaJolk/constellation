# Localized Kepler transforms: construction and continuous error

This second attempt targets the expensive construction left unresolved by the
[first attempts](README.md). It implements two different constructions and
tests both. **It does not claim the full catalogue objective is solved.**

## What is new in this repository

The first construction is an interpolative butterfly assembled from requested
kernel entries. It has a forward operator, algebraic adjoint, and regularized
least-squares inverse. Unlike the earlier diagnostic, it never first builds a
dense global matrix. Its rank caps and sampled residuals are explicitly
non-certifying. Physical tests and a Fourier positive control expose the actual
construction work, storage and accuracy.

The stronger construction separates smooth orbital geometry from rapidly
varying image response. It builds a polynomial representation of apparent
position over a **six-dimensional Cartesian initial-state cell**, including
moving observers and orientation-dependent light time. A phase-corrected
three-dimensional NUFFT evaluates whole sets of inner coordinates at once.
The other three coordinates parameterize the transform and can be queried
between construction nodes. They are not silently fixed to one orbit shape.

Validated variational integration bounds the geometry approximation throughout
the continuous cell. This is stronger than agreement at a set of test orbits.
Separate analytic bounds cover phase-series truncation. Floating-point phase
construction and the NUFFT itself are not formally interval-certified.

## The representation

Use initial state

```
state(u,z) = center + sum_a outer_basis[a] u_a + sum_b inner_basis[b] z_b,
u,z in [-1,1]^3.
```

The six basis vectors must be independent. The experiments assign distance,
radial velocity and one transverse velocity to the outer coordinates; two
transverse positions and the other transverse velocity are inner coordinates.
These coordinates are numerical choices, not restrictions to elliptic energy.

Sample the exact retarded apparent position at the 729 tensor nodes with each
normalized coordinate in `{-1,0,1}`. Transform those values into tensor
polynomial coefficients. At a requested outer coordinate, evaluate its three
polynomial dimensions to obtain an inner-coordinate position polynomial.
Retain its total-degree-two part:

```
p_j(u,z) ≈ p0_j(u) + sum_a J_ja(u) z_a
                     + sum_|alpha|=2 H_jalpha(u) z^alpha.
```

The omitted higher mixed inner terms are charged to the geometry error, rather
than assumed zero. A simpler local construction uses only 27 nodes when the
outer state is supplied as a fixed value.

For supplied Fourier response coefficients, the score is

```
S(u,z) = sum_m d_m exp(i k_m · p_visit(m)(u,z)).
```

Substitution separates the phase into a constant, a linear part `g_m · z`,
and a quadratic residual `q_m(z)`. Expand only the residual:

```
exp(i q) = sum_s=0..Q (i q)^s/s! + remainder.
```

Collect equal monomials. Each coefficient sum has the form
`sum_m c_m exp(i g_m · z)`, which is a type-3 3D NUFFT. An order-one correction
requires seven such sums, order two 22. Rapid angular oscillation remains in
the NUFFT; it is not approximated by a low-degree score polynomial.

For real `q`, the integral Taylor remainder is at most
`|q|^(Q+1)/(Q+1)!`. On the cube, the sum of absolute quadratic coefficients
bounds `|q|`. Summing this remainder against `|d_m|` gives a uniform absolute
score bound. The six-dimensional atlas also bounds the quadratic coefficients
over every outer coordinate, giving a phase-tail bound for the entire cell.

## Continuous geometry certificate

`jet_geometry.py` carries univariate third-order derivative jets in one initial
state direction at a time. The constant coefficient encloses the **whole**
six-dimensional state cell; the first coefficient is the selected basis
direction. Jet coefficients are derivatives divided by factorials.

The jets obey the differentiated two-body ODE. A Picard inclusion encloses this
extended ODE, and time-Taylor coefficients plus a validated remainder propagate
all four coefficients. This is ordinary interval automatic differentiation and
validated integration, not a new mathematical theorem.

Light time is implicit. At an enclosed emission time, let `R`, `V`, `A` and `J`
be the position, velocity, acceleration and jerk jets. Compose the position jet
with an unknown time-shift jet `dh`:

```
R(z,h(z)) = R(z,h0) + V dh + A dh²/2 + J dh³/6  (through degree 3).
```

Solve each successive coefficient of `h + distance/c = 0`. Its coefficient
of the unknown time derivative is `1 + v_los/c`; the solver requires a positive
enclosure. The light-time interval is invariant and the enclosed speed is below
`c`, establishing a unique retarded root on the tested interval. Projection
then propagates the jets through the supplied gnomonic basis. Cells intersecting
a collision, observer singularity or tangent horizon are refused.

For quadratic interpolation at `{-1,0,1}`, the Lebesgue constant is `5/4` and
the maximum nodal product is `2/(3 sqrt(3))`. A telescoping tensor-interpolation
identity bounds the error by

```
2/(3 sqrt(3)) * sum_a (5/4)^a * sup |partial_a^3 p| / 3!.
```

Each derivative bound applies throughout the whole cell. Independently
validated positions at every interpolation node provide interval polynomial
coefficients. Comparing these with the stored coefficients charges numerical
fitting error and discarded mixed terms. The atlas also bounds the roundoff
of each outer-coordinate polynomial contraction using Arb.

For position-coordinate errors `ex_j, ey_j`, a response error bound is

```
sum_m |d_m| (|kx_m| ex_visit(m) + |ky_m| ey_visit(m)).
```

The implementation binds certificates to hashes of the represented polynomial,
observer positions, projection bases and exposure times. A certificate for one
polynomial cannot be used with another through the bound helper.

## Pixels and actual likelihood statistics

`pixel_fields` constructs both Gaussian sufficient statistics from pixels:

```
N(theta) = sum_j p_j(theta)^T W_j d_j
I(theta) = sum_j p_j(theta)^T W_j p_j(theta)
Delta log L(F,theta) = F N(theta) - F² I(theta)/2.
```

The implemented test model uses odd, square periodic detectors and a finite
Fourier representation of a pixel-integrated Gaussian PSF. Background is known;
variances may vary per pixel, and masks set weights to zero. The information
field retains the **doubled frequency support of the squared PSF**. Replacing
it by constant information would be incorrect with masks or changing weights.

Independent tests calculate templates directly in pixel space and compare both
statistics, including subpixel positions. The end-to-end experiment injects
a synthetic source into synthetic Gaussian images. It is not real-survey
validation. Periodic boundaries, bandlimited PSFs and known noise remain
explicit restrictions; a real camera cannot be assumed to follow them.

## Inverse and information loss

Both constructions implement their algebraic adjoints. The local construction
also solves a finite-query, optionally Tikhonov-regularized least-squares inverse
with LSQR. Tests compare it with an independent dense augmented least-squares
solution. Iteration limits and conditioning estimates are returned.

This inverse estimates Fourier coefficients of the specified finite operator.
It cannot reconstruct modes in its nullspace or uniquely restore arbitrary
time-separated images. The earlier [full-likelihood counterexample](README.md#the-inverse-question)
still applies. No adjoint is labeled an exact inverse.

## Costs and failure conditions

For `T` frames, `M` Fourier modes, `A` outer queries and `N` inner queries per
outer query, the atlas samples 729 trajectories, costing `729*T` orbit/frame
evaluations. Its certificate additionally uses 729 interval node propagations
and six variational propagations per frame. The simpler fixed-outer construction
uses 27 trajectories and three variational directions per frame.

After construction, each outer query performs a bounded number of NUFFTs over
`M` modes and `N` targets, with a phase-space-dependent grid cost. It no longer
integrates all `A*N` trajectories. These costs, including certification, are
reported separately and together against fresh exact geometry plus per-frame
NUFFT evaluation. Dense multiplication is not the competitive baseline.

The NUFFT grid grows with coordinate bandwidth. The code checks a documented
grid-volume heuristic before entering FINUFFT. It is a work-budget guard, not a
guarantee of the native library's allocation. Broad domains can fail both the
geometry tolerance and this budget. The constant-score interval atlas preserves
all unresolved cells and verifies their exact dyadic covering volume; an
unfinished cover is never reported as a completed transform.

Remaining full-objective gaps are profitable coverage of broad orbital domains,
including difficult near-observer/collision-adjacent regions; a globally useful
outer-coordinate synthesis strategy; realistic nonperiodic instrument models;
and complete numerical certification. A one-shot speedup on a bounded cell
does not establish the catalogue's all-orbit transform.

## Reproduce

```
python -m pip install -e '.[test,research]'
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.construction_experiments new-results
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_kepler_construction.py
```

The result directory must be new. `--quick` runs a smaller smoke experiment;
its timings must not be substituted for the main benchmark. Each case is saved
when finished; `complete.json` is written only after the campaign completes.
The input seed, software versions, source hashes and synthetic pixel inputs
are retained. See the checked-in `construction_results/` for measured outcomes.

## Prior methods

- Li et al., [Butterfly Factorization](https://arxiv.org/abs/1502.01379):
  complementary low-rank operators and constructions from sampled entries.
- Barnett, Magland and af Klinteberg,
  [FINUFFT](https://arxiv.org/abs/1808.06736): fast nonuniform Fourier sums,
  including type-3 sums in one, two and three dimensions.
- The [first attempt's references](README.md#prior-work) cover generalized
  Radon transforms, rotational convolution, digital tracking and nonlinear
  orbital stacking.

Interpolation, automatic differentiation, validated ODE integration, Fourier
phase expansion and regularized least squares are established methods. This
implementation and these tests do not establish mathematical novelty.
