# Constructing the segment fields: validated flow and cascade remapping

This continues the [full-objective contract](objective.md), without changing
the original near, distant or stress boxes. **The full FKT is not complete.**
The new implementation actually constructs shared fields bottom-up; it does
not evaluate the original exposure stack again at each parent node. It has
two interpolation backends, an algebraic adjoint, an explicitly regularized
inverse of its finite nodal operator, and a real-pixel likelihood experiment.

The distinction between existence, accuracy and affordable construction remains
essential. The saved experiments include failures and unfavorable total timings.
None of the measurements establishes novelty or a breakthrough.

## 1. A reception-coordinate ODE closes the transition-cover gap

The earlier [segment identity](segment_theory.md) used exact rebasing but did
not enclose the image of a complete parent box. That allowed a serious unresolved
possibility: interpolation nodes outside the true root-domain image might leave
the next chart. `reception_flow.py` supplies an interval enclosure of these
transitions, including the overhang.

Write the relative retarded position as `delta`, its reception-time derivative
as `w`, and `n=delta/|delta|`. Let `O,V,A` be the observer position, velocity and
acceleration at reception time. Source position and velocity at emission are

```
r = O + delta,
alpha = dt_emission/dt_reception = 1 - n.w/c,
v = (V+w)/alpha,
g = -mu*r/|r|^3.
```

Differentiating the light-time equation, with
`n'=(w-n(n.w))/|delta|`, gives

```
alpha' = -[(|w|^2-(n.w)^2)/|delta| + n.(g*alpha^2-A)] / (c+n.v),
w' = g*alpha^2 + v*alpha' - A.
```

For coordinates `q=(rho,eta,wy,x,y,wx)`, resolve `w'` in the fixed tangent
basis, and put `a_n=rho*w'_n`, `a_e=rho*w'_e`, `a_b=rho*w'_b`. Then

```
rho' = -rho*eta
eta' = a_n-eta^2
wy'  = a_b-y*a_n-2*eta*wy
x'   = wx
y'   = wy
wx'  = a_e-x*a_n-2*eta*wx.
```

These are exact two-body equations in reception coordinates, not a truncation
of the orbit into constant acceleration. The implementation uses an enclosed
dual basis: floating-point tangent axes are not assumed to be exactly
orthonormal in real arithmetic.

### Validated propagation

For a real box or complex rectangle `Q` and time increment `h`, find a tube `U`
such that

```
Q + [0,h] f([t,t+h], U) is strictly inside U.
```

All arithmetic is outward Arb/Acb interval arithmetic. The right-hand side
checks positive inverse range, separation from square-root branch points,
noncollision, an increasing emission-time map and subluminal source speed.
Picard inclusion and local regularity then establish an enclosing solution
through the whole step. The endpoint enclosure is `Q+h f([t,t+h],U)`.
Negative increments use the corresponding reversed real time interval.
Steps are subdivided dyadically; their work is charged.

There is no energy-sign test. Elliptic, parabolic and hyperbolic states can all
be represented. The certificate applies only to accepted regular tubes; it
refuses a singular or excessively wide tube instead of omitting states.
Bounds can be conservative because the first-order interval endpoint formula
wraps the state box. A sampled orbit is not used to establish the enclosure.

The original emitted-rate box is converted algebraically into a reception-rate
box. Computing `alpha-1` directly avoids destroying narrow angular-rate bounds
through cancellation. This conversion covers the entire original box.

These are exact-arithmetic geometric enclosures. They do not certify the
ordinary float universal solver, WCS, interpolation matrices or contractions.
In particular, small axis-orthogonality and rebasing errors in the numerical
chart implementation remain part of the unresolved floating-point accounting.

## 2. An executable bottom-up transform

`SegmentTransform` first builds its entire data-independent graph:

1. Partition the recorded exposures recursively by sample count.
2. Enclose each parent box at the child reception epoch, including overhang.
3. Build each child's field grid and the transfer from that grid to its parent.
4. Store the leaf exposure kernels.

Only after this construction does `transform(data)` receive source
coefficients. An internal node applies its two stored transfers to its already
constructed child fields and adds them. No internal node returns to the original
image stack. An arbitrary query is evaluated from the final root field.

The default field is a tensor of local Chebyshev-Lobatto interpolants. Orders and
cell counts can differ by coordinate and segment. A one-node axis gives a
constant approximation; it is not permission to assert that the physical field
is independent of that axis. Single-exposure response fields genuinely depend
only on two angular coordinates. Optional exact leaf evaluation uses the
declared exposure model at the immediately enclosing parent's nodes.

The Fourier model is linear in its source coefficients. A custom leaf kernel
can instead map raw pixels to sufficient statistics using a detector model.
Resource budgets reject large grids and transfer arrays before allocating those
arrays. There is no fallback to a hidden complete orbit-by-exposure loop.

The cadence split is by sample count. Actual physical spans are recorded:
clustered times do not silently become a balanced physical-time tree.

### A whole-domain error bound for the tensor backend

For each coordinate separately, the interval ODE propagates a complex Bernstein
ellipse while the other coordinates remain real. It encloses every real
exposure time in that segment. From the imaginary angular components,

```
|exp(i k.theta)| <= exp(|kx| |Im x| + |ky| |Im y|).
```

This gives a modulus bound `M_i` per unit source coefficient l1 norm. A
degree-`p-1` Chebyshev truncation has error at most
`2 M_i r^(-p)/(1-1/r)`. Multiplying by `1+Lambda_i` bounds its interpolation
error. For a local cell, a second bound follows from Cauchy's derivative
estimate and the Lobatto nodal polynomial. The smaller outward bound is used.

The one-dimensional Lebesgue bound is one for constant or linear
interpolation, and `sqrt(2p-2)` is a conservative bound for higher Lobatto orders.
The tensor telescoping sum includes every preceding coordinate's amplification.
Across the tree the normalized errors obey

```
E_parent <= delta_parent + Lambda_parent max(E_left,E_right).
```

The implementation charges the whole recursion. The saved sufficient bound is
far above `1e-8`; obtaining a finite enclosure is **not** advertised as satisfying
the accuracy target. The certificate is deliberately refused for custom WCS
leaf models and for the cascade backend, which need additional analysis.

## 3. A second attempt: replace the p^6 stencil by a cascade

The tensor construction made the next obstacle concrete. A six-dimensional
degree-three interpolant uses `4^6=4096` weights per target point. Higher order
rapidly makes the transfer matrices unaffordable.

This obstacle has relevant prior art. Purser and Leslie's
[1991 cascade interpolation paper](https://doi.org/10.1175/1520-0493(1991)119%3C2492:AEIPFH%3E2.0.CO;2)
replaces a multidimensional remap by successive one-dimensional interpolations
on hybrid grids. The later
[1995 conservative version](https://doi.org/10.1175/1520-0493(1995)123%3C2551:TDMCSL%3E2.0.CO;2)
extends that framework to conservative transport. Six-dimensional
[semi-Lagrangian Vlasov solvers](https://arxiv.org/abs/1903.00308) also illustrate
the value of one-dimensional interpolation stages, although their split
Hamiltonian evolution is not an exact reception-chart Kepler remap.

We do not claim the cascade idea as new.

### Exact-map factorization

Let `R` map a parent state `x` to child coordinates `y=R(x)`. After permuting
axes, define hybrid coordinates

```
H_j(x) = (x_1,...,x_j, R_(j+1)(x),...,R_6(x)).
```

Thus `H_0=R` and `H_6=identity`. Where each `H_j` is invertible, successive
representations

```
g_j(z) = f(R(H_j^{-1}(z)))
```

differ by changing only one coordinate. Their transfer can be implemented by
one-dimensional interpolation along that coordinate. This factorization uses
the full physical rebase; it does not split the orbital dynamics into approximate
drifts and kicks.

`cascade_transfer.py` solves the mixed-coordinate equations at each hybrid
grid node with a center-Jacobian preconditioned iteration. It checks every
residual and rejects failed inversions or queries outside the child box.
Additional angular padding is explicit, and its effect on all descendant
grids is charged. A test with equal four-node grids reduces the stored stencil
count from `4096 G` to `24 G`.

The hard caveat: convergence at every constructed node does not prove that the
hybrid maps are uniquely invertible throughout the intervening boxes. Nor does
it prove complex-domain regularity for an interpolation bound. The code exposes
these as false certificate flags. A whole-box inverse-function/contraction
certificate, or a subdivision strategy that supplies one, is still required.

### Cost implication, conditional on the missing certificates

For intermediate grid sizes `G_j`, the cascade stores and applies
`sum_j p_j G_j` weights instead of `p^6 G`. It also charges every geometry
evaluation and iteration used in the mixed inverses. If these maps are uniformly
conditioned and intermediate grids remain comparable to the segment grids,
this removes the tensor-stencil factor from the earlier short-segment count:

```
total work ~ sum_segments sum_axes p_axis * G_intermediate
             + all geometry/certificate work + image preparation.
```

Together with the earlier conditional anisotropic count, this is a more
concrete route to input-plus-output complexity. It is not a proved practical
FKT: the conditioning, intermediate cover growth, high-accuracy orders and
arbitrary-cadence depth all remain in the bill.

The adaptive-order experiment implements a stated short-segment heuristic and
records its errors. It is not substituted for a certified resolution rule.

### A concrete certificate to attempt next

At a cascade stage, let `u` be the unknown parent coordinates, `z` the already
fixed parent coordinates, and `Y` the target child-coordinate box. The mixed
equation is `F(u,z)=R_unknown(u,z)=y`, with `y` in `Y`. Given a candidate box `U`,
center `u0`, and a fixed approximate inverse `P`, enclose

```
K = u0 - P*(F(u0,Z)-Y) + (I-P*DF(U,Z))*(U-u0).
```

If `K` is strictly inside `U` and a weighted norm of `I-P*DF(U,Z)` is less than
one throughout the box, the contraction map supplies a unique solution for
every parameter pair `(z,y)` in the hybrid box. An additional interval
evaluation must prove that the replaced coordinate stays inside the child
interpolation interval. The same construction on suitable complex
neighborhoods would supply the holomorphic inverse required by an interpolation
error bound.

This is a proposed use of a standard contraction argument, not a certificate
already delivered here. It requires validated variational flow bounds and
possibly shorter exact-flow steps or chart subdivision. All such work and all
intermediate grid growth must be counted. A nonsingular sampled Jacobian alone
does not fulfill these conditions.

## 4. Adjoint and inverse have explicit meanings

Both backends are fixed linear graphs after construction. Reversing their
matrix products and conjugate-transposing the leaf kernels gives the algebraic
adjoint. The tests check complex dot products, including the final query map.

The new inverse solves

```
min_d ||A d-y||_2^2 + lambda^2 ||d||_2^2,  lambda > 0,
```

using LSMR. `A` is the finite root-nodal interpolation operator and the norm
uses the stated Euclidean nodal measure. The API reports iteration count,
normal residual, data residual and termination status. A small independently
assembled dense system verifies the result.

This is a Tikhonov inverse of that approximate finite operator, not reconstruction
of arbitrary raw observations from a continuous profiled likelihood. It does
not remove the previously demonstrated nullspace and conditioning issues.
Its iteration and conditioning costs cannot be discarded from a full inverse.

## 5. Real detector data, without an affine-WCS substitute

The experiment reads the same three archived ZTF exposures, pixel variances,
masks and full TPV WCS as the preceding campaigns. At each leaf it evaluates
finite Gaussian pixel integrals and profiles a constant background. One shared
geometry graph produces both the likelihood numerator and information term.
The direct reference independently evaluates the original pixels along the
retarded Cartesian trajectories.

A quintic Hermite observer model passes through the three recorded positions
and velocities. Its between-exposure accuracy is **not** independently
certified. The comparison uses the actual recorded observers at the exposure
times. The Gaussian PSF and variance approximations remain those of the original
experiment; this is not a new survey calibration.

Increasing angular order alone revealed a floor from the hidden/rate
coordinates. Refining those too brought the sampled numerator and information
errors below `1e-8`. That is a useful implementation result, not a whole-domain
detector error certificate or a discovery claim.

## 6. What this attempt does and does not finish

The transition-cover construction and the actual shared segment fields now
exist. The cascade is a concrete second attempt at the transfer cost, backed
by relevant prior literature. The likelihood pipeline and a specified finite
regularized inverse work in the experiments.

The broad-domain trials still do not meet the joint accuracy and total-speed
requirements. The full FKT remains open. The next mathematical bottleneck is
uniformly certified, well-conditioned hybrid-coordinate remapping with a
resolution rule that keeps intermediate grids affordable. Floating-point and
full detector error accounting, and the scope/conditioning of the inverse,
remain necessary parts of completion.
