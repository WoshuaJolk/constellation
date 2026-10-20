# A global source-interpolating construction: theorem, certificates and limits

This is a derivation, not a claim of a new theorem in the literature or a
completed Fold Kepler tool. It adapts established demodulated interpolation
and butterfly ideas to the unequal dimensions of the orbital problem.
The distinction between the conditional result and the Kepler application is
part of the result. The quantitative certificates below discharge the local
interpolation hypotheses on three specified physical boxes. They do **not**
remove the six-dimensional representation cost or supply a physical inverse.

## Operator and scale

Normalize a bounded orbit chart to `X=[-1,1]^d`, with `d=6`, and time/frequency
coordinates to `S=[-1,1]^s`, with `s=3`. Let

```
K(x,y) = exp(i B psi(x,y)),   F(x) = sum_m f_m K(x,y_m).
```

`B>=1` is a physical oscillation scale, not the number of supplied points.
For Kepler, `B psi` is spatial frequency dotted with retarded apparent
position. The data points may be irregular or repeated. Neither their weights
nor the images are assumed analytic in time. Analyticity is required of the
kernel, including the observer path and projection.

Take `L=ceil(log2 B)`. At level `l`, the output boxes have half-width
`a=2^-l`; source boxes have half-width `b=2^-(L-l)`. Thus `B*a*b<=1`.
All output boxes are retained; empty source boxes can be omitted.

For output center `c_A`, demodulate

```
H_A(x,y) = K(x,y)/K(c_A,y).
```

Its oscillatory constant at the source center `c_B` can be removed without
affecting interpolation error:

```
H_A(x,y) = exp(i B[psi(x,c_B)-psi(c_A,c_B)]) exp(i R_AB(x,y)),
R_AB = B[psi(x,y)-psi(c_A,y)-psi(x,c_B)+psi(c_A,c_B)].
```

This is the standard mixed-phase residual, not a new Kepler identity.

## Explicit analytic hypothesis

Fix a Bernstein ellipse parameter `rho>1`. For every paired box `(A,B)`,
assume the source-variable extension is single-valued and holomorphic on the
product of scaled ellipses around `B`. For every real `x in A`, require a
known uniform bound

```
|R_AB(x,z)| <= D,  z in that product of ellipses.
```

The same `rho,D` must work at every level and for the entire requested domain.
This hypothesis is substantially stronger than sampled small block ranks.
It is a convenient sufficient hypothesis for the isotropic theorem. The
quantitative anisotropic proof below instead uses one complex source coordinate
at a time, with the others real; its separate error bounds suffice for the
same tensor telescoping argument without asserting the stronger product bound.

One sufficient condition is a uniform bound
`|partial_xi partial_yj psi| <= C_ij` on the relevant real-output/complex-source
sets. The ellipse and its interior are convex. Integrating along the real
output segment and the complex source segment gives

```
|R_AB(x,z)| <= B*a*b*e_rho*sum_ij C_ij
             <= e_rho*sum_ij C_ij,
e_rho = (rho + rho^-1)/2.
```

Here `|z_j-c_Bj|<=b*e_rho`; no derivative at sampled points substitutes for
the supremum. Consequently the demodulated exponential is bounded by `exp(D)`
on the ellipse product.

## Uniform local error

Use `p` first-kind Chebyshev nodes in each source coordinate, with tensor
interpolation `I_B` and rank `r=p^s`. Let `lambda_p` be a valid one-dimensional
Lebesgue bound, and `Lambda=lambda_p^s`.

A deliberately elementary bound is `lambda_p<=p^2`: at a root `x_j` of
`T_p`, the cardinal polynomial is
`T_p(x)/((x-x_j)T'_p(x_j))`. The real mean-value theorem, the polynomial
Markov inequality `max |T'_p|<=p^2`, and `|T'_p(x_j)|>=p` give each cardinal
polynomial absolute value at most `p`; summing gives `p^2`.
Sharper bounds are unnecessary for the asymptotic statement below. The actual
proof-order selector uses the stronger elementary bound
`lambda_p<=sqrt(2p-1)`. To verify it, write

```
l_j(x) = [1 + 2 sum_{k=1}^{p-1} T_k(x) cos(k theta_j)] / p,
theta_j = (j+1/2) pi/p.
```

Discrete cosine orthogonality gives
`sum_j l_j(x)^2 = [1+2 sum_{k=1}^{p-1} T_k(x)^2]/p <= (2p-1)/p`.
Cauchy–Schwarz then gives the stated Lebesgue bound. No sampled estimate of
the interpolation norm enters the certificate.

For an analytic one-dimensional function bounded by `exp(D)`, truncating
its Chebyshev expansion at degree `p-1` has error at most

```
E_p = 2 exp(D) rho^{-(p-1)} / (rho-1).
```

Applying interpolation to that polynomial and using its operator norm gives
error `(1+lambda_p) E_p`. A telescoping tensor interpolation identity therefore
gives the uniform local kernel bound

```
delta_p = (1+lambda_p) E_p sum_{j=0}^{s-1} lambda_p^j.
```

The other source coordinates remain real at each telescoping step, so the
same analytic bound applies. Multiplication by the removed real constant
phase has unit modulus.

## Actual transfers

Let `y_u^B` be source interpolation nodes and `l_u^B` their cardinal
polynomials. At level zero, the root output box is paired with fine source
leaves. Initialize

```
c_u^{A0,B} = sum_{y_m in B} l_u^B(y_m) K(c_A0,y_m) f_m.
```

At the next level, let `A` be an output child of `A'`, and `B` a source
parent with children `B'`. The transfer is

```
c_t^{A,B} =
  sum_{B' child of B} sum_u
    l_t^B(y_u^{B'})
    [K(c_A,y_u^{B'}) / K(c_A',y_u^{B'})]
    c_u^{A',B'}.
```

At the final output leaf, the source box is the whole source domain:

```
F_tilde(x) = sum_u [K(x,y_u^S)/K(c_A,y_u^S)] c_u^{A,S},  x in A.
```

This is a continuous, piecewise representation. It exists before selecting
any query orbits. Every one of the six output coordinates can vary at query
time. It does not compute a separate transform for each of three fixed outer
coordinates.

The source interpolation matrices between parent and child nodes are tensor
products of just two one-dimensional matrices per axis. Apply them by successive
one-dimensional contractions. The phase ratios are diagonal gates.

## Global error: interpolation instability must be charged

For fixed `x` and an input point `y`, follow the output ancestors of `x` and
source ancestors of `y`. Read the transfer expression backwards. At each level
it interpolates a product of a unit-modulus phase ratio and the continuation
from the next level.

The exact continuation is `K(x,y)/K(c_A,y)`. Thus at each level:

```
e_l <= delta_p + Lambda e_{l+1}.
```

The terminal interpolation also has error at most `delta_p`. By induction,
including the initial fine-source interpolation,

```
sup_{x,y} |K_tilde(x,y)-K(x,y)|
  <= delta_p sum_{j=0}^L Lambda^j =: E_global.

sup_x |F_tilde(x)-F(x)| <= E_global sum_m |f_m|.
```

This is an absolute `l1 -> Linfinity` operator estimate. It is not a relative
error guarantee at a cancellation or near a likelihood zero. It is also an
exact-arithmetic theorem: numerical dynamics, stored nodes, phase evaluation,
and contraction roundoff must be added in an implementation.

With fixed `rho,D,d,s`, the crude `lambda_p=p^2` bound permits

```
p = O((L + log(1/epsilon) + 1)
      log(L + log(1/epsilon) + 2)).
```

Indeed the negative term `-(p-1)log(rho)` in `log E_global` eventually dominates
`2s(L+1)log p + O(log L + D)`. Thus the required rank and transfer cost are
polylogarithmic in bandwidth and inverse accuracy. This does not bound the
physical constants `D` and `1/log(rho)`.

## Rectangular dimensionality and construction cost

For full trees the number of paired blocks at level `l` is exactly

```
P_l = 2^{d*l} 2^{s*(L-l)}.
```

For Kepler's `d=6,s=3`:

```
sum_l P_l = 2^{3L} (2^{3(L+1)}-1)/7 = O(2^{6L}).
```

There are `G=2^{6L}` final output boxes. The source has `2^{3L}` fine cells.
With sparse sources, replace the source-box count by its actual occupied count;
the formula above remains an upper bound.

For fixed dimension, the operation counts are:

* Initialization: `O(M p^s)` and `M` kernel samples.
* Gates: `O(p^s sum_l P_l)` kernel samples, up to fixed child-count factors.
* Tensor transfers: `O(s p^{s+1} sum_l P_l)` operations.
* Gates and field storage: `O(p^s sum_l P_l)`.
* `Q` arbitrary output evaluations: `O(Q p^s)` kernel work.

Consequently the construction and application are
`(M+G+Q) polylog(B/epsilon)` under the stated analytic hypotheses and a
bounded-cost kernel oracle. This accounts for construction, rather than hiding
a dense precomputation. For an orbit oracle, iteration/precision costs also
belong in the bound. Proving analytic existence alone does not bound numerical
conditioning of that oracle.

**This is not a theorem of near-linear cost in arbitrary sparse `M+Q`.**
For a requested continuous domain, `G` is a bandwidth-driven representation
cost even if `Q` is small. At `L=4`, a six-dimensional full output tree already
has 16,777,216 leaves; rank `p^3` multiplies that storage. A useful application
needs quantitative, preferably anisotropic, physical bounds.

In particular, if `N` means the number of resolved output cells, `G` is of
that order and this theorem is **not sublinear in N**. It can remove an
exposure-count factor from brute-force `N*M` work. If `N` instead denotes an
arbitrarily oversampled query list, constructing a continuous field before
choosing that list avoids its explicit enumeration; that different statement
must not be substituted for a sublinear physical-resolution result.

The previous balanced binary row/column trees counted samples rather than
these physical widths. Their observed failures do not refute this conditional
construction, but neither does this construction make their cost disappear.

## What must be established for retarded Kepler

For a noncollision real orbit on a finite interval, the two-body flow is
locally analytic. Bound/unbound/parabolic energy is not itself a singularity.
There is a useful qualitative corollary. Suppose the entire **compact** real
orbital domain over the necessary emission interval is separated from the
central collision, the observer, and the projection horizon, the observer
ephemeris is analytic, and the retarded root is unique with its implicit
derivative uniformly separated from zero. Analytic dependence of the ODE and
the analytic implicit-function theorem give local source-time extensions.
Compactness gives a finite cover and a positive common extension width.
Uniqueness patches the local branches adjoining the real interval. Shrinking
that width leaves finite mixed derivative bounds.

Thus some `rho>1,D<infinity` exist for a fixed regular compact Kepler domain,
independently of increasing spatial-frequency bandwidth. This yields the
preceding **asymptotic existence** result for such a domain; it does not yet
give an affordable construction. A constant hidden by that existence argument
can be enormous. An explicit **uniform quantitative complex extension** needs:

* separation from complex zeros of `r dot r`, and a consistent square-root
  branch for gravitational acceleration;
* separation from zeros of the observer-distance square;
* a unique holomorphic retarded-time branch, including a nonzero implicit
  derivative;
* separation from the projection horizon;
* a smooth observer ephemeris on the entire source interval and its complex
  neighborhood, including between observed epochs;
* bounds on mixed derivatives in that neighborhood.

Real-valued third/fifth derivative geometry certificates from the previous
campaign do not establish these conditions. The new Kepler prototype evaluates
the exact numerical retarded model at artificial source times, but a successful
sampled test does not prove the analytic hypothesis.

For arbitrary per-exposure WCS, prepare response fields in a common projection
or extend the source coordinates and count the additional dimension. It would
be incorrect to silently assume arbitrary headers are an analytic time function.

## Quantitative complex-time proof for the declared boxes

`complex_source_certificate.py` implements this step using 128-bit Arb/Acb
balls with outward rounding. It works with the all-energy regularized
identities in [the second route](regularization_attempt.md). All stored
floating-point physical constants are treated as exact numbers in this
mathematical model; the different rounding in the fast orbit solver is part
of the still-uncertified numerical implementation error.

Scale `u=rho*r`, `w=rho*v`, `mu'=mu*rho^3`, where `rho` here is inverse
distance, **not** the Bernstein ellipse parameter. With `R0=|u0|`,
`b=u0 dot w0`, `lambda=|w0|^2-2mu'/R0`, and `z=-lambda*s^2`,

```
u(s) = u0 + R0*w0*s + (-mu'*u0/R0+b*w0)*s^2*C(z)
       + lambda*R0*w0*s^3*S(z),
T(s) = R0*s + b*s^2*C(z) + (lambda*R0+mu')*s^3*S(z).
```

The entire functions `C,S` and their derivatives are evaluated with explicit
power-series tails. The certifier currently requires `|z|<1/2`; refusal
outside this range is a limit of this certificate, not a conic-domain filter.
The observer is exactly `O(t)=(cos(sqrt(mu)*t), sin(sqrt(mu)*t), 0)`.
The reference-emission chart has the known root `s=0` at reception `t=0`.
The retarded equation is

```
F(s;q,t) = T(s)-t + (sqrt((u(s)-rho*O(t)) dot (u(s)-rho*O(t)))
                      - reference_path)/(c*rho) = 0.
```

Every square root is certified in the right half-plane. For a complex
argument `a+ib`, the implementation uses
`U=sqrt((sqrt(a^2+b^2)+a)/2)`, `V=b/(2U)`. This preserves a small imaginary
width when `b` is small. It also verifies a positive real part of the
regularized radius `T_s(s)` and of the projection denominator, separating the
real branch from collision and the tangent horizon.

For a proposed disk `|s-s0|<=r`, choose a fixed approximate inverse `Y`.
Bound, uniformly in the entire real orbital cell and complex time rectangle,

```
beta = sup |1-Y*F_s| < 1,
d = sup |Y*F(s0)|,
d + beta*r < r.
```

The disk is convex, so the derivative bound proves that `s -> s-YF(s)` is
a contraction mapping the disk strictly into itself. Its unique root varies
holomorphically with source time; `F_s` is bounded away from zero. The
rectangle used for all interval evaluations contains the disk and the time
Bernstein ellipse. Newton iterations only propose `s0,Y,r`; their convergence
is not the proof.

The disk contains the known `s=0` reference root and the time rectangle
contains `t=0`. This identifies the physical branch by analytic continuation.
For real time, conjugation symmetry and uniqueness make the root real.
The certificate additionally checks positive real `Y` and `beta<1`, implying
positive real `F_s` on the real branch. The present depth-one source time
boxes all adjoin zero. Deeper boxes that cannot prove this connection are
refused; branch continuation across a longer time atlas is not implemented.

For each projected coordinate `P`, first-order interval jets give

```
dP/dq_j = P_qj - P_s * F_qj/F_s.
```

The initial direct interval expression was very loose. On `F=0`, substitute
`T=t+(reference_path-distance)/(c*rho)` into the position **before**
evaluating the intervals:

```
u(s)-rho O(t)
 = d + rho[O0-O(t)+V0*t] + (eta*d+wx*east+wy*north)*t
   + w0*(reference_path-distance)/(c*rho)
   - mu'[u0/R0*s^2*C(z)+w0*s^3*S(z)].
```

Off the root, the right side equals the original position minus `w0*F`;
therefore it has the same value and total implicit derivative on the root.
The observer terms `1-cos(theta)` and `theta-sin(theta)` are themselves
evaluated as `theta^2*C(theta^2)` and `theta^3*S(theta^2)`.
This exact cancellation is essential to practical bounds.

If a physical box fails interval evaluation, the proof can partition its
inverse-distance/radial-rate axes. **Every child must pass**, the union must
cover the original cell, and derivatives retain the **original** cell's
coordinate scales. Unioning the resulting derivative enclosures gives a bound
on the original cell. These subdivisions do not alter the transform hierarchy
or exclude difficult orbits. Branches agree on overlaps because each connects
to the same reference root.

An attempted all-time periapsis/subluminal certificate failed on the distant
box because it includes zero-angular-momentum states. Excluding those states
would change the objective. The local reference-connected certificate above
does not require that unnecessary all-time condition.

## Anisotropic interpolation orders from the certified derivatives

Use `p_t,p_x,p_y` nodes, with `lambda_j=sqrt(2p_j-1)`. For time interpolation
the other frequencies remain real. Integrating the enclosed imaginary orbital
derivatives along the output-center-to-query segment bounds the demodulated
time kernel by `exp(D_t)`. Thus

```
E_t = (1+lambda_t) * 2 exp(D_t) rho_ellipse^{-(p_t-1)}/(rho_ellipse-1).
```

For a spatial-frequency coordinate, time is real and the function is exactly
`exp(i*a*z)` times a unit-modulus constant, with `|a|<=U` from the real
derivative enclosures and that source box's half-width. The Chebyshev expansion
has coefficients `2 i^n J_n(a)` for `n>=1`. Poisson's integral gives
`|J_n(a)| <= (|a|/2)^n/n!` for real `a`. Consequently, if `U<2(p+1)`,

```
E_k <= 2(1+lambda_k) (U/2)^p / [p! (1-U/(2(p+1)))].
```

These identities are standard; see the NIST DLMF
[Bessel integral representations](https://dlmf.nist.gov/10.9)
and [Jacobi–Anger expansions](https://dlmf.nist.gov/10.12).
Taking the tensor interpolation in the stated order gives

```
delta = E_t + lambda_t E_x + lambda_t lambda_x E_y,
Lambda = lambda_t lambda_x lambda_y,
E_global <= delta * sum_{l=0}^L Lambda^l.
```

The selector searches for integer orders using ordinary floating-point logs,
then verifies the final bound using outward Arb arithmetic. Each proof stores
all cell/root enclosures and the maximum derivative-derived constants.
All three declared orbit boxes obtain `E_global<1e-8`; see the
[complete campaign records](global_results/README.md).
Those executions fix `L=1` and let the certified anisotropic orders grow;
they are not an empirical verification of the asymptotic
`L=ceil(log2 B)` cost regime. Increasing bandwidth at fixed depth can require
much larger orders. This distinction prevents a successful bounded execution
from being mislabeled as a practical polylogarithmic-bandwidth result.
This certifies the exact-arithmetic kernel approximation, not its rounded
nodes, numerical orbit evaluations, NUFFT tolerances, or image preparation.
It is an auditable computational proof argument, not a formal proof-assistant
verification or an independently reviewed mathematical discovery.

## Exact algebra that reduces implementation cost

Two standard separations make the sufficient orders executable.

First, for source positions `y=cos(theta)`, reflect `theta` in all `2^s`
sign combinations. A type-one NUFFT computes cosine moments
`sum_m f_m prod_j cos(k_j theta_mj)`. Applying a type-three DCT and dividing
by `p_j` on each axis produces precisely the Chebyshev cardinal moments.
This replaces explicit `M*prod(p_j)` initial interpolation with a reflected
Fourier transform plus tensor DCTs. The reverse type-two transforms implement
the adjoint. Numerical NUFFT tolerance is separately testable but not an Arb
certificate.

Second, at every source-time node, the kernel gate separates exactly:

```
exp(i[kx*DeltaX(t)+ky*DeltaY(t)])
 = exp(i*kx*DeltaX(t)) * exp(i*ky*DeltaY(t)).
```

Store `p_t*(p_x+p_y)` values per gate instead of `p_t*p_x*p_y`.
For final evaluation, compute the Kepler positions at the `p_t` artificial
times and evaluate the two-frequency sums using a two-dimensional type-three
NUFFT (or direct separated contraction for small query groups). This is an
evaluation of the same field, not a fitted replacement trajectory.
Tensor transfer cost and the six-dimensional leaf count remain; spatial gate
separation does not solve the global dimensionality problem.

## A concrete obstruction to the fixed-correction shortcut

Consider approximating `exp(i u)` on `[-R,R]` by a complex polynomial of degree
`n`, with uniform absolute error strictly below one. At the points `u=j*pi`
inside the interval, the real part of the approximating polynomial must alternate
in sign. There are `2 floor(R/pi)+1` such points. The intermediate value theorem
then forces at least `2 floor(R/pi)` distinct real roots. Therefore

```
n >= 2 floor(R/pi).
```

Any method that treats an unremoved phase residual ranging over this interval
as a constant-degree polynomial cannot have bandwidth-independent order.
For an exact quadratic residual `u=B*x^2`, use the alternating points in
`[0,B]` to obtain degree at least `floor(B/pi)` **in u**.

This rules out a particular constant-order correction claim. It does not
rule out Fourier transforms, rational approximations, hierarchical
refactorization, adaptive geometric degree, or a different exact phase.

## Inverse and nullspace

An adjoint follows by reversing and conjugate-transposing every gate and
interpolation contraction. It is an adjoint of the implemented approximate
operator, not automatically an inverse of the physical one.

For unrestricted exposure fields, constants give a simple exact nullspace:
if exposure `j` contributes a constant `a_j` everywhere on the domain and
`sum_j a_j=0`, their summed response vanishes for every orbit. Repeated
time/frequency samples also give identical columns. Thus a universal exact
inverse for arbitrary independent exposure fields cannot be asserted.
This argument applies to the linear response transform; additional detector
models or restrictions can alter its domain.

A finite generalized inverse requires a specified nullspace/gauge and spectral
conditioning. For a full-column-rank finite matrix with smallest singular
value `sigma`, an entrywise perturbation `E` bounds the spectral error by
`sqrt(MQ)*E`. Only when that is below `sigma` does the usual singular-value
perturbation argument even preserve full column rank. Fast inversion still
requires a separate algorithm/conditioning analysis.

## Relation to prior work

The residual-phase decomposition and complementary interpolation are established
in [Candès, Demanet and Ying (2009)](https://arxiv.org/abs/0809.0719).
Curved-path acceleration is already studied by
[Hu et al., generalized Radon transforms](https://www.ahay.org/RSF/book/tccs/butterfly/paper_html/paper.html).
[Kielstra et al. (2025)](https://users.math.msu.edu/users/jqian/papers/KielstraShiLuoQianLiuMMS2025.pdf)
study tensor butterfly structure and dimensional costs. Their results do not
automatically verify the physical Kepler hypotheses here.

The source-only construction, the elementary conservative error bound, the
six-versus-three dimensional count, and the counterexamples are a working
derivation for this project. They need independent specialist review before
any novelty or theorem-level research claim. They do not establish completion.
