# Segment composition and the inverse: a full-objective attempt

The original [acceptance criteria](objective.md) still apply. This attempt does
not deliver the full FKT, does not replace it with a straight-line detector, and
does not claim mathematical novelty. It develops a second route to a fast
forward operator, tests the prerequisite geometry, and derives a quantitative
obstruction to an unconditioned inverse.

The code and saved interval certificates are computational mathematics, not
formal proof-assistant output or independent specialist review.

## Why investigate a different recurrence?

[Nguyen et al. (2024), FaXT](https://tamz.umd.edu/papers/Nguyen_2024_AJ.pdf)
reuse short discrete line segments across a four-dimensional track space.
[Zackay and Ofek, FDMT](https://arxiv.org/pdf/1411.5373) reuse partial sums along
dispersion curves; section 4 discusses other curve families with a unique curve
through two endpoints. Neither result gives a six-parameter Kepler recurrence.
The useful question is whether a short Kepler segment can have a cheaper
representation than a long one, while retaining all orbital states.

### Two angular endpoints are insufficient

At fixed reception epochs define

```
E(q) = (x(q,t_left), y(q,t_left), x(q,t_right), y(q,t_right)),
M(q) = (x(q,t_middle), y(q,t_middle)).
```

All six coordinates of `q` vary. `endpoint_rank_certificate` uses outward
Arb enclosures of implicit retarded Kepler derivatives at an interior state of
each unchanged near/distant/stress box. It verifies:

* a four-by-four minor of `DE` excludes zero;
* the determinant of `D(E,M)` excludes zero.

The implicit-function theorem then gives a local two-dimensional set of states
with exactly the same endpoints. Moreover, `DM` restricted to the tangent space
of that set has rank two: otherwise the combined six-by-six derivative would
have a nontrivial null vector. The midpoint therefore varies on that set.

Local root disks need not contain the reference epoch. For these **point**
certificates, a separate nonzero-angular-momentum and globally subluminal
periapsis certificate identifies the unique physical real retarded branch.
This is different from incorrectly accepting an unconnected complex root
solely because its local Newton iteration converges. The disk also certifies
positive radius and projection denominator and a simple retarded root.
Regularity at the point supplies an open neighborhood for the theorem.

Numerically fitted pairs inside the original boxes make the scale tangible:
the stress pair differs by about 7.25 arcseconds at the midpoint while its
endpoints agree to below 1e-7 arcseconds. An independent universal-variable
solver checks that witness. The finite fitted pair is a numerical witness;
the interval derivative argument separately certifies **local existence** of
endpoint ambiguity, not the exact fitted decimal pair.

This rules out discarding the hidden coordinates in a literal endpoint-only
FDMT adaptation. It does not rule out six-state dynamic programming.

## An exact reception-time chart

Using emitted velocity as if it were observed angular velocity leaves small
first-order light-time terms. Those terms matter for a claimed short-segment
scaling theorem. Define rates with respect to **reception time** instead.

At a reference reception epoch, let `O,V` be the observer position and
velocity. With orthonormal tangent axes `(n,e,b)`, write

```
D = n + x e + y b,
delta = D/rho,
w = [eta D + wx e + wy b]/rho.
```

Here `rho` is inverse axial range, `eta` its fractional axial range rate, and
`wx,wy` are the actual tangent-coordinate derivatives. The vector `w` is
`d delta/dt` along the retarded observation, not the emitted Cartesian velocity.

Since `t_emission = t_reception - |delta|/c`,

```
alpha = dt_emission/dt_reception = 1 - delta_hat dot w/c,
r_emission = O + delta,
v_emission = (V+w)/alpha.
```

These are exact identities. The implementation rejects nonincreasing
reception/emission maps and nonsubluminal states; it does not filter by conic
energy. Noncollision and a positive tangent denominator are also required
where the chart is used.

Conversely, from an emitted state,

```
alpha = (1 + delta_hat dot V/c)/(1 + delta_hat dot v_emission/c),
w = alpha v_emission - V.
```

Taking axial and tangent components recovers `rho,eta,x,y,wx,wy`. Propagating
the emitted state by its light delay yields the state at the reception epoch.
`ReceptionChart` implements both directions, including generic scalar/interval
initial-state algebra. Its digest and serialized rate convention distinguish
it from the older emitted-rate `ObserverChart`.

### Exact composition is available

Let `C_a` map the reception chart at epoch `a` into a Cartesian state at `a`,
and let `Phi_h` be the two-body flow. Define

```
R_ab = C_b^{-1} o Phi_(b-a) o C_a.
```

On overlapping regular chart domains,

```
R_bc o R_ab = R_ac.
```

For exposure response fields `f_j` and a segment `I`,

```
S_I(q_a) = sum_(j in I) f_j(apparent_position(q_a,t_j)).
S_I(q_a) = S_left(R_a,left(q_a)) + S_right(R_a,right(q_a)).
```

This identity retains observer motion, light time, and all conic energies.
Different images need not be smooth in time. It applies separately to the
numerator and information sufficient statistics; a profiled likelihood is
formed afterward.

The numerical experiment checks rebase composition and this score identity,
including bound/unbound samples, and separately tests an exactly parabolic
state. A DOP853 integration plus independent scalar light-time root checks the
new coordinates. These are validations of the implementation, not uniform
floating-point error certificates.

**Evaluating this recursion independently for each query still costs O(Q T).**
Composition alone is not the fast transform.

## The potential saving: anisotropic short-segment resolution

In the new coordinates, for time offset `s` the apparent direction satisfies

```
theta(q,s) = theta_0 + omega_0 s + s^2 A(q,s).
```

One exact definition is
`A(q,s) = integral_0^1 (1-u) theta''(q,us) du`.
On a regular compact neighborhood with a twice differentiable observer this
identity follows from Taylor's theorem with integral remainder. If the
observer and physical model are analytic, the corresponding local analytic
extension can be used for high-order interpolation.

Differentiating with respect to the two hidden coordinates gives

```
partial_(rho,eta) theta(q,s) = s^2 partial_(rho,eta) A(q,s).
```

The angular-position sensitivities have order `1`, the angular-rate
sensitivities order `h`, and the hidden-coordinate sensitivities order `h^2`
on a segment of half-span `h`. The constants require bounds over the full
domain and can grow badly near singularities or over long baselines. They are
not one merely because these powers hold. The experiment observes this
short-time behavior; it does not use observed ratios as certified constants.

### A conditional count, not a finished mesh construction

Normalize coordinate ranges and put physical derivative constants into a
bandwidth parameter `B>=1`. Suppose, additionally, that:

1. boundedly many regular product charts cover every segment domain and all
   transitions needed by the interpolation nodes, including nodes outside the
   exact image of the original domain;
2. uniform complex-neighborhood bounds justify the anisotropic cell widths
   `min(1, 1/B)`, `min(1, 1/(B h))`, and `min(1, 1/(B h^2))`;
3. local interpolation and all transition evaluations have the charged costs;
4. a balanced dyadic time partition of depth `L` suffices.

Then a segment needs, up to fixed constants and interpolation order,

```
G(h) = B^2 (1+B h)^2 (1+B h^2)^2
```

cells. There are two coordinates of each kind. At single-exposure leaves,
only the two angular coordinates remain. For `h_l=H/2^l`, the internal count
is bounded by `sum_l 2^l G(h_l)`, with a separate `N B^2` leaf cost.

Expanding the product gives nonnegative terms `C_a h^a`, `a=0,...,6`.
For `a>=2`, the level sum is a convergent geometric sum bounded by a constant
times the corresponding root term. For `a=1` it costs `L` times that term;
for `a=0` it costs the number of leaves. Consequently,

```
sum_l 2^l G(H/2^l) = O(L G(H) + N B^2).
```

The count is compatible with near input-plus-output work. It does not prove
that the required chart covers or interpolation constants are affordable.
The stored integer count experiment uses `max(1,ceil(B h^a))` factors and
marks those missing hypotheses explicitly.

For arbitrary clustered observing times, a balanced tree by **sample count**
need not halve physical span. A physical-time tree can have depth depending
on the smallest time separation, not `log N`. This cost must be counted; the
dyadic count is not silently a theorem for every cadence.

### Error and construction requirements that remain

If a local interpolant has sup norm at most `Lambda` and a local weighted
error `delta * sum_(j in I) ||f_j||`, then normalized errors obey

```
e_parent <= delta + Lambda max(e_left,e_right).
```

Thus the global bound includes `delta sum_(l=0)^L Lambda^l`. Smooth-looking
plots or small sampled errors cannot replace that amplification factor.
With quantitative analytic hypotheses, exponential-in-order local error
can offset it using growing interpolation order. The resulting stencil
cost in six dimensions must also be counted.

Each parent node must be evaluated by interpolating its two **already built**
child fields at rebased states. Filling parent values by summing the original
images would restore the very `Q T` construction we are trying to avoid.
Leaf fields can use image filtering/Fourier evaluation without orbit-bank
enumeration. Cover construction, rebase evaluation, image preparation,
storage and query costs all belong in the final result.

This attempt implements and validates the chart and exact identities. It has
**not** constructed and certified that complete anisotropic field hierarchy.
There is no supported claim that it beats the existing source butterfly.
The missing cover/transition/interpolation construction is a mathematical and
implementation gap, not an invitation to omit hard states.

## A stronger inverse obstruction than the DC nullspace

Removing constant exposure offsets does not ensure a well-conditioned inverse.
Here is a quantitative example using a nonzero spatial frequency and distinct
epochs, on the entire unchanged near orbital box.

For fixed physical frequency `k`, let

```
K(q,t) = exp(i k dot theta(q,t)),
H(q,t) = K(q,t)/K(q_center,t).
```

The existing complex-time machinery, recomputed by this experiment, certifies
that `H` is holomorphic on a time ellipse with center `t_c`, half-width `a`,
parameter `rho_ellipse`, and `|H| <= exp(D)` for every real orbit in the box.
The bound follows by integrating the imaginary part of the enclosed orbital
derivatives from `q_center` to `q`. It is a whole-domain bound, not a sampled
matrix singular value.

The ellipse contains a disk of radius
`b = a*(rho_ellipse-rho_ellipse^-1)/2`.
Take `m+1` distinct equally spaced epochs in
`[t_c-S/2,t_c+S/2]`, with `S/2<b` and step `h=S/m`.
Every point in that interval has a complex disk of radius `R=b-S/2`
inside the ellipse. Cauchy's derivative estimate gives

```
sup_(q,t) |partial_t^m H(q,t)| <= m! exp(D)/R^m.
```

Set

```
a_j = (-1)^j binomial(m,j)/sqrt(binomial(2m,m)),
d_j = a_j / K(q_center,t_j).
```

Vandermonde's binomial identity gives `sum_j |d_j|^2=1`. The repeated-integral
formula for a finite difference yields

```
sup_q |sum_j d_j K(q,t_j)|
 <= exp(D) m! (h/R)^m / sqrt(binomial(2m,m)) = E.
```

This is a **near-null** direction, not an exact nullspace assertion.
If an inverse reconstructs all these coefficients, its Lipschitz constant
from the sup norm on the full response field to input Euclidean norm is at
least `1/E`. With a probability-normalized output sampling measure the same
bound applies to the output L2 norm. The unnormalized finite-matrix norm
introduces its usual square-root-of-row-count factor.

For real input fields use
`f_j(x)=sqrt(2)*a_j*cos(k dot x-k dot theta(q_center,t_j))`,
or the conjugate-symmetric Fourier coefficient pair. The coefficient norm
remains one and the response bound is `sqrt(2)*E`. Thus this is not an
artifact of allowing complex-valued data.

The saved certificate uses 33 distinct epochs over one day, one nonzero
frequency, and the original six-dimensional near box. Its real-field bound
implies an inverse amplification greater than `1e17`. A second certificate
uses 17 epochs over three hours. Both are evaluated outward with Arb and
include newly computed physical root enclosures.

These statements concern the response-field coefficient operator. They do
not automatically identify a null direction in a particular finite,
nonperiodic detector's raw-pixel subspace. That restriction could change the
spectrum and must be analyzed explicitly. Conversely, profiling backgrounds,
masking pixels or retaining only a profiled scalar likelihood introduces
additional information loss.

The phase-compensated coefficients above are defined in exact arithmetic.
Ordinary floating-point coefficients, time/geometry evaluations and summation
introduce a floor around 1e-12 in this experiment. Those numerical residuals
are **not** verification of a smaller exact bound. Both are recorded, with
different labels. The exact certificate is established by derivative bounds,
not cancellation observed in float64.

### What this means for an inverse

A specified gauge removes exact null directions. A regularized or truncated
generalized inverse can recover components above a declared singular-value
threshold. Neither recovers arbitrary lost or arbitrarily ill-conditioned
components, and its cost must include iteration count and conditioning.
[Feliu-Fabà and Ying (2021)](https://web.stanford.edu/~lexing/invfio.pdf)
provide fast inverse machinery under additional operator assumptions; those
assumptions are not established for this Kepler operator.

The result does **not** prove that a fast forward FKT or a useful generalized
inverse is impossible. It does show why "forward plus adjoint" or
"remove DC and invert" cannot constitute an unqualified completion.

## Outcome

Three things are now more concrete: endpoint-only reuse fails locally by a
certified rank argument; a six-state reception-time recurrence has a plausible
conditional cost mechanism; and the inverse has severe whole-domain
conditioning even without zero frequencies or repeated epochs.

The full objective remains unresolved. Required next work is a complete
certified field hierarchy (or an improved butterfly construction), measured
total cost at physical resolution, detector-level coverage/error accounting,
and an explicitly conditioned generalized inverse. These cannot be claimed
from the identities or small tests here.
