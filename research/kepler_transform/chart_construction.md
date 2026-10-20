# Broad observer-chart construction

This backend supplies an adaptive, continuous six-dimensional **geometry**
representation, fast finite-query Fourier responses, an adjoint and regularized
inverse, and a finite-detector likelihood adapter. It is an experimental library
and command, not completion of the unrestricted all-orbit transform. A new
celestial discovery is not a completion requirement.

See [the fixed campaign objectives](chart_objective.md) and
[measured results](chart_results/README.md). The stronger compiled comparison
supersedes speed claims based only on the Python physical reference.

## Coordinates and dynamics

Let `O0,V0` be the reference observer state; `n,e,b` its tangent basis. Define

```
q = (rho, eta, wy, x, y, wx)
d = n + x*e + y*b
r = O0 + d/rho
v = V0 + (eta*d + wx*e + wy*b)/rho
```

`rho>0` is inverse axial distance in AU^-1; `eta,wx,wy` have units day^-1.
`x,y` are dimensionless tangent coordinates. With `retarded_epoch=True`,
this state is specified at emission time `-norm(d)/(rho*c)`, seen by the
reference observer at reception time zero. Angles then describe that observed
direction; the rate parameters still describe the emitted state, not exactly
the observed angular derivative. `ObserverChart.states` converts to Cartesian
states at epoch zero. Its inverse solves the reference light-time problem.
No bound/unbound energy cut is imposed.

This removes much of the spurious distance dependence at the reference image.
It does not make arbitrary large orbital domains polynomial.

For certificates integrate `u=rho*r` instead of dividing a wide interval by
`rho`. The dynamics are

```
u'' = -mu*rho^3 * u / norm(u)^3
(c*rho)*(h-t) + norm(u(h)-rho*O(t)) - norm(d) = 0.
```

The second equation is the retarded observation equation. The same `rho`
changes gravity, observer displacement, and the effective speed of light.
All these derivatives enter the variational equations. Cubic coefficients of
`rho^3` are evaluated analytically; monotone endpoint powers preserve its
known positivity even in wide interval cells.

Third- or fifth-order parameter jets carry derivatives divided by factorial.
Their constant coefficient encloses the **whole** parameter cell. Validated
time-Taylor steps use a Picard-enclosing tube and a remainder. The implicit
light-time coefficients are solved triangularly, with denominator
`c*rho + dot(q,v)/norm(q)`. The fifth-order construction composes the full
fifth-order time series, not just position/velocity/acceleration/jerk.
Collision, observer-intersection, horizon or enclosure failures remain visible.

Independent 60–70 digit universal-variable calculations test these derivatives,
including the parameter dependence of the reference emission epoch.

With `translated_certificate=True`, the interval state instead stores
`s = rho*(r-O0)`, with force evaluated at `s+rho*O0` and the observer at
`rho*(O(t)-O0)`. Validated position updates retain the relative constant
coefficient directly. This avoids subtracting two wide origin enclosures
after propagation. It changes the proof representation, not the orbit model.
The widest benchmark then needs five certified cells instead of seven.
The unsplit quartic polynomial still exceeds the requested accuracy on
sampled points; changing interval coordinates alone cannot make it sufficient.

## Interpolation and proof

The quadratic atlas uses 3^6 = 729 nodes. The higher-order atlas uses
5^3*3^3 = 3,375 nodes: quartic in the three outer coordinates, quadratic in
the three inner ones. Nodes are exact rationals.

For a tensor interpolant, a telescoping identity bounds the error by a sum of
one-dimensional remainders times the preceding interpolation operator norms.
Quadratic interpolation uses Lebesgue bound 5/4 and nodal-product bound
2/(3 sqrt(3)). Quartic interpolation uses bounds 3 and 1/8. The latter are
checked with interval polynomial evaluation on an exact 256-interval cover;
this is not a sampled maximum. Third/fifth variational derivatives supply the
corresponding remainders.

Only inner terms of total degree at most two are retained for fast summation.
Arb node calculations independently construct the exact tensor polynomial and
charge both discarded terms and differences from the stored floating-point
coefficients. Geometry certificates bind the chart, exposure geometry and
polynomial by hashes. Slice contraction roundoff is separately bounded.

The adaptive cover first screens derivative bounds, then spends the nodal
verification cost only where those bounds can meet the tolerance. A failed
sufficient bound means “subdivide,” not “the actual error is necessarily large.”
Every unresolved leaf remains in the cover. Exact rational volume and
disjoint-interior checks verify the logical partition; outward physical boxes
cover rounding at the dyadic subdivision endpoints. The time budget is checked
between subdivisions, so it can overrun by the work of one pair of children.

### Faster certified nodes

`validated_universal.py` accelerates nodal certification. Newton iterations
only propose an anomaly. Acceptance requires opposite interval endpoint signs,
a positive interval derivative, and a global subluminal bound from periapsis
speed. The latter establishes uniqueness of the retarded root for the
noncollision conic. Entire Stumpff series have explicit geometric tail bounds.
The short-anomaly implementation refuses `abs(alpha*chi^2)>=1/2`; a validated
ODE provides the fallback. Zero angular momentum also uses that fallback.

This is a numerical certificate built from standard identities, not a claim
that Newton convergence proves accuracy. A zero-containing Arb ball must not
be raised through python-flint's generic real-power path; explicit products
and nonnegative norm enclosures handle that case.

## Applying the field

For exposure-associated modes `m`, the represented response is

```
F(q) = sum_m data[m] * exp(i * dot(frequency[m], position[visit[m]](q))).
```

Fixing three outer coordinates leaves a three-dimensional position polynomial.
Its linear phase is summed by a type-3 NUFFT. Polynomial corrections approximate
the quadratic phase. If `tau` bounds that quadratic phase, truncation at order
`Q` contributes at most `tau^(Q+1)/(Q+1)!` per unit input amplitude.

`phase_tolerance` chooses the cheapest order 0–4 whose analytic bound holds
everywhere in the six-dimensional cell. The guarantee has the form
`absolute error <= phase_tolerance * sum(abs(data))`; it is not a relative
error bound near cancellations. If the budget cannot be met, evaluation uses
the exact exponential of the position polynomial with per-exposure summation.
Omitting unnecessary corrections can be much cheaper than always using Q=1.

Repeated outer coordinates share slice construction in **both** the fast route
and the optimized comparison. Arbitrary six-dimensional queries instead use
the full retained position polynomial and per-frame NUFFTs. That route avoids
orbit integrations but still performs work proportional to queries times
exposures. Globally efficient synthesis in the outer coordinates is unresolved.

### What the bounds do not certify

Continuous geometry bounds and analytic phase-tail bounds do **not** certify
the complete fast numerical output. Normalized-query conversion, phase
coefficient arithmetic, and FINUFFT error remain separate. FINUFFT's requested
epsilon is not presented as an interval theorem. Selected scores can be
audited independently with validated physical dynamics and Arb trigonometric
sums, including arithmetic, at the cost of direct evaluation for those queries.

`ChartOperator` gives a consistent finite-query forward/adjoint pair using the
exact exponential of the retained polynomial. LSQR provides a regularized
Fourier-coefficient inverse. Nullspaces remain; arbitrary original images are
not uniquely reconstructible. Tests compare with a dense regularized solution
and an explicit duplicate-column nullspace.

## Finite detectors and real exposures

`finite_pixel_fields` models a finite odd square detector with unrenormalized,
pixel-integrated separable Gaussian PSFs. It supports distinct widths, masks,
known variances, flux scales, and an optional profiled constant background in
each exposure. It computes numerator and information fields; it does not
assume information is independent of source position.

Zero padding is computational. Artificial periodic copies and frequency
truncation have explicit Gaussian tail bounds over a declared source-position
box. The squared PSF and profiled-background overlap retain doubled Fourier
support. For template error `delta`, numerator error is bounded by
`delta * sum(W*abs(background-adjusted data))`; information error is bounded
by `delta*(2+delta)*sum(W)`, doubled when a constant background is profiled.
Flux scales enter linearly/quadratically. FFT/coefficient arithmetic is not
included in these analytic detector-model bounds.

Three archived real ZTF cutouts exercise this adapter. Their supplied variance
is an estimate, not an authoritative survey covariance product. Gaussian PSFs
replace the retained empirical survey PSFs. A local affine WCS is compared with
the actual TPV WCS at every tested query; that measured discrepancy is not a
continuous astrometric calibration certificate. Exposure trailing and spatial
PSF variation remain unmodeled. This is an operator correctness check on real
pixels, not a discovery, false-positive calibration, or completeness result.
Direct pixels are the appropriate strong baseline for this small stack.

## Installed command and library

```
pip install -e '.[research,accelerate]'
magictools-fold-chart manifest.json NEW_OUTPUT_DIRECTORY
```

The production `magictools fold_kepler` interface remains unchanged. The new
experimental library is also included in wheels under
`research.kepler_transform`; the optional research dependencies are required.

The manifest has:

```json
{
  "schema_version": 1,
  "arrays": "fields-and-queries.npz",
  "chart": {
    "observer": [1, 0, 0],
    "velocity": [0, 0.01720209895, 0],
    "normal": [1, 0, 0],
    "east": [0, 1, 0],
    "north": [0, 0, 1],
    "retarded_epoch": true
  },
  "center": [0.5, 0, 0, 0, 0, -0.0015],
  "widths": [0.05, 0.001, 0.0000096963, 0.0000484814, 0.0000484814, 0.0000096963],
  "position_tolerance_arcsec": 0.0005,
  "phase_tolerance": 0.0001,
  "outer_degree": 2,
  "translated_certificate": false,
  "max_leaves": 16,
  "max_seconds": 600,
  "allow_direct_fallback": false,
  "audit_count": 3
}
```

The NPZ contains `times[T]` in days relative to reference reception time zero,
`observers[T,3]` in heliocentric AU, integer `visits[M]`, tangent-plane
`frequencies[M,2]`, complex `data[M]`, and physical chart `parameters[N,6]`.
Optional `normal/east/north[T,3]` specify exposure tangent bases; otherwise
they are constructed from the central trajectory. Use `outer_degree:4` for
the higher-order construction. An omitted phase tolerance preserves Q=1.

The command saves the cover, certificates, complex query scores, input hashes
and selected full-Arb audits. It refuses queries in unresolved cells unless
direct fallback was explicitly enabled. It does not silently drop those
queries or shrink the requested domain. A refused evaluation exits with code 2
and keeps its coverage report.

## Prior work and remaining objective

Observer-relative/inverse-distance orbital coordinates and universal variables
are established techniques. Relevant primary sources include
[Bernstein & Khushalani, orbit fitting](https://arxiv.org/abs/astro-ph/0008348),
[Wisdom & Hernandez, a universal Kepler solver](https://arxiv.org/abs/1508.02699),
and [Barnett et al., FINUFFT](https://arxiv.org/abs/1808.06736).
The present certified solver uses a bounded Stumpff series; it does not claim
to reproduce Wisdom & Hernandez's series-free implementation.

The full target still requires economical outer-coordinate synthesis, complete
fast numerical error control, general exposure models, and convincing costs
on larger realistic survey workloads. Resolving these bounded geometry cases
is evidence of progress, not proof that the unrestricted target is possible.
