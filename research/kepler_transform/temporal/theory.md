# Time compression without an orbit grid

This construction changes the representation. It does not reduce the order of
the six-dimensional orbit tensor: it removes that tensor. The output is a sum
of `p` two-dimensional Fourier fields pulled back along the exact Kepler flow
at `p` certified time nodes. It represents every orbit in the declared box,
before any query orbits are supplied.

It is useful when the exposure count greatly exceeds the sufficient time order.
It is not an economical representation for every cadence, bandwidth or domain.
No new mathematical identity, general FKT, stable inverse, or archive solution
is claimed.

## Operator and input convention

Let `q` lie anywhere in the original six-dimensional box `B`, and let
`theta(q,t)` be its apparent tangent-plane coordinates, including the existing
all-conic Kepler propagation, circular observer motion and light-time model.
For `N` exposures at arbitrary times in `[-H,H]`, the forward score is

```
F(q) = sum_j sum_k c[j,k] exp(i k . theta(q,t_j)).
```

There are `K = nx*ny` modes on a common rectangular Fourier grid. Each coordinate
of `k` has magnitude at most the certified cutoff, one inverse arcsecond.
For sampled input images the definition is their *periodic trigonometric
interpolant*: `c = fftshift(fft2(image))/K`, with pixel positions at integer
fractions of the Fourier period. The entire input FFT is charged.

This convention is exact for the declared finite-dimensional input model.
It does not establish that arbitrary archival images, WCS mappings, varying
PSFs, variance maps, masks, or finite-duration exposure trails can be converted
to that model without error or additional cost. An arbitrary Gaussian
likelihood may require multiple linear score fields and further nonlinear
operations; this code implements the linear operator above.

## Remove the common reference motion

Take the fixed center orbit `q0` of `B`. Define

```
R(q,t,k) = exp(i k . [theta(q,t) - theta(q0,t)])
b[j,k]   = c[j,k] exp(i k . theta(q0,t_j)).
```

For every real exposure time the modulation has unit modulus. Interpolate
`R` in **time only**, at `p` first-kind Chebyshev nodes `tau_l` in `[-H,H]`.
With their cardinal polynomials `L_l`,

```
R(q,t,k) ~= sum_l L_l(t/H) R(q,tau_l,k)
g[l,k]   = sum_j L_l(t_j/H) b[j,k]
Ftilde(q)= sum_l sum_k g[l,k] R(q,tau_l,k).
```

Compression constructs `g` without knowing `q`. Its size is `pK`, independent
of any grid in orbit space, any output orbit-resolution count, and the number
of queries. A query uses `p` orbit propagations and evaluations of ordinary
two-dimensional Fourier fields. Querying `Q` values necessarily costs at
least `Omega(Q)` simply to return them.

The source data may vary arbitrarily between exposures. We interpolate the
**kernel**, not the image sequence; no temporal smoothness of pixel values or
low rank of the image data is assumed.

## Uniform error over the whole orbit domain

The existing complex-root certificate establishes a reference-connected
holomorphic Kepler/light-time branch on an ellipse in normalized time and the
entire real orbit box. Its interval derivatives are with respect to the
original normalized orbit coordinates, including when analysis subdivisions
are needed. The enclosing derivative union covers every subdivision.

Write `rho > 1` for the ellipse parameter. Along the real line segment from
`q0` to any `q` in `B`, each normalized coordinate displacement is at most one.
Thus

```
D = Kmax * sum_(angular axis a, orbit axis s)
              sup_(q in B, t in ellipse) |Im d theta_a / d q_s|

|R(q,t,k)| <= exp(D).
```

The imaginary part is taken before the absolute value: the integral runs in
real orbit coordinates at a fixed complex time. The line segment stays in the
box. The interval proof also verifies the analytic branch and tangent-plane
denominators; bounded derivatives alone would not suffice.

A Chebyshev truncation of degree `p-1` has uniform error at most
`2 exp(D) rho^(-(p-1))/(rho-1)`. Interpolating that truncation is exact.
The interpolation operator has norm at most `sqrt(2p-1)`: discrete cosine
orthogonality gives a cardinal-vector squared norm at most `(2p-1)/p`,
and Cauchy-Schwarz bounds its l1 norm. Therefore

```
epsilon_p =
  2 (1 + sqrt(2p-1)) exp(D) rho^(-(p-1)) / (rho-1)

sup_(q,t,k) |R - I_p R| <= epsilon_p
sup_(q in B) |F(q) - Ftilde(q)| <= epsilon_p sum_(j,k) |c[j,k]|.
```

The implementation chooses `p` with outward Arb arithmetic. Fresh proofs at
the unchanged original physical bandwidth yield:

| Full orbit domain | Time span | Ellipse rho | Sufficient p | Uniform kernel bound |
| --- | ---: | ---: | ---: | ---: |
| Near | 8 days | 2 | 82 | < 8.95e-9 |
| Distant | 30 days | 2 | 127 | < 6.32e-9 |
| Broad near stress | 8 days | 1.5 | 490 | < 7.70e-9 |

These are sufficient orders, not minimum ranks. They do not depend on exposure
count or query count. They do depend on time span, bandwidth, orbit-box extent,
analyticity margins and error tolerance. For a fixed admissible ellipse,
`p = O((D + log(1/epsilon) + log p)/log rho)`. Treating `p` as a universal
constant while increasing those physical parameters would be unjustified.

The certificate is an exact-arithmetic interpolation result. FFT/NUFFT requested
tolerances and sampled discrepancies do not certify floating-point output.
If implementation errors contribute `Delta g` to the compressed coefficients,
their score contribution is at most `||Delta g||_1`, since `|R|=1` for real
queries; output Fourier evaluation and orbit errors must be added separately.
These latter numerical bounds are not yet interval certified.

## Compress the input efficiently

Dense compression costs `O(NpK)` and stores an `N*p` interpolation matrix.
The second backend uses the existing Fourier/Chebyshev moment identity:

```
alpha_j = arccos(t_j/H)
S[r,k] = sum_j b[j,k] cos(r alpha_j),  0 <= r < p
g[l,k] = (S[0,k] + 2 sum_(r=1)^(p-1) S[r,k] cos(r alpha_l))/p.
```

One batched type-1 NUFFT at the reflected points `+alpha_j,-alpha_j`, followed
by a DCT-III, computes the same cardinal sums to a numerical tolerance. This
does not introduce an additional physical approximation. The code retains a
dense backend, checks both against one another, and provides an algebraic
adjoint. The adjoint is not an inverse.

For output access, each `g[l,:]` is an ordinary two-dimensional Fourier series.
Type-2 NUFFTs evaluate it at
`theta(q,tau_l)-theta(q0,tau_l)` for all requested orbits. The definitions and
the distinction between adjoint and inverse are given in the
[FINUFFT mathematical documentation](https://finufft.readthedocs.io/en/latest/math.html).

## Total costs, including construction and access

Let `w` be the spreading width required by the NUFFT numerical tolerance, and
let `g` bound the numerical cost of one orbit/light-time solution on this
certified domain. Let `Ccert` be the cost of the actual interval proof, including
all attempted enclosures and analysis subdivisions. The implementation records
that cost; no uniform bound on it over arbitrary orbit domains is claimed.
For the implemented bounded proof search, subdivision depth is capped at six,
so there are at most 127 root-enclosure attempts. Each has at most eight
nominal Newton proposals and 16 enclosure iterations, using the existing
fixed-size jets and 16-term Stumpff expansions at 128-bit Arb precision.
The order search makes at most 511 proposals. Thus proof construction is
bounded independently of `N`, `K` and `Q` for these fixed domains and cutoff;
failure to prove a domain within those caps is explicit, not a hidden fallback.
The numerical orbit solver likewise caps its bracket, Newton and light-time
iterations and refuses failed queries. Changing the physical domain or required
precision can change these constants or make the fixed budgets fail.

Using the NUFFT compression backend, a first input-to-output call has the
conditional arithmetic bound

```
Ccert
+ O((N+p)g)                         reference-orbit geometry
+ O(NK log K)                       input image FFTs
+ O(NK)                             reference modulation
+ O(K[Nw + p log p])                time moments and DCTs
+ O(pQg)                            arbitrary-query geometry
+ O(p[K log K + Qw^2])              output Fourier fields
+ O(Q)                              output reduction.
```

Moment and output NUFFT planning/set-point work is included in the corresponding
terms. The software stores the required reference modulation, reflected points,
fields, and optional query plans. At fixed tolerance its non-dense storage is
`O(NK + pK + pQ + N)` up to NUFFT workspaces and oversampling constants, plus
the proof record. It has no `QNK` response tensor and no six-dimensional
orbit tensor. The implementation materializes inputs; a streaming input
implementation is not claimed.

Dense time compression replaces the moment term by `O(NpK)` and adds `O(Np)`
storage. Dense output caching instead costs/stores `O(pQK)` and is sometimes
faster for very small Fourier images. All such costs are explicit alternatives,
not hidden behind the NUFFT complexity statement.

Repeated inputs on fixed geometry reuse both methods' plans. New query points
require fresh query geometry and set-point work; retaining `g` does not make
query access free. At fixed domain, bandwidth and tolerance, the fast backend
can separate the large exposure and query counts into input and output costs
instead of multiplying them.

## The fair direct alternatives

The comparison must include all of:

1. Direct orbit solutions at all `NQ` observation/query pairs and direct Fourier
   sums.
2. Cached direct sky coordinates, avoiding repeated orbit solutions.
3. Cached direct type-2 NUFFT plans, with application
   `O(N[K log K + Qw^2])`.
4. A fully cached direct coefficient-response matrix, when it fits the declared
   budget: construction `O(NQg + NQK)`, storage/application `O(NQK)`.

The same Fourier evaluator implementation serves both the compressed and direct
methods. Both receive identical coefficient inputs, precision requests, thread
limits and cache budgets. Repeated images still undergo their FFT and, for the
compressed method, time compression. No method substitutes a previously
computed output vector for a new application.

For `N >> p` and sufficiently many queries, there is a concrete avenue to a
cost advantage, including against cached direct NUFFTs. For `N <= p`, direct
evaluation is generally the sensible choice. This is a dense-cadence
compression result, not a general assertion that fewer time samples make every
astronomical archive tractable. It does not solve the search over all orbit
maxima, arbitrary input preparation, numerical certification, or stable inversion.

The relationship to earlier work is straightforward: this retains the
demodulated source-kernel approximation from the earlier construction, but
uses time alone and preserves the common spatial Fourier grid. Generalized
Radon butterfly methods motivate sharing oscillatory-kernel work; their
complexity theorems do not automatically supply the bound above. See
[Hu, Fomel, Demanet and Ying](https://math.mit.edu/icg/papers/hyp-rad.pdf).
