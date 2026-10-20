# From finite detector pixels to time-compressed likelihoods

This extends the common-grid time representation to two finite-detector
likelihood statistics. It does not certify arbitrary instrumental PSFs,
distorted WCS models, real ephemerides, correlated noise or floating-point
arithmetic. Its exact model and separate error contributions are specified
below.

## Keep the original pixels and their noise model

Exposure `j` has finite pixels `r`, supplied independent variances `v_jr`,
mask weights `w_jr = 0` or `1/v_jr`, flux scale `s_j`, and a pixel-integrated
separable Gaussian PSF with widths `(sigma_x,sigma_y)` in detector pixels.
The declared affine map into the common angular plane is

```
angular position = offset_j + B_j * pixel position.
```

`B_j` may rotate, shear, reflect and change scale between exposures. Images
are never resampled. Therefore no interpolation-induced pixel covariance is
introduced or silently discarded.

Let `T_jr(x)` be the fraction of a unit source at angular position `x` falling
in pixel `r`. The unprofiled sufficient statistics are

```
N_j(x) = s_j sum_r w_jr d_jr T_jr(x)
I_j(x) = s_j^2 sum_r w_jr T_jr(x)^2.
```

To profile a separate constant background in each exposure, subtract the
weighted image mean in `N_j` and subtract
`s_j^2 (sum_r w_jr T_jr(x))^2 / sum_r w_jr` from `I_j`. An entirely masked
exposure contributes zero.

Summing along an orbit gives `N(q), I(q)` and the Gaussian log-likelihood
ratio `F N(q) - F^2 I(q)/2`. For positive information, the unconstrained
fitted flux is `N/I` and the signed matched-filter S/N is `N/sqrt(I)`.
These statistics do not by themselves calibrate false alarms or discoveries.

## A common Fourier grid without a periodic-camera assumption

Use a common angular period `P` solely as a computational construction.
For integer lattice indices `k`, let `omega = 2*pi*k/P`. The Fourier
coefficient of the periodized pixel-integrated PSF is

```
A_j(k) = |det B_j| / P^2
         exp(-0.5 sum_a sigma_ja^2 (B_j^T omega)_a^2)
         prod_a sinc((B_j^T omega)_a / (2*pi)).
```

A pixel centered at `r` multiplies this by
`exp(-i omega . (offset_j + B_j r))`. Thus a two-dimensional type-1 NUFFT
of the original weighted pixels gives the numerator coefficients directly.
The same sum of weights and the convolution `A_j*A_j` give the information
coefficients. The profiled-background correction is a convolution of the
weighted-overlap coefficients. All of these fields lie on the same angular
frequency lattice, despite their differing detector coordinates.

This is the ordinary nonuniform Fourier-sum definition documented by
[FINUFFT](https://finufft.readthedocs.io/en/latest/math.html).
The physical model is a finite image with a nonperiodic Gaussian template;
the artificial periodic copies are bounded rather than equated with the
physical detector.

## Uniform finite-detector approximation bound

Let `Sigma = B diag(sigma^2) B^T`, with interval-certified lower and upper
eigenvalue bounds `lambda_min, lambda_max`. Arb arithmetic evaluates the
two-by-two eigenvalue formula and determinant. Reject a singular map or a
nonpositive covariance bound.

For a source square `|x_a| <= H` and an `n*n` stamp, every point within every
pixel satisfies

```
|source_a - pixel_point_a| <= D_a
D_a = H + |offset_a| + (n/2) sum_b |B_ab|.
```

Choose `P > max_a D_a`. With `g_a=P-D_a`, bound the one-dimensional sum of
nonzero lattice copies by

```
U_a = 2 exp(-g_a^2/(2 lambda_max))
      / [1 - exp(-(P^2 + 2 P g_a)/(2 lambda_max))].
```

The rectangular two-dimensional wrapped-copy template error is at most

```
delta_wrap <= (U_x + U_y + U_x U_y)/(2*pi*sigma_x*sigma_y).
```

This follows by bounding the Gaussian with its largest covariance eigenvalue
and summing a geometric upper bound for successive lattice copies. The pixel
integration area gives `|det B|/sqrt(det Sigma)=1/(sigma_x sigma_y)`.

For a retained template-frequency square `|k_a|<=m`, define

```
a = lambda_min (2*pi/P)^2 / 2
S <= 1 + sqrt(pi/a)
T <= sqrt(pi/a) erfc(m sqrt(a)).
```

`S` bounds the complete one-dimensional Gaussian lattice sum; `T` bounds its
discarded tails by the integral of a decreasing Gaussian. Dropping sinc factors
only enlarges absolute values. The discarded two-dimensional coefficients have
l1 norm at most

```
delta_Fourier <= |det B|/P^2 * 2 S T.
```

The uniform template error is `delta=delta_wrap+delta_Fourier`. Since a
pixel-integrated Gaussian is in `[0,1]`, this implies

```
|Delta N_j| <= s_j delta sum_r |w_jr (d_jr-weighted_mean_j)|
|Delta I_j| <= s_j^2 delta(2+delta) sum_r w_jr
```

with an extra factor two in the information bound when background profiling
is enabled. That extra term bounds the error in the squared weighted overlap.
All bounds apply to every source position in the declared square.

The finite Fourier arithmetic and calculation of the profiled mean are
floating-point operations. Their rounding errors are separate from these
exact-model approximation bounds.

## Remove negligible modes, then use the actual bandwidth

For already constructed coefficient fields, removing any set of modes changes
the response by at most their coefficient l1 norm at **every** real sky
position. `trim_fields` finds a smaller square under separate numerator and
information budgets. Its bound on stored coefficients uses
`|Re c|+|Im c|` and an outward nonnegative-summation bound. This does not certify
how those coefficients were computed.

The numerator can also be prepared directly on a smaller grid. Its cutoff is
selected from the analytic template bound and weighted pixel values.
`prepare_numerator` computes only those modes; it does not construct a large
information grid and discard it on every changed image. A changed image must
pass its own error budget or the existing frequency/time plan is refused.

Numerator and information need not use the same cutoff or time order. Each
gets a fresh interval time-interpolation certificate at its **actual retained
spatial bandwidth**, using the complete original orbit box. The former
one-inverse-arcsecond certificate is never reused at a larger frequency.
Sufficient orders above 512 are supported up to an explicit allocation/search
limit; exceeding a budget fails without reducing the physical domain.

The error contributions are added:

```
total numerator error =
  finite-template approximation + optional discarded modes
  + time-kernel bound * numerator coefficient l1

total information error =
  finite-template approximation + discarded modes
  + time-kernel bound * information coefficient l1.
```

The implementation propagates these to S/N and fitted-flux intervals by
checking all endpoint combinations. If the information interval reaches zero,
the score is marked unresolved. It is not clipped to a positive denominator
and presented as a detection. The interval endpoint computations themselves
are not outward-rounded certificates.

## Which geometry is covered?

For the dense controlled experiment, the observation model and original
six-dimensional near-orbit box are unchanged. Each stamp points at the
mathematical reference orbit. The real-time interval derivatives bound
`theta(q,t)-theta(q0,t)` across the entire box and all eight days. This supplies
the detector's source square before any query orbits are selected. Different
affine pixel maps and PSFs are permitted at individual exposures.

The real ZTF replay uses the recorded Earth/site ephemerides and actual TPV
WCS to generate and check positions. Its Fourier adapter uses local affine
maps into a common tangent plane; discrepancies against the full TPV WCS are
measured independently. Astropy supplies the FITS WCS transformations; see
[the WCS interface documentation](https://docs.astropy.org/en/stable/wcs/wcsapi.html).
Those sampled discrepancies are not a continuous WCS-distortion bound.

The circular-observer time proof is **not** applied to the real ZTF stack.
The real experiment validates the detector adapter; the controlled dense
experiment validates its composition with certified time compression. Closing
that distinction requires a validated real-ephemeris/pointing model.

## Complete work and fair comparisons

Let `N` be exposure count, `Ppix` the original pixels per exposure, `L` the
full Fourier-field mode count, `K_N,K_I` the retained mode counts, and
`p_N,p_I` the certified time orders. The initial detector preparation costs
`O(N[Ppix*w^2 + L log L])` for NUFFT spreading and transforms/convolutions,
plus the explicit per-exposure bound and cutoff calculations. The implemented
coefficient-trimming search scans arrays `O(log sqrt(L))` times and costs
`O(NL log L)`. It materializes two full fields before trimming; its peak
storage includes those arrays and scan temporaries.

Dense time compression costs `O(N[p_N K_N+p_I K_I])`. The optional fast
Chebyshev-moment backend uses the bound derived in the preceding
[time-compression theory](../temporal/theory.md). Arbitrary queries charge
both sets of orbit solves and output NUFFTs:

```
O((p_N+p_I) Q g
  + p_N[K_N log K_N + Qw^2]
  + p_I[K_I log K_I + Qw^2]).
```

No orbit tensor is constructed. Every elapsed phase, including detector
preparation, source-domain certification, bandwidth selection and both time
plans, is included in the first-use comparison.

The direct baselines operate in the original pixels:

* Independent pixel integrals provide the numerical reference.
* An optimized separable evaluation avoids allocating query-by-image tensors.
* A factor cache stores the one-dimensional Gaussian integrals in x and y,
  using `O(NQn)` storage for `n*n` pixels, and `O(NQn^2)` application work.
* A full pixel-response matrix, when it fits the stated budget, uses
  `O(NQn^2)` storage/application and folds in masks and background profiling.

For changed image values with fixed variance, mask, calibration and PSF,
information is independent of the data. **Both routes reuse it.** The
compressed route still prepares the new numerator, checks its error budget
and compresses it; direct caches multiply the new original pixel values.
This is a stronger and more relevant baseline than comparing only Fourier
evaluation methods.
