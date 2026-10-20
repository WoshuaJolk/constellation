# `fold_kepler`: shared orbit sums with bounded screening

`fold_kepler` searches a **finite, specified trajectory bank** using FFT image
filters, shared partial sums, and analytic bounds. Every candidate that could
cross the detection threshold or enter the requested top list is evaluated at
its original continuous coordinates. Cell rounding affects screening efficiency,
not the final reported template positions. This is a restricted implementation
of the catalogue idea, using established numerical/statistical ingredients;
no new general Kepler transform or mathematical novelty is asserted.

## Run it

```bash
constellation fold_demo fold-demo-output
# Larger, 64,000-orbit benchmark (48 images):
constellation fold_demo fold-large-output --grid-size 40 --frames 48
constellation fold_kepler fold-large-output/input.json -o checked-search.json
```

The demo saves synthetic FITS images, an explicit elliptic orbit bank, the input
manifest, full result, and summary. A seed fixes the science results; timings
depend on hardware and load. It includes an illustrative circular Earth observer,
8 days of observations, varying flux scales, and a masked detector gap. The source
is injected using the same PSF model being searched. This tests the transform,
not robustness to model mismatch or real-survey completeness.

On the development host, the 64,000-orbit / 48-image test gave:

| Quantity | Result |
| --- | --- |
| Direct track/frame pairs | 3,072,000 |
| Distinct cell leaves / shared subpath additions | 178,354 / 672,710 |
| Precisely evaluated / screened tracks | 5,184 / 58,816 |
| Median direct evaluation, three warmed runs | 0.737 s |
| Median folded evaluation, including refinement | 0.205 s |
| Evaluation speedup | **3.60×** |
| Plan compilation | 4.59 s |
| Evaluations to amortize compilation at those timings | 9 |
| Single-use total speedup, including preprocessing and propagation | **0.50× (slower)** |
| Direct versus folded detections and top ten | Identical |
| Maximum difference in refined S/N | 0 in this run |
| All computed S/N bounds contain direct S/N | Yes |

The injected orbit ranked first, with S/N 24.41. There were 719 passing *orbit
hypotheses* around the injected source, not 719 objects. These correlated
hypotheses need grouping and orbit inference before any discovery interpretation.
The source's injected flux was 55; the best fit was 63.07 ± 2.58 in this noise
realization. The result is a synthetic test, not a claimed observation.

See [the saved benchmark summary](../examples/fold-benchmark.json) for timings
and software versions. `benchmark_repeats` times both methods on the **same FFT
maps**, and includes refinement in the folded method. It reports each sample
and the ratio of medians. The direct baseline is vectorized, not a Python
pixel loop. No speed threshold is asserted in CI.

## Input

Use `schema_version: 1`, `exposures`, and exactly one of `trajectories` or `orbits`.
Paths are relative to the manifest. All images and the manifest are hashed.

```json
{
  "schema_version": 1,
  "exposures": [
    {"id": "first", "image": "first.fits", "variance": 16,
     "background": 100, "psf_sigma_pixels": 1.2, "flux_scale": 1},
    {"id": "second", "image": "second.fits", "variance": 16,
     "background": 100, "psf_sigma_pixels": 1.2, "flux_scale": 0.9}
  ],
  "trajectories": [
    {"id": "orbit-a", "xy": [[21.1, 30.7], [23.6, 29.8]]},
    {"id": "orbit-b", "xy": [[21.3, 30.8], [24.2, 29.7]]}
  ],
  "cell_subdivisions": 1,
  "alpha": 0.01,
  "top_k": 2,
  "validate_direct": true,
  "benchmark_repeats": 0
}
```

- Each `xy` row follows **exposure order**, with zero-based continuous pixel
  centers. All entries must be finite. Centers with `0 <= x < width-1` and
  `0 <= y < height-1` are evaluated; centers outside that interpolation domain
  contribute zero information. Partial PSFs at included detector edges retain
  their original normalization. The last row/column of centers is deliberately
  excluded so each interpolation cell has four anchor centers. Pixels there
  still contribute to PSFs centered in valid cells.
- `variance` is a positive scalar or FITS path. Optional `image_hdu`,
  `variance_hdu`, `mask`, and `mask_hdu` select planes; nonzero mask pixels and
  nonfinite/nonpositive-variance pixels contribute zero weight. Duplicate
  image/HDU references are rejected. Copies in other files remain the caller's
  responsibility.
- **`background` is required and treated as known**, in image units. This is
  not the fitted local background used by `force_track`/`dig_floor`. Background
  estimation errors and uncertainty are not included in the null distribution.
- Specify exactly one of `psf_sigma_pixels` (0.2–10) or `psf_image`, with optional
  `psf_hdu`. Gaussian mode first creates a centered, finite, pixel-integrated
  kernel, extending to ceil(5 sigma). The kernel is normalized once, then
  translated by bilinear interpolation. Empirical negative wings are retained.
  At fractional positions this differs from directly integrating an optical
  Gaussian: **the interpolation model is part of the stated likelihood**.
- `flux_scale` is positive, known, and defaults to 1. A source has one common
  reference flux. No source shot noise, flux variability, spatial PSF variation,
  trailing, covariance between pixels/exposures, or unknown background is fitted.
- `cell_subdivisions` is 1, 2, 4, 8 or 16. Smaller cells tighten bounds but reduce
  sharing. The default usually works best for dense banks. `top_k` is 1 through
  min(bank size,1000), default min(bank size,10). Ties retain input order.
- `validate_direct: true` evaluates the whole bank using the same model and
  raises an error on a bound violation or different detections/top list.
  `benchmark_repeats: 0..9` additionally repeats validated timings on the
  compiled bank; the demo uses 3.

For Kepler-generated positions, replace `trajectories` with:

```json
"epoch_mjd_tdb": 60000,
"orbits": [
  {"id": "example", "a_au": 2.6, "e": 0.12,
   "inclination_deg": 10, "node_deg": 80,
   "periapsis_deg": 30, "mean_anomaly_deg": 40}
]
```

Each exposure then needs `mjd_tdb`, a three-vector
`observer_heliocentric_au`, and a celestial FITS WCS. **The orbital reference
plane is the ICRS equator, not the ecliptic.** `periapsis_deg` is the argument of
periapsis; `mean_anomaly_deg` is at the common epoch. The generator solves the
elliptic two-body Kepler equation with the Sun's GM and iterates emission time;
the supplied observer is at reception. It omits planets, aberration, relativity,
and atmospheric refraction. Limits: 0.05–1000 AU semimajor axis, 0–0.95
eccentricity, 0–180° inclination, other angles within ±360°, time offsets within
100 years. These bounds are software limits, not accuracy promises.

You may instead obtain more accurate trajectories from `map_cover` or another
propagator and supply their pixel coordinates. Convert `map_cover`'s per-orbit,
per-exposure midpoint `x,y` values to the ordered `xy` arrays above, and supply
explicit backgrounds. This version does not model trails; do not ignore
significant exposure motion when making that conversion. Nonfinite WCS
projections are errors rather than silently discarded bank members.

## Derivation and what is bounded

For frame j, let d be the image minus known background, W its diagonal inverse
variance with masked weights zero, and p(x,y) the unit-reference-flux template
(including `flux_scale`). The sufficient statistics are

```
N_j(x,y) = p(x,y)^T W d
I_j(x,y) = p(x,y)^T W p(x,y)
N = sum_j N_j, I = sum_j I_j
log L(F) - log L(0) = F N - (F^2 / 2) I
F_hat = N/I, sigma_F = 1/sqrt(I), Z = N/sqrt(I)
```

In one interpolation cell, p is a convex bilinear combination of four integer
anchor templates. FFT correlations compute all anchor numerators and ten distinct
entries of their 4×4 Gram matrices, including masks and varying variances.
This is an exact algebraic factorization of the **chosen bilinear model**.

Restrict the cell to a smaller subcell if requested. Write its four corner
templates as v00, v10, v01, v11. The numerator is bilinear, so its extrema are
bounded by the four corner numerators. Information is biquadratic. Its nine
degree-(2,2) Bernstein coefficients are the four squared corner norms, the four
edge inner products, and

```
(<v00,v11>_W + <v10,v01>_W) / 2.
```

Bernstein basis functions are nonnegative and sum to one, so the information
lies between the smallest and largest coefficients (with zero as an additional
physical lower limit). This also works with signed empirical PSFs.

Sum the per-frame intervals to bound N and I for a trajectory. For a positive
numerator, the S/N upper bound uses the information lower bound; for a negative
numerator it uses the information upper bound. The lower-bound cases reverse.
Zero information lower bounds can yield an infinite upper bound; those
trajectories are retained for precise evaluation.

The shared transform uses a balanced tree over exposures. Within each node,
identical pairs of child subpaths share a single interval sum. Leaf nodes share
the bound computation for an exposure/subcell. Thus nearby tracks can reuse
short segments even when their full tracks differ.

Let L_k be the kth largest S/N lower bound, and z_alpha the Bonferroni threshold
for **every supplied trajectory**, including duplicates and off-chip members.
Refine every track with upper bound >= min(L_k, z_alpha). A pruned track can
neither reach the threshold nor displace a top-k member. Retaining equality
preserves ties. Under exact arithmetic this proves identical detection sets and
top-k results to evaluating the entire continuous finite bank.

**Numerical qualification:** this implementation uses float64 FFTs and sums,
with 1e-10 map-scale padding on local bounds. That is a numerical safeguard,
not a formal outward-rounded interval-arithmetic certificate. Independent pixel
likelihood tests, random subcell tests, and the optional whole-bank check verify
the implementation numerically. They do not prove an arbitrary floating-point
error bound. Extremely ill-conditioned inputs need rescaling or a future
verified-arithmetic implementation.

Empty PSF footprints are kept at exactly zero information. When FFT information
is very small, the implementation recomputes that footprint directly in pixel
space so numerical leakage cannot create a spurious fit; small *real*
information is retained.

## Reuse and complexity

For N orbits and T images, compilation and orbit projection still inspect O(NT)
positions. Compilation uses sorting to identify shared paths, approximately
O(NT log N) work; it is often slower than one direct search. FFT preprocessing
costs O(T P log P) for P pixels per frame. Folded evaluation ordinarily costs
O(L+U+RT+N+R log R), including sorting refined candidates,
where L is the number of distinct exposure/cells, U the number of shared
internal subpaths, and R the number of refined tracks. In unfavorable cases
these are O(NT); screening can even be slower than direct evaluation. The
ill-conditioned-footprint fallback additionally reads the affected PSF pixels.

The reusable Python API avoids recompilation:

```python
from magictools.folding import FilterFrame, FoldPlan, search_fold

# positions has shape (bank_size, frame_count, 2).
plan = FoldPlan(positions, image_shapes, subdivisions=1)
# For each new control/injection image set with exactly the same geometry:
frames = [FilterFrame(image, variance, kernel, background, mask, scale)
          for image, variance, kernel, background, mask, scale in image_inputs]
result = search_fold(frames, plan, ids=orbit_ids, top_k=10, validate_direct=True)
```

Image shapes are checked on reuse. **The caller must also preserve the pixel
trajectory geometry**: changing WCS, observing times or observer positions
requires a new plan. Kernels, noise, masks, backgrounds and image values can
change: the newly constructed filter frames supply new statistics and bounds.
The CLI currently recompiles each invocation; reusable plans live in the Python
API, not an on-disk cache.

Limits are 100,000 tracks, 256 frames, 4 million track/frame pairs, and 2 million
total input image pixels. Limits fail explicitly; banks are never truncated.

This removes unnecessary work for shared finite banks and retains the specified
detector's sensitivity within the chosen model. It does **not** guarantee
continuous-orbit coverage, physical PSF accuracy, robustness to artifacts, or
uniform sub-trajectory cost across all Keplerian orbits. The Gaussian false-alarm
bound requires a bank fixed independently of the image noise and correct known
noise/background models. It is not a probability of astronomical reality.
