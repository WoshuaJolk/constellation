# `prove_floor`: a precise finite-family detection problem

This tool implements an analytic detection-power bound, a necessary-flux lower bound, and a likelihood-ratio detector for a specified finite template family in known Gaussian noise. It uses established statistical theory. It does not claim a new universal detection theorem, a fast all-orbit transform, or a proof that the astronomical archive excludes a population.

The distinction from a five-sigma sensitivity calculation is important: the bound applies to **every level-alpha detector in the specified statistical experiment**, averaged over the supplied trajectory prior. Template overlap and correlated noise enter the mathematics explicitly.

## Statistical experiment

Before any nuisance projection, pixels follow

```
H0: y ~ N(0, C)
H1: choose h with probability w_h; y ~ N(F t_h, C).
```

`F` is a fixed nonnegative flux, `t_h` is a fixed unit-flux pixel template, and `C` is a **known, positive-definite, flux-independent** noise covariance. Weights are nonnegative and normalized. The target is prior-average power, not a guarantee for every orbit. All templates, priors, masks, and noise parameters must be specified independently of the tested pixels.

Optional fixed background columns are first whitened and projected out. The following claims then apply to the **projected-data experiment**. They do not assert that no method using extra background information could do better. A source lying in the removed background subspace is unidentifiable in this experiment. Unknown noise covariance, fitted PSF parameters, Poisson signal-dependent noise, artifacts, and arbitrary adaptive template selection require a different analysis.

Write the whitened/projected templates as `u_h`, and define their Gram matrix

```
K_ij = u_i dot u_j.
```

This matrix includes the covariance of the matched-filter scores across trajectories. Treating overlapping templates as independent changes the experiment and may give invalid results.

## Derivation of the power bound

The likelihood ratio for the prior-mixture alternative is

```
L(y) = sum_h w_h exp(F * s_h - F^2 * K_hh / 2),
s_h = u_h dot whitened_projected_y.
```

Under the null, the score vector is Gaussian with covariance `K`, including when `K` is singular. Its moment-generating function gives

```
E0[L]   = 1
E0[L^2] = sum_ij w_i w_j exp(F^2 K_ij)
D       = chi-square(P1 || P0) = E0[L^2] - 1.
```

Let `phi(y)` be any test with values in [0,1] and null rejection probability alpha. Tests with smaller size can be randomized up to alpha without decreasing power. Then

```
power - alpha = E0[(L - 1) * (phi - alpha)]
              <= sqrt(D * alpha * (1 - alpha)).
```

The inequality follows from Cauchy–Schwarz and `Var0(phi) <= alpha(1-alpha)`. Separately, total variation is at most `sqrt(D)/2`, giving `power <= alpha + sqrt(D)/2`.

An oracle that knows the true trajectory has power

```
oracle_power = sum_h w_h Phi(F sqrt(K_hh) - Phi^-1(1-alpha)).
```

Each mixture detector is also a level-alpha test against each fixed component, so its prior-average power cannot exceed this oracle average. The tool reports the minimum of the two divergence bounds, the oracle bound, and one.

The divergence is monotone in nonnegative flux even if some `K_ij` are negative: expanding the exponential makes each coefficient a squared norm of the weighted template tensor moment. The oracle bound is also monotone. This permits a numerical root search for the smallest flux at which the upper bound allows a chosen target power. The result is a **necessary** flux; it is not a sufficient threshold or a claim that the upper bound is attained. If too much prior mass lies on zero-information templates, a target power may be unreachable at any finite flux.

All matrix operations and bounds are evaluated in floating-point arithmetic. This is a documented analytic derivation with numerical checks, not an interval-arithmetic proof certificate. A general mixture bound can be loose. The single-template oracle bound is exact and is included as a test case.

## Detector and finite calibration

By the Neyman–Pearson lemma, thresholding `L` is most powerful for this simple null and specified mixture alternative at a fixed test size. It optimizes average power under these weights; a different prior or fixed flux defines a different optimal detector. A finite brightness prior can be encoded by scaling templates and assigning weights to the combined trajectory/brightness bank.

The software calibrates a conservative threshold using an independent null simulation. With `n` calibration draws and requested failure probability `delta`, choose the smallest order statistic index `k` such that

```
P[Binomial(n, 1-alpha) <= k-1] >= 1-delta.
```

Set the threshold to the kth smallest calibration log-likelihood ratio and reject only for a **strictly larger** value. With probability at least `1-delta` over calibration draws, the resulting false-alarm probability is at most alpha under the stated Gaussian model. The reported upper tolerance bound is a Beta quantile. If too few calibration draws support the requested alpha/confidence, the command fails instead of making an unsupported guarantee. Ties produce a conservative test; no-information templates may produce no rejections.

Separate random streams generate null holdout draws and prior-mixture holdout draws. Their rejection fractions are reported with exact binomial 95% intervals. These are Monte Carlo estimates conditional on the calibrated threshold. Calibration confidence and holdout interval coverage are separate statements; they are not a single joint 95% claim. As calibration size grows for a continuous statistic, the conservative threshold approaches the size-alpha threshold. The finite-calibration detector need not attain the exact Neyman–Pearson power.

## Input and output

Minimal Gram-matrix input:

```json
{
  "schema_version": 1,
  "gram_matrix": [[1.0, 0.3], [0.3, 0.8]],
  "prior_weights": [1, 1],
  "flux": 2.0,
  "alpha": 0.01,
  "target_power": 0.9,
  "calibration_samples": 4096,
  "evaluation_samples": 4096,
  "calibration_failure_probability": 0.05,
  "seed": 2026
}
```

Alternatively, replace `gram_matrix` with `templates` (hypotheses × pixels) and exactly one of `variance` (positive per-pixel variances) or `covariance` (full pixels × pixels covariance). Optionally add `background_design` (pixels × nuisance parameters, independent columns) and `observation` (a pixel vector to test). Do not combine Gram input with pixel-model fields.

The template bank is capped at 128 hypotheses; the dense pixel path is capped at 2,048 pixels. Larger problems can compute the small Gram matrix externally, using a valid covariance solve/projection. Gram matrices must be symmetric positive semidefinite; pixel covariance must be positive definite. Numerical roundoff in tiny negative Gram eigenvalues is clipped. Background projection removes columns numerically indistinguishable from the nuisance subspace.

`target_power` defaults to 0.9 and must exceed alpha. Calibration and evaluation sample counts default to 4,096 and support 100–1,000,000 draws. `calibration_failure_probability` defaults to 0.05. The simulator uses the Gram matrix directly, so it does not need to generate full pixel arrays on every draw.

```bash
constellation prove_floor input.json -o bounds.json
constellation floor_demo floor-demo-output
```

The report contains the effective Gram matrix, prior weights, nuisance rank, all power bounds, necessary-flux lower bound, conservative detector threshold, calibration confidence/tolerance bound, holdout power/false-alarm estimates and intervals, and an observation decision when pixels were supplied. `necessary_flux_lower_bound: null` with `target_unreachable_at_any_flux: true` means the requested target is incompatible with the information available in this model. It does not mean the source is absent.

## Reproducible experiment and checks

`floor_demo` constructs four motion templates in six seven-by-seven frames, correlated pixel noise, and one unknown constant background per frame. At flux 4 and alpha 0.01, the upper bound is approximately 0.1729; the calibrated detector's simulated power is approximately 0.1326. At flux 14, power is approximately 0.9857. The necessary flux for 90% average power is approximately 10.43; this lower bound does not guarantee that flux 10.43 is sufficient.

Tests check the exact single-template ROC; symmetric/opposite and identical-template divergence identities; the stronger bound for many alternatives; whitening/projection against independent matrix algebra; singular Gram matrices; background-degenerate sources; the likelihood ratio against direct pixel likelihoods; exact one-template power after calibration; and rejection of insufficient calibration samples.

The relevant classical foundation is [Neyman and Pearson (1933)](https://doi.org/10.1098/rsta.1933.0009). The divergence and calibration derivations above state the additional assumptions directly. None of these results validates a real survey's noise or artifact distribution.
