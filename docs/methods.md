# Methods and validation

## Forced photometry

For usable stamp pixels, model `d = f p + b + noise`, with known diagonal variance `v`. The template `p` is a circular Gaussian integrated analytically across each pixel using normal-CDF differences. Its infinite-plane integral is one. No recentering on the observed image is performed.

With `w = 1/v`, weighted means `p_bar` and `d_bar`, and

```
A = sum(w * (p - p_bar)^2)
B = sum(w * (p - p_bar) * (d - d_bar))
f_hat = B / A
sigma_f = 1 / sqrt(A)
b_hat = d_bar - f_hat * p_bar
```

this is the weighted least-squares/maximum-likelihood solution after profiling out a constant background. It accounts for the covariance between flux and background. Uncertainty is conditional on the supplied variance and model, and is not rescaled by the residual chi-square. A bad model is diagnosed by residuals, not silently made acceptable by inflating errors.

## Joint trajectory likelihood

For independent exposures with per-frame estimates `f_i`, variances `s_i^2`, and known scales `a_i`, the assumed flux is `a_i F`. The exact Gaussian sufficient-statistic combination is:

```
I = sum(a_i^2 / s_i^2)
F_hat = sum(a_i * f_i / s_i^2) / I
sigma_F = 1 / sqrt(I)
Z = F_hat / sigma_F
```

Under these assumptions this is equivalent to fitting every pixel jointly with a separate background per exposure. It combines likelihoods without resampling images or smoothing their PSFs. It evaluates each supplied trajectory separately; it is linear in the number of trajectories times exposures times fitted stamp pixels, and is not the proposed `fold_kepler` transform.

For the positive-flux alternative, the profile log likelihood ratio is `max(0, Z)^2 / 2`. The single-test one-sided probability is `Phi(-Z)`. With M supplied trajectories, `min(1, M * Phi(-Z))` is a Bonferroni upper bound on the chance of any exceedance under the ideal Gaussian null. Correlation *between trajectories* does not invalidate that bound; correlation *between pixels/exposures* invalidates the assumed uncertainty unless modeled. Trial counts must also include searches, grids, cuts, and choices made outside this command. These model probabilities are not artifact probabilities.

## Non-detections

For a detection statistic distributed as `N(mu, 1)` and threshold `t`,

```
P(null | on chip, hypothesis) = Phi(t - mu)
P(null | hypothesis) = (1 - c) + c * Phi(t - mu)
```

where `c` is usable-coverage probability. For injection-calibrated total detection efficiency `e`, simply use `P(null) = 1 - e`. With independent observations, multiply the likelihoods into the prior; calculations use log-CDF, log-add-exp, and log-sum-exp to avoid underflow. Zero likelihood for an individual hypothesis is allowed; zero likelihood for every supported hypothesis is a model inconsistency.

For future detection probabilities `q_h`, posterior weights `w_h`, and binary entropy `H_b`, expected information gain is

```
H_b(sum(w_h * q_h)) - sum(w_h * H_b(q_h))
```

This is mutual information between the hypothesis and the binary detection outcome. A frame that detects every surviving hypothesis with the same probability has zero discriminatory value. Use a sufficiently sampled joint orbit/brightness/selection model; likelihoods averaged over uncertain shared nuisance parameters generally do not factor independently across frames.

## Validation included

- Gaussian pixel integrals checked against numerical quadrature and total-flux normalization.
- Noiseless subpixel flux/background recovery, including masks, nonfinite pixels, and clipped detector edges.
- Independent weighted matrix solution for heteroscedastic pixel noise and flux/background covariance.
- Independent multi-exposure pixel design matrix compared with the sufficient-statistic joint fit.
- 1,200 seeded noise-only fits checking the signed statistic's mean and variance.
- Signed negative-flux behavior, scale conversions, trial-count correction, and invalid parameter rejection.
- Hand-computed Bayes updates, off-chip and threshold cases, extreme tail stability, impossible evidence, and analytically known information gains.
- FITS named/numeric extensions, WCS/pixel agreement across RA wrap, masks/variance planes, provenance hashes, duplicate image rejection, CLI JSON, and overwrite protection.
- End-to-end seeded injection with changing PSF/noise/scale/WCS, a correct track and two control tracks, and all three tool reports.

These validate the implementation under its stated model. They do not validate any telescope's noise distribution, PSF, detection efficiency, or astrometric solution. Next validation should use injections into real calibrated survey images, with null/reversed-rate controls and independently chosen thresholds.

## References and existing foundations

- [Astropy's shared WCS interface](https://docs.astropy.org/en/stable/wcs/wcsapi.html): coordinate conventions and celestial-to-pixel transformations used here.
- [SciPy `log_ndtr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html): numerically stable Gaussian log probabilities.
- [KBMOD](https://github.com/dirac-institute/kbmod): established moving-object search software; this package does not claim to originate likelihood stacking.
- [Original catalogue](https://magictools-theta.vercel.app/): source of the tool names and broader goals. Claims there about what has never been done are aspirations to investigate, not novelty claims established by this implementation.
