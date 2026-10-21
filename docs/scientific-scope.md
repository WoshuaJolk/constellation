# Scientific inputs, interpretation and Python API

## Using your own data

See [input and output reference](formats.md) for complete manifests. Supply calibrated images, **variance** (not standard deviation or inverse variance), the Gaussian PSF sigma in pixels, and positions at the actual exposure times. Positions may be zero-based pixel centers or ICRS RA/Dec in degrees. Paths are relative to the manifest, so commands work from any directory.

`prove_floor` instead accepts pixel templates and a known covariance, or their noise-weighted Gram matrix. `find_ejector` accepts present-day Galactocentric 6D states/covariances or complete posterior samples, including an optional ICRS conversion. Missing radial velocities and uncertain distances require supplied posteriors; neither is silently invented.

Alternatively, provide barycentric Cartesian orbit samples in **ICRS axes, AU and AU/day, at an explicit MJD TDB epoch**, and let `map_cover` generate the positions. The [orbit coverage and stamp comparison guide](ambitious-tools.md) specifies the new formats, observer conventions, model priors, and limitations. The dynamics default uses Astropy's offline approximate ephemerides; optionally supply a local JPL SPK kernel after installing `.[ephemeris]`.

`dig_floor` assumes a single reference flux with known positive per-exposure `flux_scale`: predicted image flux = reference flux × scale. Match filters or supply a defensible color model. Exposure time, throughput, magnitude zero points, and unit conversion are **not** inferred from headers. A variable source needs a different model.

Use injection-measured `detection_probability` in `learn_null` when available. That probability must include usable coverage and the actual detection procedure. For the Gaussian option, `coverage` means the probability the hypothesis is on usable detector area; it is **not** `force_track`'s `psf_fraction`. A non-detection is an observed failure to pass a specified detection test. A missing image, invalid fit, or off-chip prediction is not itself a measured upper limit.

## Scientific scope

- The likelihood assumes a correct Gaussian or supplied empirical PSF, optionally integrated over a constant-rate linear trail; a constant background per stamp; known, independent Gaussian pixel noise; and independent exposures. ZTF preparation supplies an estimated noise model, so its Gaussian p-values need empirical calibration. Shared difference-image templates and resampling can violate independence. Correlated noise, blends, curved trails, PSF mismatch, and WCS errors need additional modeling.
- Forced positions and the orbit family must be specified independently of the fitted pixel noise. `map_cover` propagates supplied states; it does not fit them to detections, search every possible orbit, or account for adaptive searches performed outside this run. It defaults to a geocentric observer; supply an observatory/spacecraft barycentric state when parallax matters.
- `weigh_stamp` reports model-conditional posterior weights, Bayes factors, and profile likelihood ratios. These depend on the stated template bank and amplitude/model priors; they are not calibrated probabilities that an object is real. A zero-length trail is explicitly indistinguishable from the stationary-point template in a single stamp.
- Each per-frame flux remains signed. The joint fit includes negative values and reports a signed S/N. The one-sided Gaussian p-value tests a positive source. Bonferroni uses **every supplied trajectory**, including candidates with no usable data; it controls only the stated Gaussian null model and does not establish an astrophysical discovery.
- The reported `five_sigma_flux_sensitivity` is five times the fitted flux uncertainty for a fixed trajectory. It is neither an injection-derived survey limit nor a posterior upper bound, and it does not fulfill the catalogue's proposed `prove_floor` theorem.
- Non-detections must be conditionally independent given each hypothesis. Do not reuse the same pixels as both a flux likelihood in the prior and a non-detection likelihood. Discrete weights cannot recover missing hypotheses; a concentrated posterior or low effective sample size calls for a better sampled family.
- Invalid/missing predictions are listed in `unavailable`; no fabricated zero-flux measurements are inserted. A trajectory with no usable fits has `joint: null`. Check these fields before interpreting the ranking. Duplicate image-HDU references are rejected; duplicated data stored in different files or HDUs remain the caller's responsibility.

The [methods](methods.md) explain the equations and validation. The [roadmap](roadmap.md) assesses the remaining catalogue ideas and the practical next steps.

## Python API

```python
from magictools import forced_photometry, combine_measurements, update_nulls, rank_exposures
from magictools.nulls import gaussian_log_null

# image is a 2D NumPy array; variance has image-units squared.
measurement = forced_photometry(image, variance, x=42.3, y=17.8, psf_sigma=1.4)
joint = combine_measurements([measurement])

log_null = gaussian_log_null([[10, 2, 10]], threshold_snr=5, coverage=[[1, 1, 0]])
posterior = update_nulls([1, 1, 1], log_null)
ranked = rank_exposures(posterior['posterior_weights'], [[0.5, 0.95, 0.05]])
```

Runtime dependencies: NumPy, SciPy, Astropy. Tests use pytest. The repository includes GitHub Actions research checks on Python 3.11 and 3.12; archive checks also cover Python 3.10.

Independent validation includes analytic bound/unbound two-body trajectories, light-time/parallax cases, and frozen JPL Horizons Ceres vectors. In the ten-day Ceres check, the propagated position differs by about **0.18 km** and the geocentric direction by **0.034–0.038 arcseconds**. These results validate that particular case, not arbitrary close encounters, long arcs, or astrometric precision across the sky. All tests run offline.

