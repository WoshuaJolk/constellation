# Orbit coverage and artifact-model comparison (v0.2)

The new path is `orbit samples → map_cover → force_track / dig_floor`, with a separate `map_cover → learn_null` observation-planning export. `weigh_stamp` evaluates a specified candidate in a local image stamp. No tool assumes that a supplied exposure contains a non-detection, accepts a discovery automatically, or queries external archives.

## `map_cover`: state vectors to detector pixels

```json
{
  "schema_version": 1,
  "epoch_mjd_tdb": 60000,
  "orbits": [
    {
      "id": "ceres-seed",
      "state_au_au_per_day": [-2.512136901356719, 0.05592387543738495, 0.5361712539349595, -0.001470661278658604, -0.01011189067455166, -0.004469093415014039],
      "prior_weight": 1,
      "reference_flux": 100
    }
  ],
  "exposures": [
    {
      "id": "visit-1",
      "image": "frame.fits",
      "image_hdu": "SCI",
      "variance": "frame.fits",
      "variance_hdu": "VAR",
      "mask": "frame.fits",
      "mask_hdu": "MASK",
      "psf_sigma_pixels": 1.2,
      "flux_scale": 1,
      "mjd_tdb": 60000.25,
      "exposure_seconds": 120
    }
  ],
  "threshold_snr": 5,
  "dynamics": {
    "kernel": "builtin",
    "light_time_margin_days": 10,
    "rtol": 1e-11,
    "atol": 1e-13,
    "ephemeris_step_days": 0.25
  }
}
```

The example state is a real JPL Horizons Ceres state; the flux, exposure, and file names are illustrative. Generate complete runnable inputs with `magictools orbit_demo orbit-demo-output`.

### Coordinates, observers, and epochs

All six-component states have order `[x,y,z,vx,vy,vz]`, solar-system **barycentric** origin, ICRS axes, AU and AU/day. All states share `epoch_mjd_tdb`. Do not paste heliocentric ecliptic elements or UTC MJD into these fields. Bound and hyperbolic Cartesian states follow the same equations. The input family must already represent the desired uncertainty; the code does not perform ranging, fit an arc, or sample a covariance.

Exposure `mjd_tdb` is explicitly the **midpoint** in MJD TDB. It is not read from `MJD-OBS`, which often means UTC start time. `exposure_seconds` defaults to zero (instantaneous), and must be nonnegative. Start/mid/end directions are evaluated with independent light-time iterations. Expected brightness is the supplied `reference_flux` times the exposure's `flux_scale`, as in the existing tools; changing distance, phase, rotational variation, and color are not inferred.

The default observer is Earth's **geocenter**. A supplied exposure observer overrides it:

```json
"observer": {
  "position_au": [0.1, 0.2, 0.3],
  "velocity_au_per_day": [0.001, -0.002, 0.0001]
}
```

These illustrative values must be replaced by the actual observatory/spacecraft barycentric state in ICRS axes at exposure midpoint. Position is extrapolated linearly within that exposure using the supplied velocity. For nearby objects and long exposures, observer acceleration/rotation may require shorter sub-exposures or another observer model. Observatory coordinates are not automatically converted into an observer state. The geocentric default must not be mistaken for a topocentric prediction.

### Force model and numerical limits

Targets are massless particles in the Newtonian field of the Sun, Mercury, Venus, Earth, Moon, Mars, Jupiter, Saturn, Uranus, and Neptune. Planetary trajectories come from ephemerides; this is a restricted/test-particle n-body calculation, not a self-consistent new planetary ephemeris. Giant-planet system masses are point masses. DE440 gravitational parameters are from [JPL's astrodynamic parameters](https://ssd.jpl.nasa.gov/astro_par.html). Pluto, asteroids, relativity, radiation pressure, nongravitational acceleration, collisions, and finite-body forces are omitted. Close encounters and long-baseline extrapolation require independent validation.

The default Astropy `builtin` ephemerides are approximate and fully offline. Planet positions and velocities are sampled and interpolated with cubic Hermite splines, then SciPy DOP853 integrates each target forward/backward from its epoch (maximum integrator step two days). The interval is restricted to 1900–2100; this is a supported interval, not an accuracy guarantee. The ephemeris grid is capped at 100,000 times, with a maximum permitted grid step of one day. Tolerances and interpolation spacing affect numerical convergence but cannot remove ephemeris or force-model errors.

For a local JPL kernel:

```bash
python -m pip install -e '.[ephemeris]'
```

Set `dynamics.kernel` to its local `.bsp` path, relative to the manifest. It must contain all required Sun/planet/Earth/Moon segments for the full integration interval. Kernel files are hashed into provenance; runtime never downloads a kernel. JPL support uses Astropy and `jplephem`. The integration endpoint and the earliest light-time emission must lie inside the supported interval. The default ten-day light-time margin is roughly 1,700 AU; increase it explicitly for more distant targets. Missing coverage causes an error, not extrapolation.

For each reception time, solve `t_emit = t_receive - |r_object(t_emit) - r_observer(t_receive)| / c`. The output is a geometric **astrometric** direction in ICRS axes with light time. Stellar aberration, gravitational light deflection, and atmospheric refraction are omitted. FITS WCS must represent the corresponding astrometric sky. Projection through an incompatible apparent-coordinate WCS is not corrected automatically.

### Coverage, gaps, and sensitivity

Exposures have the same FITS/variance/mask/PSF/scale fields as [the photometry manifest](formats.md), plus midpoint, duration, and optional observer. A selected image HDU represents one chip/footprint. Its celestial WCS must be in that HDU's header. Include separate HDUs as separate entries and encode gaps/bad pixels in masks. Duplicate references to one file/HDU are rejected; duplicated data in other files remain the caller's responsibility.

The reported `on_chip` checks the projected midpoint against the rectangular image bounds. `usable` additionally requires a valid, unmasked centroid pixel and a solvable flux/background fit. This is a **centroid-based selection rule**: an object centered in a gap is excluded even if usable wings remain nearby. A partial PSF near an edge is fitted without renormalization. The full mask/variance plane determines how much information survives. This is not a general model of a survey's detection selection.

The expected flux error is calculated from the PSF/trail template and variance, with background uncertainty included. Measured pixel values are not used to estimate expected S/N; nonfinite science pixels are excluded. The prior-weighted fractions of family members on-chip and usable are separate outputs. They are probabilities under the **supplied discrete family**, not continuous orbit-space completeness guarantees.

The trail is the difference between projected exposure-end and exposure-start coordinates. If the projected midpoint departs from the endpoint midpoint by more than 0.05 pixel, that member/exposure is marked unavailable for the linear-trail model. This three-time check is a limited diagnostic, not a proof of small curvature at all intermediate times. Trails over 64 PSF sigma are also unavailable. No fallback to an incorrect point-source fit is made.

### Outputs and chaining

```bash
magictools map_cover orbits.json -o coverage.json \
  --photometry-output photometry.json --planning-output planning.json
magictools force_track photometry.json -o measured.json
magictools dig_floor photometry.json -o candidates.json
magictools learn_null planning.json -o ranked-exposures.json
```

The main report contains provenance, actual dynamics settings, per-exposure family fractions, and per-member sky/pixel positions, light times, trails, expected S/N, fit uncertainty, and unavailable reasons. All supplied exposures appear, including those covering none of the family. It also embeds the two export manifests.

`photometry_manifest` contains **absolute local file paths** and only usable positions. Entirely uncovered hypotheses retain empty position lists so the search trial count remains intact. `planning_manifest` has the normalized prior weights and Gaussian-model candidate-exposure probabilities. It initially has **no non-detections**. Add an exposure to `non_detections` only after the actual specified detection procedure was run and found no detection; remove it from `candidate_exposures`, and prefer measured injection efficiencies. Do not turn every supplied archival frame into a null by assumption.

## `weigh_stamp`: explicit model evidence

```json
{
  "schema_version": 1,
  "image": "stamp.fits",
  "variance": 4,
  "x": 12.2,
  "y": 11.8,
  "psf_sigma_pixels": 1.1,
  "trail_dx": 8,
  "trail_dy": 2,
  "flux_prior_sigma": 500
}
```

The selected image is already a local stamp (4–16,384 pixels). Coordinates refer to that stamp, zero-based. The tool uses **all** common valid stamp pixels in every model. Variance may be a positive scalar or FITS path; `image_hdu`, `variance_hdu`, `mask`, and `mask_hdu` follow the existing conventions. `trail_dx/dy` are total detector-pixel displacements over the exposure, centered on the specified position. All noise is assumed independent Gaussian with the provided variance.

The model bank is fixed and exposed:

| Model | Templates and internal prior |
| --- | --- |
| `moving_source` | Gaussian PSF integrated along the specified linear trail |
| `stationary_point` | Untrailed PSF at the same specified centroid |
| `extended_source` | Three circular Gaussians with sigma 1.5, 2, or 3 times the PSF; uniform discrete prior |
| `dipole_residual` | Positive PSF minus negative PSF separated by one PSF sigma about the centroid, in eight directed orientations; uniform discrete prior |
| `narrow_streak` | Sigma 0.35 pixel, lengths 3, 6, or 12 pixels, four orientations; uniform discrete prior over 12 templates |
| `cosmic_ray_spike` | A positive impulse at any valid stamp pixel; uniform discrete prior across those pixels |
| `background` | A fitted constant with no source |

This is an interpretable but incomplete bank. A broad-source template is not a full galaxy-blend model; an impulse/streak bank does not capture every cosmic-ray morphology. Chip-edge patterns, arbitrary blends, empirical PSFs, non-Gaussian residuals, and multi-epoch stationary-source tests remain future work. Moving-source geometry and the stamp must be chosen independently of noise, or the selection must be modeled separately. A zero-length trail gives exactly the stationary-point template and is reported as `zero_motion_degeneracy: true`.

### Priors and likelihoods

Every source/artifact amplitude has a **positive half-normal prior** with the required `flux_prior_sigma`. For unit-integral PSFs this is total flux in image units; for the dipole it is the coefficient of the positive-minus-negative PSF pair (each component has unit infinite-plane integral). Templates are not normalized over their cropped/masked pixels. Pick the amplitude prior based on external information and check sensitivity to it; choosing it after looking at the candidate changes the inference.

All models marginalize an identical constant background with the same flat prior, whose arbitrary normalization cancels in ratios. Let `I = sum(w * p_centered^2)`, `B = sum(w * p_centered * d_centered)`, `s = flux_prior_sigma`, and `a = I + 1/s^2`. Then

```
log BF(template vs background)
  = log(2) - 0.5 log(1 + s^2 I) + B^2 / (2 a) + log Phi(B / sqrt(a))
```

The code uses stable log-CDF/log-sum-exp calculations. Within each model class, it averages **evidences**, not maximum fits, over the finite templates. This includes the declared prior mass over cosmic-ray location and other template choices. It is not a correction for choices made outside the model bank.

Model classes have equal prior probability by default. Optional `model_priors` must specify all seven names with positive finite weights; weights are normalized. Unequal priors can change the preferred class. There is no learned training set or hidden classifier.

The output reports each model's marginalized log evidence relative to background, normalized prior and `posterior_within_model_set`, template count, best-evidence template, and two comparisons with the moving model:

- `log_bayes_factor_moving_vs_model`: ratio of prior-marginalized evidences, averaged over each class's finite templates.
- `profile_log_likelihood_ratio_moving_vs_model`: ratio of the maximum nonnegative-amplitude likelihoods, using the best-fit template in each class. This does **not** include the prior-volume penalty and is not a calibrated frequentist test.

Positive ratios favor the moving model. The report includes moving-model residual chi-square so a preferred but inadequate model can be recognized. Posterior weights are conditional on this finite set of models and priors; they are **not** probabilities of astrophysical reality, and missing alternatives can invalidate an interpretation. No detection is automatically accepted or rejected.

## Validation and provenance

Tests compare bound circular motion to analytic positions, conserve energy and reverse time for an unbound trajectory, check light time against exact uniform motion, and check observer parallax. A frozen, independently fetched [JPL Horizons](https://ssd-api.jpl.nasa.gov/doc/horizons.html) fixture contains Ceres geometric barycentric states and light-time-corrected geocentric vectors, together with the exact query URLs and source solution. The tests require no network. The ten-day Ceres check gives about 0.18 km position discrepancy and 0.034–0.038 arcseconds direction discrepancy with the default approximate ephemerides. Halving the interpolation spacing and tightening the integrator also checks numerical convergence. These are limited reference cases, not an instrument-grade certification.

Image tests check trail normalization/recovery, coverage through gaps and off-image regions, sensitivity independence from measured flux, preservation of uncovered trial hypotheses, portable CLI exports, and fully masked exposures. Evidence is checked against independent numerical quadrature over flux, with synthetic controls for every included model class, masking, prior sensitivity, and the zero-motion degeneracy. Model-class controls are implementation tests under their own templates, not real-world artifact discrimination benchmarks.

The optional local-kernel path was also exercised with [JPL DE440s](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp), including a complete `map_cover` run and kernel provenance hashing. In the same short Ceres comparison it reduced the ten-day position discrepancy to about **3.9 metres**. This remains a single reference case with a high-quality initial state, not a promised prediction accuracy. To repeat the optional offline test after obtaining a compatible kernel:

```bash
MAGICTOOLS_JPL_KERNEL=/absolute/path/de440s.bsp python -m pytest tests/test_orbits.py -q
```

Without that variable, the optional kernel test is skipped; all default tests use bundled Astropy ephemerides and the frozen Horizons fixture.

Run `magictools orbit_demo new-directory` for the complete synthetic workflow. The original `magictools demo` remains supported.
