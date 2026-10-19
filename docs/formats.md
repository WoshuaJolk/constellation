# Version 1 input and output formats

All analysis commands accept a local JSON manifest. Unknown fields, duplicate JSON keys, `NaN`/`Infinity`, repeated IDs, and ambiguous coordinate pairs are rejected. Arrays are ordered. FITS images are read one exposure at a time; memory use includes that exposure's science, variance, and mask arrays plus the measurements for all hypotheses. See [orbit and stamp formats](ambitious-tools.md), [detection-bound formats](prove-floor.md), and [stellar-search formats](find-ejector.md) for the newer tools.

## `force_track` and `dig_floor`

```json
{
  "schema_version": 1,
  "exposures": [
    {
      "id": "visit-001",
      "image": "science.fits",
      "image_hdu": "SCI",
      "variance": "science.fits",
      "variance_hdu": "VAR",
      "mask": "science.fits",
      "mask_hdu": "DQ",
      "psf_sigma_pixels": 1.3,
      "flux_scale": 1.0
    },
    {
      "id": "visit-002",
      "image": "other-image.fits",
      "variance": 25.0,
      "psf_sigma_pixels": 1.6,
      "flux_scale": 0.8
    }
  ],
  "trajectories": [
    {
      "id": "orbit-sample-0",
      "positions": [
        {"exposure_id": "visit-001", "ra_deg": 150.1, "dec_deg": 12.3},
        {"exposure_id": "visit-002", "x": 123.4, "y": 56.7}
      ]
    }
  ]
}
```

Required exposure fields: `id`, `image`, `variance`, `psf_sigma_pixels`. Other exposure fields are optional. HDU selectors are integer indices (primary HDU = 0) or extension names; all default to 0. There is no implicit fallback to the first image extension. WCS must be in the selected science extension header. Extended `EXTVER` selectors are not part of version 1.

`variance` is a positive scalar in squared image units or a FITS filename. Nonfinite/zero/negative variance pixels and nonfinite science pixels are excluded. Nonzero mask pixels, including nonfinite mask values, are excluded. No data-quality bit selection is inferred. Science/variance/mask arrays must have matching two-dimensional shapes. PSF sigma is circular Gaussian sigma, not FWHM (`FWHM = 2.35482 sigma`). The stamp extends at least five sigma around the prediction and is clipped at image boundaries.

`flux_scale` defaults to 1. It must be positive and gives image flux for a unit reference flux. All units and filter corrections are supplied by the user. A PSF is integrated over pixels and normalized over the infinite plane; cropped or masked stamps are never renormalized. The centroid must be within the detector's pixel footprint, from -0.5 inclusive to width/height minus 0.5 exclusive. This check describes a rectangular image, not a survey chip model; encode detector gaps in the mask.

Each trajectory has at most one position per exposure. Omitted positions are explicitly unavailable; an empty `positions` list is permitted for a wholly uncovered orbit and still counts toward the trajectory-bank trial count. Pixel coordinates are `(x, y)`, zero-based centers, while arrays are indexed `[y, x]`. Sky coordinates are ICRS degrees (`0 <= ra_deg < 360`, `-90 <= dec_deg <= 90`). Astropy transforms ICRS into the WCS's celestial frame. Generate positions externally or with `map_cover` at each exposure midpoint, using a consistent observer location/time scale and light-time convention.

Each position may additionally include `trail_dx` and `trail_dy`, the signed total displacement in **detector pixels** from exposure start to end (default zero). These apply even when the midpoint is given as RA/Dec. The model integrates a constant-flux Gaussian along the linear segment centered on the midpoint using Gauss–Legendre quadrature. The fitted stamp grows to enclose the trail plus five PSF sigma. Trails longer than 64 PSF sigma are rejected; curved or variable-flux trajectories require another model. Both the measured flux uncertainty and the joint likelihood account for the supplied trail.

### Reports

`trajectories` contains each ID, `status`, `measurements`, and `unavailable`. A measurement reports:

| Field | Meaning |
| --- | --- |
| `flux`, `flux_error`, `snr` | Signed fitted total source flux, standard error, and their ratio |
| `background` | Fitted constant per-pixel background in image units |
| `chi2`, `dof` | Weighted residual sum of squares and valid pixel count minus two |
| `n_pixels` | Number of usable pixels in the fitted stamp |
| `psf_fraction` | Unit-total PSF integral on usable stamp pixels; not a detection efficiency |
| `x`, `y`, `exposure_id`, `flux_scale` | Position and exposure association |

`dig_floor` additionally reports `joint` and sorts trajectories by descending joint S/N. `joint` includes the reference flux/error/S/N; `constant_flux_chi2` and `constant_flux_dof` for across-exposure consistency; `five_sigma_flux_sensitivity`; the positive-source profile log likelihood ratio; single-test and Bonferroni family probability bounds; and the `alpha`, `trials`, `n_exposures`, and `passes_gaussian_threshold` decision. Log p-values remain usable when probability fields underflow to zero. Large S/N with poor PSF chi-square or poor constant-flux consistency deserves inspection; this version does not classify artifacts.

`status: "no_usable_measurements"` and `joint: null` mean no test was possible. Missing input files or malformed global settings cause the command to fail. Fit-level failures, such as an off-image position or fully masked stamp, are recorded with reasons in `unavailable`.

## `learn_null`

```json
{
  "schema_version": 1,
  "hypotheses": [
    {"id": "bright-near", "prior_weight": 1},
    {"id": "faint-far", "prior_weight": 2},
    {"id": "other-location", "prior_weight": 1}
  ],
  "non_detections": [
    {
      "id": "calibrated-past-frame",
      "detection_probability": [0.99, 0.2, 0.0]
    },
    {
      "id": "gaussian-past-frame",
      "expected_snr": [10, 2, 10],
      "threshold_snr": 5,
      "coverage": [1, 1, 0]
    }
  ],
  "candidate_exposures": [
    {"id": "future-frame", "detection_probability": [0.9, 0.9, 0.1]}
  ]
}
```

The hypotheses may be orbit/brightness samples, population parameters, or any discrete alternatives whose likelihoods are supplied. Prior weights need not sum to one, but must be nonnegative with positive total. A zero prior cannot gain mass. All per-hypothesis arrays follow **exactly the `hypotheses` order**.

Each observed or future exposure uses exactly one of:

1. `detection_probability`: a length-N array in [0,1] giving the **unconditional** probability of detection for each hypothesis, including on-chip coverage and losses in the actual selection procedure. Ideally measured by injections through that procedure. Do not supply `coverage` again.
2. `expected_snr`: a length-N array of nonnegative expected signed detection statistics, together with `threshold_snr` (positive scalar or length-N array) and optional `coverage` (scalar or length-N array in [0,1], default 1). This assumes a unit-variance Gaussian detection statistic with that mean. At the threshold, on-chip recovery probability is 50%. This approximation does not infer efficiency from a header limiting magnitude.

`non_detections` may be empty. `candidate_exposures` is optional and may be empty. IDs cannot repeat or appear in both sets. Statistical independence is still the caller's responsibility; different IDs do not make shared pixels independent. If all prior-supported hypotheses make the observed nulls impossible, the tool returns an error instead of inventing a posterior.

The output provides `hypothesis_ids`, normalized `posterior_weights`, `log_evidence` (probability of all nulls under the normalized prior), `effective_sample_size` (1 / sum of squared weights), discrete `entropy_nats`, and `ranked_exposures`. Each ranked entry reports ID, posterior predictive `detection_probability`, and `expected_information_nats`. Ranking optimizes a single binary observation, without scheduling costs, weather, visibility, or joint optimization of a multi-exposure campaign. Entropy is relative to the supplied discretization, not a coordinate-invariant orbit-space volume.
