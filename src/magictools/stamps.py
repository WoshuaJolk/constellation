"""Bayesian finite-model comparison on one common set of stamp pixels."""

import numpy as np
from scipy.special import log_ndtr, logsumexp

from .photometry import gaussian_psf, trailed_psf, sampled_psf, validate_psf_kernel


def template_evidence(data, variance, templates, flux_prior_sigma):
    """Evidence ratios to constant background, with a positive half-normal flux prior.

    All models marginalize the SAME constant background with the same flat prior;
    its arbitrary normalization cancels. Templates may have negative lobes. Their
    physical amplitude normalization is part of the prior, not refit automatically.
    """
    data, variance, templates = map(lambda x: np.asarray(x, dtype=float), (data, variance, templates))
    if data.ndim != 1 or data.size < 3 or variance.shape != data.shape or templates.ndim != 2 or templates.shape[1] != data.size:
        raise ValueError("Expected data/variance vectors and templates of shape (models, pixels)")
    if not all(np.all(np.isfinite(x)) for x in (data, variance, templates)) or np.any(variance <= 0):
        raise ValueError("Evidence requires finite data/templates and positive finite variances")
    if not np.isfinite(flux_prior_sigma) or flux_prior_sigma <= 0:
        raise ValueError("flux_prior_sigma must be finite and positive")
    w = 1 / variance
    d = data - np.average(data, weights=w)
    p = templates - np.average(templates, weights=w, axis=1)[:, None]
    information = np.sum(w * p**2, axis=1)
    score = np.sum(w * p * d, axis=1)
    return _evidence_statistics(information, score, flux_prior_sigma)


def _evidence_statistics(information, score, prior_sigma):
    precision = information + 1 / prior_sigma**2
    log_bf = np.log(2) - 0.5 * np.log1p(prior_sigma**2 * information) + 0.5 * score**2 / precision + log_ndtr(score / np.sqrt(precision))
    improvement = np.divide(np.maximum(0, score)**2, information, out=np.zeros_like(score), where=information > 0)
    return log_bf, improvement


def weigh_stamp(image, variance, x, y, psf_sigma, trail_dx, trail_dy, flux_prior_sigma, mask=None, model_priors=None, psf_kernel=None, empirical_artifacts=None):
    """Compare a moving Gaussian PSF against a finite, explicitly named model bank.

    Reports both marginal evidence ratios and profile likelihood ratios. Neither
    is a calibrated false-positive rate or a probability of astronomical reality.
    The image is the stamp; x/y are local zero-based coordinates within it.
    """
    image = np.asarray(image, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if image.ndim != 2 or min(image.shape) < 2 or image.size > 16384:
        raise ValueError("Supply a 2D stamp of 4--16384 pixels")
    if variance.ndim == 0:
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Scalar variance must be finite and positive")
        variance = np.broadcast_to(variance, image.shape)
    if variance.shape != image.shape:
        raise ValueError("Variance shape must match stamp")
    if not np.all(np.isfinite([x, y, psf_sigma, trail_dx, trail_dy, flux_prior_sigma])) or psf_sigma <= 0 or flux_prior_sigma <= 0:
        raise ValueError("Coordinates/trail must be finite; PSF sigma and flux prior sigma positive")
    if not (-0.5 <= x < image.shape[1] - 0.5 and -0.5 <= y < image.shape[0] - 0.5):
        raise ValueError("Candidate centroid must lie within the stamp")
    valid = np.isfinite(image) & np.isfinite(variance) & (variance > 0)
    if mask is not None:
        if np.shape(mask) != image.shape:
            raise ValueError("Mask shape must match stamp")
        valid &= np.asarray(mask) == 0
    if valid.sum() < 3:
        raise ValueError("At least three valid pixels are required")
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    d, v = image[valid], variance[valid]
    models = {}
    def add(name, templates, labels):
        bank = np.array([template[valid] for template in templates])
        log_bf, improvement = template_evidence(d, v, bank, flux_prior_sigma)
        best = int(np.argmax(log_bf))
        models[name] = {"log_evidence_vs_background": float(logsumexp(log_bf) - np.log(len(bank))),
                        "profile_chi2_improvement": float(np.max(improvement)),
                        "template_count": len(bank), "best_template": labels[best]}
    if psf_kernel is not None:
        psf_kernel = validate_psf_kernel(psf_kernel)
    def point(px, py, dx=0, dy=0):
        return (trailed_psf(xx, yy, px, py, psf_sigma, dx, dy) if psf_kernel is None
                else sampled_psf(xx, yy, px, py, psf_kernel, dx, dy))
    add("moving_source", [point(x, y, trail_dx, trail_dy)], ["specified linear trail"])
    add("stationary_point", [point(x, y)], ["untrailed PSF at specified centroid"])
    add("extended_source", [gaussian_psf(xx, yy, x, y, psf_sigma * factor) for factor in [1.5, 2, 3]],
        [f"sigma multiplier {factor}" for factor in [1.5, 2, 3]])
    templates, labels = [], []
    for angle in np.arange(8) * np.pi / 4:
        dx, dy = psf_sigma * np.cos(angle), psf_sigma * np.sin(angle)
        templates.append(point(x - dx / 2, y - dy / 2) - point(x + dx / 2, y + dy / 2))
        labels.append(f"positive-minus-negative PSF; angle {np.degrees(angle):g} deg")
    add("dipole_residual", templates, labels)
    templates, labels = [], []
    for length in [3, 6, 12]:
        for angle in np.arange(4) * np.pi / 4:
            templates.append(trailed_psf(xx, yy, x, y, 0.35, length * np.cos(angle), length * np.sin(angle)))
            labels.append(f"sigma 0.35; length {length}; angle {np.degrees(angle):g} deg")
    add("narrow_streak", templates, labels)
    if empirical_artifacts is not None:
        if not isinstance(empirical_artifacts, dict) or not 1 <= len(empirical_artifacts) <= 128:
            raise ValueError("Supply 1..128 named empirical artifact templates")
        templates, labels = [], []
        for label, template in empirical_artifacts.items():
            template = np.asarray(template, dtype=float)
            if not isinstance(label, str) or not label or template.shape != image.shape or not np.all(np.isfinite(template)):
                raise ValueError("Artifact templates need names and finite arrays matching the stamp")
            positive_mass = np.maximum(template, 0).sum()
            if not np.isfinite(positive_mass) or positive_mass <= 0:
                raise ValueError("Artifact templates must have positive template mass")
            # One amplitude unit is one unit of positive-lobe flux; negative
            # lobes remain at their supplied ratio. Normalize before masking.
            templates.append(template / positive_mass)
            labels.append(label)
        add("empirical_artifact", templates, labels)
    # Exact impulse-at-each-valid-pixel statistics, avoiding an N by N array.
    w = 1 / v
    information = w - w**2 / w.sum()
    score = w * (d - np.average(d, weights=w))
    log_bf, improvement = _evidence_statistics(information, score, flux_prior_sigma)
    best_y, best_x = np.argwhere(valid)[int(np.argmax(log_bf))]
    models["cosmic_ray_spike"] = {"log_evidence_vs_background": float(logsumexp(log_bf) - np.log(len(d))),
        "profile_chi2_improvement": float(np.max(improvement)), "template_count": len(d),
        "best_template": f"impulse at x={best_x}, y={best_y}"}
    models["background"] = {"log_evidence_vs_background": 0.0, "profile_chi2_improvement": 0.0,
                             "template_count": 1, "best_template": "constant background only"}
    priors = {name: 1.0 for name in models} if model_priors is None else model_priors
    if set(priors) != set(models):
        raise ValueError("model_priors must specify every model name exactly once")
    values = np.array([priors[name] for name in models], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All model prior weights must be positive and finite")
    values /= values.max()
    values /= values.sum()
    log_weights = np.log(values) + [row["log_evidence_vs_background"] for row in models.values()]
    posterior = np.exp(log_weights - logsumexp(log_weights))
    source = models["moving_source"]
    for (name, row), prior, probability in zip(models.items(), values, posterior):
        row.update(prior_probability=float(prior), posterior_within_model_set=float(probability),
                   log_bayes_factor_moving_vs_model=source["log_evidence_vs_background"] - row["log_evidence_vs_background"],
                   profile_log_likelihood_ratio_moving_vs_model=0.5 * (source["profile_chi2_improvement"] - row["profile_chi2_improvement"]))
    background_chi2 = float(np.sum(w * (d - np.average(d, weights=w))**2))
    return {"models": models, "preferred_model": max(models, key=lambda name: models[name]["posterior_within_model_set"]),
            "zero_motion_degeneracy": bool(trail_dx == 0 and trail_dy == 0),
            "point_source_psf": "native_pixel_empirical" if psf_kernel is not None else "gaussian",
            "n_valid_pixels": int(valid.sum()), "background_only_chi2": background_chi2,
            "moving_source_profile_chi2": max(0.0, background_chi2 - source["profile_chi2_improvement"]),
            "flux_prior": {"distribution": "positive half-normal", "sigma": flux_prior_sigma},
            "interpretation": "Conditional comparison of specified templates/priors, not a calibrated real/bogus probability"}


def run_stamp(manifest_path):
    from pathlib import Path
    from .io import _fields, _fits_plane, _provenance, read_json
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ["schema_version", "image", "variance", "x", "y", "psf_sigma_pixels", "trail_dx", "trail_dy", "flux_prior_sigma"],
            ["image_hdu", "variance_hdu", "mask", "mask_hdu", "model_priors", "psf_image", "psf_hdu", "stamp_radius_pixels", "empirical_artifacts"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    files = [path]
    def plane(key):
        filename = (path.parent / manifest[key]).resolve()
        files.append(filename)
        return _fits_plane(filename, manifest.get(f"{key}_hdu", 0))
    image = plane("image")
    variance = plane("variance") if isinstance(manifest["variance"], str) else manifest["variance"]
    mask = plane("mask") if "mask" in manifest else None
    kernel = None
    if "psf_image" in manifest:
        psf_path = (path.parent / manifest["psf_image"]).resolve()
        kernel = _fits_plane(psf_path, manifest.get("psf_hdu", 0))
        files.append(psf_path)
    x, y = manifest["x"], manifest["y"]
    if image.ndim != 2 or not np.all(np.isfinite([x, y])) or not (-.5 <= x < image.shape[1]-.5 and -.5 <= y < image.shape[0]-.5):
        raise ValueError("Candidate must be inside the 2D image")
    for value in [variance, mask]:
        if value is not None and np.ndim(value) != 0 and np.shape(value) != image.shape:
            raise ValueError("Variance and mask shapes must match image before cutting stamp")
    bounds = None
    if "stamp_radius_pixels" in manifest:
        radius = manifest["stamp_radius_pixels"]
        if isinstance(radius, bool) or not isinstance(radius, int) or not 1 <= radius <= 63:
            raise ValueError("stamp_radius_pixels must be an integer in [1,63]")
        x0, x1 = max(0, int(np.floor(x))-radius), min(image.shape[1], int(np.floor(x))+radius+2)
        y0, y1 = max(0, int(np.floor(y))-radius), min(image.shape[0], int(np.floor(y))+radius+2)
        image = image[y0:y1, x0:x1]
        if np.ndim(variance) == 2:
            variance = variance[y0:y1, x0:x1]
        if mask is not None:
            mask = mask[y0:y1, x0:x1]
        x, y = x-x0, y-y0
        bounds = {"x_start": x0, "x_stop": x1, "y_start": y0, "y_stop": y1}
    artifacts = None
    if "empirical_artifacts" in manifest:
        from .io import _named_rows
        _named_rows(manifest["empirical_artifacts"], "empirical_artifacts")
        artifacts = {}
        for row in manifest["empirical_artifacts"]:
            _fields(row, ["id", "image"], ["image_hdu"])
            filename = (path.parent / row["image"]).resolve()
            files.append(filename)
            artifacts[row["id"]] = _fits_plane(filename, row.get("image_hdu", 0))
    result = weigh_stamp(image, variance, x, y, manifest["psf_sigma_pixels"],
                         manifest["trail_dx"], manifest["trail_dy"], manifest["flux_prior_sigma"], mask, manifest.get("model_priors"), kernel, artifacts)
    return {"schema_version": 1, "tool": "weigh_stamp", "provenance": _provenance(files), "stamp_bounds": bounds, **result}
