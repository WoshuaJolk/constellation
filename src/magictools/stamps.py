"""Bayesian finite-model comparison on one common set of stamp pixels."""

import numpy as np
from scipy.special import log_ndtr, logsumexp

from .photometry import gaussian_psf, trailed_psf


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


def weigh_stamp(image, variance, x, y, psf_sigma, trail_dx, trail_dy, flux_prior_sigma, mask=None, model_priors=None):
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
    add("moving_source", [trailed_psf(xx, yy, x, y, psf_sigma, trail_dx, trail_dy)], ["specified linear trail"])
    add("stationary_point", [gaussian_psf(xx, yy, x, y, psf_sigma)], ["untrailed PSF at specified centroid"])
    add("extended_source", [gaussian_psf(xx, yy, x, y, psf_sigma * factor) for factor in [1.5, 2, 3]],
        [f"sigma multiplier {factor}" for factor in [1.5, 2, 3]])
    templates, labels = [], []
    for angle in np.arange(8) * np.pi / 4:
        dx, dy = psf_sigma * np.cos(angle), psf_sigma * np.sin(angle)
        templates.append(gaussian_psf(xx, yy, x - dx / 2, y - dy / 2, psf_sigma) -
                         gaussian_psf(xx, yy, x + dx / 2, y + dy / 2, psf_sigma))
        labels.append(f"positive-minus-negative PSF; angle {np.degrees(angle):g} deg")
    add("dipole_residual", templates, labels)
    templates, labels = [], []
    for length in [3, 6, 12]:
        for angle in np.arange(4) * np.pi / 4:
            templates.append(trailed_psf(xx, yy, x, y, 0.35, length * np.cos(angle), length * np.sin(angle)))
            labels.append(f"sigma 0.35; length {length}; angle {np.degrees(angle):g} deg")
    add("narrow_streak", templates, labels)
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
            ["image_hdu", "variance_hdu", "mask", "mask_hdu", "model_priors"])
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
    result = weigh_stamp(image, variance, manifest["x"], manifest["y"], manifest["psf_sigma_pixels"],
                         manifest["trail_dx"], manifest["trail_dy"], manifest["flux_prior_sigma"], mask, manifest.get("model_priors"))
    return {"schema_version": 1, "tool": "weigh_stamp", "provenance": _provenance(files), **result}
