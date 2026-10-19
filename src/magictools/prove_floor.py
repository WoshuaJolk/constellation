"""Finite-family Gaussian detection bounds and a prior-optimal LR statistic.

This implements established Neyman-Pearson and chi-square divergence results,
not a universal moving-source limit or a newly discovered theorem.
"""

from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import beta, binom, norm

from .io import _fields, _provenance, read_json
from .nulls import _weights


def projected_templates(templates, covariance, background_design=None, observation=None):
    """Whiten known correlated Gaussian noise and remove a common nuisance subspace.

    Rows of templates are pixel signals per unit flux. Background columns must be
    fixed in advance; optimality is in this projected-data experiment.
    """
    templates, covariance = np.asarray(templates, float), np.asarray(covariance, float)
    if templates.ndim != 2 or min(templates.shape) < 1 or not np.all(np.isfinite(templates)):
        raise ValueError("Templates must be a finite hypotheses-by-pixels matrix")
    h, p = templates.shape
    if p > 2048 or h > 128:
        raise ValueError("Dense model supports at most 128 templates and 2048 pixels")
    if covariance.shape != (p, p) or not np.all(np.isfinite(covariance)) or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Covariance must be a finite symmetric pixels-by-pixels matrix")
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("Noise covariance must be positive definite") from error
    white = solve_triangular(factor, templates.T, lower=True)
    y = None
    if observation is not None:
        observation = np.asarray(observation, float)
        if observation.shape != (p,) or not np.all(np.isfinite(observation)):
            raise ValueError("Observation must be a finite pixel vector")
        y = solve_triangular(factor, observation, lower=True)
    nuisance_rank = 0
    if background_design is not None:
        design = np.asarray(background_design, float)
        if design.ndim != 2 or design.shape[0] != p or not 0 < design.shape[1] < p or not np.all(np.isfinite(design)):
            raise ValueError("Background design must have shape (pixels, nuisance parameters), with rank below pixels")
        design = solve_triangular(factor, design, lower=True)
        q, singular, _ = np.linalg.svd(design, full_matrices=False)
        if singular[-1] <= singular[0] * 1e-12:
            raise ValueError("Background design columns must be numerically independent")
        nuisance_rank = len(singular)
        original_norms = np.linalg.norm(white, axis=0)
        white -= q @ (q.T @ white)
        white[:, np.linalg.norm(white, axis=0) <= 1e-12 * original_norms] = 0
        if y is not None:
            y -= q @ (q.T @ y)
    return white.T @ white, None if y is None else white.T @ y, nuisance_rank


def validate_gram(gram):
    gram = np.asarray(gram, float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not 1 <= len(gram) <= 128 or not np.all(np.isfinite(gram)):
        raise ValueError("Gram matrix must be a finite square matrix of size 1--128")
    scale = max(float(np.max(np.abs(gram))), np.finfo(float).tiny)
    if not np.allclose(gram, gram.T, atol=scale * 1e-12, rtol=1e-12):
        raise ValueError("Gram matrix must be symmetric")
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
    if eigenvalues.min() < -scale * 1e-10:
        raise ValueError("Gram matrix must be positive semidefinite")
    factor = vectors * np.sqrt(np.maximum(0, eigenvalues))
    return factor @ factor.T, factor


def power_bounds(gram, weights, flux, alpha):
    """Analytic upper bounds on prior-averaged power of ANY level-alpha test.

    chi^2(P_mixture || P_null) = sum_ij w_i w_j exp(flux^2 K_ij) - 1.
    The oracle knows the correct trajectory; it is an upper bound, not available
    to a real unknown-trajectory search. Floating-point evaluation is not a formal
    interval-arithmetic proof certificate.
    """
    if not np.isfinite(flux) or flux < 0 or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Flux must be nonnegative and alpha in (0,1)")
    weights = _weights(weights)
    gram, _ = validate_gram(gram)
    if len(weights) != len(gram):
        raise ValueError("One prior weight is required per template")
    if flux == 0:
        log_second = 0.0
    else:
        with np.errstate(divide="ignore"):
            log_weights = np.log(weights)
        log_second = max(0.0, float(logsumexp(log_weights[:, None] + log_weights[None, :] + flux**2 * gram)))
    oracle = float(weights @ norm.cdf(flux * np.sqrt(np.maximum(0, np.diag(gram))) - norm.isf(alpha)))
    # Saturate before exponentiation: either divergence bound is trivial once
    # chi-square is sufficiently large. Retain log second moment for inspection.
    if log_second > 50:
        tv_bound = centered_bound = 1.0
    else:
        chi_square = np.expm1(log_second)
        tv_bound = min(1.0, alpha + 0.5 * np.sqrt(chi_square))
        centered_bound = min(1.0, alpha + np.sqrt(alpha * (1 - alpha) * chi_square))
    return {"flux": float(flux), "alpha": float(alpha), "oracle_power_upper_bound": oracle,
            "total_variation_power_upper_bound": float(tv_bound),
            "centered_cauchy_schwarz_power_upper_bound": float(centered_bound),
            "best_power_upper_bound": float(min(oracle, tv_bound, centered_bound)),
            "log_null_second_moment_of_likelihood_ratio": log_second}


def necessary_flux(gram, weights, alpha, target_power):
    """Necessary flux from the upper bound; not a sufficient detection threshold."""
    if not np.isfinite(target_power) or not alpha < target_power < 1:
        raise ValueError("target_power must be between alpha and one")
    gram, _ = validate_gram(gram)
    weights = _weights(weights)
    diagonal = np.maximum(0, np.diag(gram))
    if len(weights) != len(diagonal):
        raise ValueError("One prior weight is required per template")
    maximum_power = float(weights @ np.where(diagonal > 0, 1.0, alpha))
    if maximum_power <= target_power:
        return None
    def residual(flux):
        return power_bounds(gram, weights, flux, alpha)["best_power_upper_bound"] - target_power
    high = 1 / np.sqrt(diagonal.max())
    for _ in range(100):
        if residual(high) >= 0:
            return float(brentq(residual, 0, high, xtol=high * 1e-10))
        high *= 2
    raise ValueError("Could not bracket the necessary-flux bound")


def mixture_log_lr(scores, gram, weights, flux):
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    return logsumexp(log_weights + flux * scores - 0.5 * flux**2 * np.diag(gram), axis=-1)


def binomial_interval(successes, trials, confidence=0.95):
    tail = (1 - confidence) / 2
    return [0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1)),
            1.0 if successes == trials else float(beta.ppf(1 - tail, successes + 1, trials - successes))]


def calibrate_detector(gram, weights, flux, alpha=0.01, calibration_samples=4096,
                       evaluation_samples=4096, calibration_failure_probability=0.05, seed=2026,
                       observed_scores=None):
    """Independent order-statistic calibration and holdout power evaluation.

    A one-sided tolerance bound selects a conservative null quantile. With
    probability >= 1-delta over calibration draws, the test's null rejection
    probability is <= alpha under this exact Gaussian model. This is distinct
    from the analytic impossibility bound and from an instrument calibration.
    """
    gram, factor = validate_gram(gram)
    weights = _weights(weights)
    if len(weights) != len(gram) or not np.isfinite(flux) or flux < 0:
        raise ValueError("Weights must match templates and flux must be finite/nonnegative")
    if not 0 < alpha < 1 or not 0 < calibration_failure_probability < 1:
        raise ValueError("alpha and calibration_failure_probability must be in (0,1)")
    for count in [calibration_samples, evaluation_samples]:
        if isinstance(count, bool) or not isinstance(count, int) or not 100 <= count <= 1000000:
            raise ValueError("Simulation sample counts must be integers in [100,1000000]")
    order = int(binom.ppf(1 - calibration_failure_probability, calibration_samples, 1 - alpha)) + 1
    if order > calibration_samples:
        raise ValueError("Too few calibration samples for the requested alpha/confidence")
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    def simulate(count, rng, alternative=False):
        values = []
        for start in range(0, count, 2048):
            n = min(2048, count - start)
            scores = rng.normal(size=(n, len(weights))) @ factor.T
            if alternative:
                labels = rng.choice(len(weights), size=n, p=weights)
                scores += flux * gram[labels]
            values.append(mixture_log_lr(scores, gram, weights, flux))
        return np.concatenate(values)
    calibration = simulate(calibration_samples, generators[0])
    threshold = float(np.partition(calibration, order - 1)[order - 1])
    null = simulate(evaluation_samples, generators[1])
    alternate = simulate(evaluation_samples, generators[2], alternative=True)
    false = int(np.count_nonzero(null > threshold))
    recovered = int(np.count_nonzero(alternate > threshold))
    result = {"statistic": "prior-mixture log likelihood ratio", "threshold_log_lr": threshold,
              "reject_when": "log_lr > threshold_log_lr", "calibration_samples": calibration_samples,
              "calibration_order_statistic": order, "calibration_confidence": 1 - calibration_failure_probability,
              "null_rejection_probability_upper_bound": float(beta.ppf(1 - calibration_failure_probability, calibration_samples + 1 - order, order)),
              "evaluation_samples": evaluation_samples, "null_holdout_rejection_fraction": false / evaluation_samples,
              "null_holdout_interval_95": binomial_interval(false, evaluation_samples),
              "mixture_power_estimate": recovered / evaluation_samples,
              "mixture_power_interval_95": binomial_interval(recovered, evaluation_samples)}
    if observed_scores is not None:
        statistic = float(mixture_log_lr(np.asarray(observed_scores), gram, weights, flux))
        result["observation"] = {"log_lr": statistic, "rejects_gaussian_null": statistic > threshold}
    return result


def run_floor(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ["schema_version", "prior_weights", "flux", "alpha"],
            ["gram_matrix", "templates", "variance", "covariance", "background_design", "observation",
             "target_power", "calibration_samples", "evaluation_samples", "calibration_failure_probability", "seed"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    scores, nuisance_rank = None, 0
    if "gram_matrix" in manifest:
        if set(manifest) & {"templates", "variance", "covariance", "background_design", "observation"}:
            raise ValueError("Gram input cannot be combined with pixel-model fields")
        gram, _ = validate_gram(manifest["gram_matrix"])
    else:
        if "templates" not in manifest or len(set(manifest) & {"variance", "covariance"}) != 1:
            raise ValueError("Supply gram_matrix or templates with exactly one noise model")
        covariance = manifest.get("covariance")
        if covariance is None:
            variance = np.asarray(manifest["variance"], float)
            if variance.ndim != 1 or not np.all(np.isfinite(variance)) or np.any(variance <= 0):
                raise ValueError("variance must be a positive finite pixel vector")
            covariance = np.diag(variance)
        gram, scores, nuisance_rank = projected_templates(manifest["templates"], covariance,
                                                         manifest.get("background_design"), manifest.get("observation"))
    weights = _weights(manifest["prior_weights"])
    bounds = power_bounds(gram, weights, manifest["flux"], manifest["alpha"])
    target = manifest.get("target_power", 0.9)
    floor = necessary_flux(gram, weights, manifest["alpha"], target)
    detector = calibrate_detector(gram, weights, manifest["flux"], manifest["alpha"],
        **{key: manifest[key] for key in ["calibration_samples", "evaluation_samples", "calibration_failure_probability", "seed"] if key in manifest},
        observed_scores=scores)
    return {"schema_version": 1, "tool": "prove_floor", "provenance": _provenance([path]),
            "model": "finite fixed-flux Gaussian template mixture, after any specified nuisance projection",
            "prior_weights": weights.tolist(), "gram_matrix": gram.tolist(), "nuisance_rank": nuisance_rank,
            "bounds": bounds, "target_power": target, "necessary_flux_lower_bound": floor,
            "target_unreachable_at_any_flux": floor is None, "detector": detector,
            "scope": "Prior-average power under stated noise/templates; not a worst-case orbit guarantee or a survey limit"}
