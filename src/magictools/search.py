"""Joint likelihood for a finite, pre-specified bank of trajectories."""

import numpy as np
from scipy.special import log_ndtr


def combine_measurements(measurements, flux_scales=None, trials=1, alpha=0.01):
    """Combine independent measurements assuming f_i = scale_i * reference_flux.

    Positive-source p-values use the signed, unconstrained flux estimate.
    Bonferroni controls the family-wise Gaussian-noise error for the full supplied
    bank, even with correlated trajectories. It does not calibrate artifacts or
    adaptive searches. Negative per-exposure fluxes MUST be retained.
    """
    measurements = list(measurements)
    if not measurements:
        raise ValueError("At least one measurement is required")
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    flux = np.array([m.flux for m in measurements])
    error = np.array([m.flux_error for m in measurements])
    scales = np.ones(len(flux)) if flux_scales is None else np.asarray(flux_scales, dtype=float)
    if scales.shape != flux.shape or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("One finite positive flux scale is required per measurement")
    if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(error)) or np.any(error <= 0):
        raise ValueError("Fluxes must be finite and uncertainties finite and positive")
    information = np.sum((scales / error)**2)
    joint_error = 1 / np.sqrt(information)
    joint_flux = np.sum(scales * flux / error**2) / information
    snr = joint_flux / joint_error
    log_p = float(log_ndtr(-snr))
    log_family_p = min(0.0, log_p + np.log(trials))
    return {
        "flux": float(joint_flux), "flux_error": float(joint_error), "snr": float(snr),
        "positive_source_log_likelihood_ratio": float(0.5 * max(0, snr)**2),
        "log_p_single": log_p, "p_single": float(np.exp(log_p)),
        "log_p_family_upper_bound": float(log_family_p),
        "p_family_upper_bound": float(np.exp(log_family_p)),
        "passes_gaussian_threshold": bool(log_family_p <= np.log(alpha)),
        "alpha": float(alpha), "trials": int(trials), "n_exposures": len(flux),
        "five_sigma_flux_sensitivity": float(5 * joint_error),
        "constant_flux_chi2": float(np.sum(((flux - scales * joint_flux) / error)**2)),
        "constant_flux_dof": len(flux) - 1,
    }
