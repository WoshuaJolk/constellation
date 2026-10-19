"""Discrete-hypothesis inference from calibrated binary detection efficiencies."""

import numpy as np
from scipy.special import log_ndtr, logsumexp, xlogy


def _weights(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0 or not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Weights must be a nonempty finite nonnegative vector with positive total")
    weights = weights / weights.max()
    return weights / weights.sum()


def gaussian_log_null(expected_snr, threshold_snr=5, coverage=1):
    """log P(no detection | hypothesis) for a specified Gaussian threshold test.

    Coverage is P(on usable detector), not the fraction of PSF flux in a stamp.
    On-chip: P(null) = Phi(threshold - expected SNR); off-chip: P(null) = 1.
    Prefer injection-calibrated efficiencies for real pipeline non-detections.
    """
    snr, threshold, coverage = np.broadcast_arrays(
        np.asarray(expected_snr, dtype=float), np.asarray(threshold_snr, dtype=float),
        np.asarray(coverage, dtype=float))
    if not np.all(np.isfinite(snr)) or np.any(snr < 0) or not np.all(np.isfinite(threshold)) or np.any(threshold <= 0):
        raise ValueError("Expected SNR must be finite/nonnegative and threshold finite/positive")
    if not np.all(np.isfinite(coverage)) or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("Coverage must lie in [0, 1]")
    with np.errstate(divide="ignore"):
        return np.logaddexp(np.log1p(-coverage), np.log(coverage) + log_ndtr(threshold - snr))


def update_nulls(prior_weights, log_null_probabilities):
    """Update using rows=independent exposures, columns=hypotheses.

    Inputs can be log(1 - injection recovery probability), or gaussian_log_null.
    Use only binary non-detections here, never a second use of flux data already
    used in the prior. This reweights a supplied family; it creates no new orbits.
    """
    prior = _weights(prior_weights)
    log_null = np.asarray(log_null_probabilities, dtype=float)
    if log_null.ndim != 2 or log_null.shape[1] != len(prior):
        raise ValueError("Null likelihood must have shape (exposures, hypotheses)")
    if np.any(np.isnan(log_null)) or np.any(log_null > 0):
        raise ValueError("Log probabilities must lie in [-infinity, 0]")
    with np.errstate(divide="ignore"):
        log_joint = np.log(prior) + log_null.sum(axis=0)
    evidence = logsumexp(log_joint)
    if not np.isfinite(evidence):
        raise ValueError("All hypotheses assign zero probability to these non-detections")
    posterior = np.exp(log_joint - evidence)
    return {"posterior_weights": posterior.tolist(), "log_evidence": float(evidence),
            "effective_sample_size": float(1 / np.sum(posterior**2)),
            "entropy_nats": float(-np.sum(xlogy(posterior, posterior)))}


def rank_exposures(weights, detection_probabilities):
    """Expected information gain of a future binary outcome, in nats.

    Rows are candidate exposures, columns the same hypotheses as weights.
    Equal detection probability for every hypothesis conveys zero information,
    even when that probability is high. Candidates are ranked individually.
    """
    weights = _weights(weights)
    probabilities = np.asarray(detection_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(weights):
        raise ValueError("Detection probabilities must have shape (exposures, hypotheses)")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Detection probabilities must lie in [0, 1]")
    def entropy(p):
        return -xlogy(p, p) - xlogy(1 - p, 1 - p)
    marginal = probabilities @ weights
    gain = np.maximum(0, entropy(marginal) - entropy(probabilities) @ weights)
    return [{"index": int(i), "expected_information_nats": float(gain[i]),
             "detection_probability": float(marginal[i])}
            for i in np.argsort(-gain, kind="stable")]
