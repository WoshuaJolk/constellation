import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import norm

from magictools.prove_floor import (calibrate_detector, mixture_log_lr, necessary_flux,
    power_bounds, projected_templates, run_floor, validate_gram)


def test_single_template_matches_exact_neyman_pearson_roc():
    gram = np.array([[2.25]])
    alpha, target = 0.01, 0.9
    expected = (norm.isf(alpha) + norm.ppf(target)) / 1.5
    assert necessary_flux(gram, [1], alpha, target) == pytest.approx(expected, rel=1e-8)
    flux = 2
    bound = power_bounds(gram, [1], flux, alpha)
    assert bound["oracle_power_upper_bound"] == pytest.approx(norm.cdf(3 - norm.isf(alpha)))
    assert bound["best_power_upper_bound"] >= norm.cdf(3 - norm.isf(alpha)) - 1e-12


def test_chi_square_identity_for_symmetric_and_identical_templates():
    flux = 0.7
    opposite = power_bounds([[1, -1], [-1, 1]], [1, 1], flux, 0.05)
    assert np.expm1(opposite["log_null_second_moment_of_likelihood_ratio"]) == pytest.approx(np.cosh(flux**2) - 1)
    identical = power_bounds([[1, 1], [1, 1]], [0.2, 0.8], flux, 0.05)
    assert identical["log_null_second_moment_of_likelihood_ratio"] == pytest.approx(flux**2)


def test_divergence_bound_is_usefully_stronger_than_oracle():
    result = power_bounds(np.eye(20), np.ones(20), 1, 0.01)
    assert result["best_power_upper_bound"] < result["oracle_power_upper_bound"]
    # Can be checked directly without Monte Carlo: chi^2 = (exp(F^2)-1)/20.
    expected = 0.01 + np.sqrt(0.01 * 0.99 * np.expm1(1) / 20)
    assert result["best_power_upper_bound"] == pytest.approx(expected)


def test_correlated_noise_background_projection_matches_matrix_formula():
    templates = np.array([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=float)
    covariance = np.diag([1, 2, 3, 4]) + 0.3 * np.ones((4, 4))
    design = np.ones((4, 1))
    observation = np.array([2, 4, 5, 1])
    gram, scores, rank = projected_templates(templates, covariance, design, observation)
    precision = np.linalg.inv(covariance)
    projected_precision = precision - precision @ design @ np.linalg.inv(design.T @ precision @ design) @ design.T @ precision
    np.testing.assert_allclose(gram, templates @ projected_precision @ templates.T, atol=1e-12)
    np.testing.assert_allclose(scores, templates @ projected_precision @ observation, atol=1e-12)
    assert rank == 1


def test_background_degeneracy_and_inaccessible_alternative():
    gram, _, _ = projected_templates([[1, 1, 1]], np.eye(3), np.ones((3, 1)))
    assert gram[0, 0] == 0
    assert necessary_flux(gram, [1], 0.01, 0.9) is None
    assert power_bounds(gram, [1], 10, 0.01)["best_power_upper_bound"] == pytest.approx(0.01)
    assert necessary_flux([[1, 0], [0, 0]], [1, 1], 0.01, 0.9) is None


def test_singular_correlated_bank_and_calibration_guarantee():
    result = calibrate_detector([[1, 1], [1, 1]], [1, 1], 3.5, alpha=0.05,
                               calibration_samples=8192, evaluation_samples=8192, seed=14)
    assert result["null_rejection_probability_upper_bound"] <= 0.05
    # In the one-template case the actual calibrated power is known exactly.
    z_threshold = (result["threshold_log_lr"] + 3.5**2 / 2) / 3.5
    true_power = norm.sf(z_threshold - 3.5)
    assert abs(result["mixture_power_estimate"] - true_power) < 0.02
    assert result["null_holdout_rejection_fraction"] < 0.06


def test_null_moment_matches_independent_direct_likelihood():
    templates = np.array([[1, 0.3], [-0.2, 0.8]])
    gram = templates @ templates.T
    y = np.array([0.4, -0.7])
    flux, weights = 0.8, np.array([0.4, 0.6])
    expected = np.log(sum(w * np.exp(-0.5 * np.sum((y - flux * t)**2) + 0.5 * np.sum(y**2)) for w, t in zip(weights, templates)))
    assert mixture_log_lr(templates @ y, gram, weights, flux) == pytest.approx(expected)


@pytest.mark.parametrize("gram", [[[1, 2], [2, 1]], [[1, 0], [1, 1]], [[np.nan]]])
def test_invalid_gram(gram):
    with pytest.raises(ValueError):
        validate_gram(gram)


def test_insufficient_calibration_is_not_silently_accepted():
    with pytest.raises(ValueError, match="Too few"):
        calibrate_detector([[1]], [1], 2, alpha=0.001, calibration_samples=100)


def test_cli_gram_input_and_pixel_input_agree(tmp_path):
    common = {"schema_version": 1, "prior_weights": [1], "flux": 3, "alpha": 0.05,
              "calibration_samples": 512, "evaluation_samples": 512, "seed": 5}
    path = tmp_path / "gram.json"
    path.write_text(json.dumps({**common, "gram_matrix": [[1]]}))
    result = run_floor(path)
    path2 = tmp_path / "pixels.json"
    path2.write_text(json.dumps({**common, "templates": [[2]], "variance": [4], "observation": [8]}))
    pixels = run_floor(path2)
    assert pixels["bounds"] == result["bounds"]
    assert pixels["detector"]["observation"]["rejects_gaussian_null"]
    process = subprocess.run([sys.executable, "-m", "magictools", "prove_floor", str(path)], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["tool"] == "prove_floor"
