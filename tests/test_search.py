from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

from magictools.search import combine_measurements


def measurements(flux, error):
    return [SimpleNamespace(flux=f, flux_error=e) for f, e in zip(flux, error)]


def test_scaled_joint_flux_and_sensitivity():
    result = combine_measurements(measurements([10, 20, 5], [2, 4, 1]), [1, 2, 0.5])
    assert result["flux"] == pytest.approx(10)
    assert result["flux_error"] == pytest.approx(2 / np.sqrt(3))
    assert result["five_sigma_flux_sensitivity"] == pytest.approx(10 / np.sqrt(3))
    assert result["constant_flux_chi2"] == pytest.approx(0)


def test_negative_flux_is_preserved_and_not_detected():
    result = combine_measurements(measurements([5, -7], [1, 1]))
    assert result["flux"] == pytest.approx(-1)
    assert result["positive_source_log_likelihood_ratio"] == 0
    assert not result["passes_gaussian_threshold"]
    assert result["p_single"] > 0.5


def test_trials_correction():
    result = combine_measurements(measurements([3], [1]), trials=100)
    assert result["p_family_upper_bound"] == pytest.approx(100 * norm.sf(3))
    assert not result["passes_gaussian_threshold"]
    assert combine_measurements(measurements([3], [1]))["passes_gaussian_threshold"]


def test_joint_matches_direct_pixel_fit_with_separate_backgrounds():
    from magictools.photometry import forced_photometry, gaussian_psf
    rng = np.random.default_rng(44)
    yy, xx = np.mgrid[:9, :9]
    design = np.zeros((3 * 81, 4))
    data, errors, measured = [], [], []
    scales = [0.5, 1.2, 2]
    for i, scale in enumerate(scales):
        p = gaussian_psf(xx, yy, 3.8, 4.1, 1 + i * 0.2)
        image = 24 * scale * p + 100 * i + rng.normal(0, i + 1, p.shape)
        design[i * 81:(i + 1) * 81, 0] = scale * p.ravel()
        design[i * 81:(i + 1) * 81, i + 1] = 1
        data.extend(image.ravel())
        errors.extend([i + 1] * 81)
        measured.append(forced_photometry(image, (i + 1)**2, 3.8, 4.1, 1 + i * 0.2))
    a = design / np.array(errors)[:, None]
    expected = np.linalg.lstsq(a, np.array(data) / errors, rcond=None)[0]
    covariance = np.linalg.inv(a.T @ a)
    result = combine_measurements(measured, scales)
    assert result["flux"] == pytest.approx(expected[0])
    assert result["flux_error"] == pytest.approx(np.sqrt(covariance[0, 0]))


@pytest.mark.parametrize("kwargs", [{"trials": 0}, {"alpha": 0}, {"flux_scales": [0]}, {"flux_scales": [1, 1]}])
def test_bad_search_parameters(kwargs):
    with pytest.raises(ValueError):
        combine_measurements(measurements([1], [1]), **kwargs)
