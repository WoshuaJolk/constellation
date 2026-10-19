import numpy as np
import pytest
from scipy.integrate import quad

from magictools.photometry import forced_photometry, gaussian_psf, trailed_psf
from magictools.stamps import template_evidence, weigh_stamp


def test_positive_prior_evidence_matches_numerical_integration():
    d = np.array([1.2, 2.1, 5.0, 1.5, 0.3])
    v = np.array([1, 2, 1, 2, 3])
    p = np.array([0.05, 0.2, 0.5, 0.2, 0.05])
    sigma = 8
    w = 1 / v
    def log_likelihood(flux):
        residual = d - flux * p
        residual -= np.average(residual, weights=w)
        return -0.5 * np.sum(w * residual**2)
    def integrand(flux):
        return np.sqrt(2 / np.pi) / sigma * np.exp(-flux**2 / (2 * sigma**2) + log_likelihood(flux) - log_likelihood(0))
    expected = np.log(quad(integrand, 0, np.inf, epsabs=1e-10)[0])
    actual, _ = template_evidence(d, v, p[None, :], sigma)
    assert actual[0] == pytest.approx(expected, abs=1e-10)


def test_trail_normalization_and_flux_recovery():
    yy, xx = np.mgrid[:51, :51]
    p = trailed_psf(xx, yy, 25.2, 24.8, 1.2, 12, -6)
    assert p.sum() == pytest.approx(1, abs=1e-12)
    measured = forced_photometry(71 + 300 * p, 4, 25.2, 24.8, 1.2, trail_dx=12, trail_dy=-6)
    assert measured.flux == pytest.approx(300)
    assert measured.background == pytest.approx(71)
    assert measured.chi2 < 1e-20
    np.testing.assert_allclose(trailed_psf(xx, yy, 25.2, 24.8, 1.2, 0, 0), gaussian_psf(xx, yy, 25.2, 24.8, 1.2))


@pytest.mark.parametrize("kind", ["moving_source", "cosmic_ray_spike", "extended_source", "dipole_residual", "background", "stationary_point", "narrow_streak"])
def test_forward_model_controls(kind):
    yy, xx = np.mgrid[:25, :25]
    image = np.full((25, 25), 30.0)
    if kind == "moving_source":
        image += 400 * trailed_psf(xx, yy, 12, 12, 1.1, 8, 2)
    elif kind == "cosmic_ray_spike":
        image[9, 14] += 400
    elif kind == "extended_source":
        image += 400 * gaussian_psf(xx, yy, 12, 12, 2.2)
    elif kind == "dipole_residual":
        image += 400 * (gaussian_psf(xx, yy, 12 - 0.55, 12, 1.1) - gaussian_psf(xx, yy, 12 + 0.55, 12, 1.1))
    elif kind == "stationary_point":
        image += 400 * gaussian_psf(xx, yy, 12, 12, 1.1)
    elif kind == "narrow_streak":
        image += 400 * trailed_psf(xx, yy, 12, 12, 0.35, 6, 0)
    result = weigh_stamp(image, 4, 12, 12, 1.1, 8, 2, 500)
    assert result["preferred_model"] == kind
    assert sum(row["posterior_within_model_set"] for row in result["models"].values()) == pytest.approx(1)


def test_no_motion_cannot_distinguish_stationary_source():
    yy, xx = np.mgrid[:15, :15]
    image = 100 * gaussian_psf(xx, yy, 7, 7, 1)
    result = weigh_stamp(image, 1, 7, 7, 1, 0, 0, 100)
    assert result["models"]["stationary_point"]["log_bayes_factor_moving_vs_model"] == pytest.approx(0)


def test_masked_cosmic_ray_cannot_bias_source_comparison():
    yy, xx = np.mgrid[:25, :25]
    image = 400 * trailed_psf(xx, yy, 12, 12, 1.1, 8, 2)
    image[2, 2] = 1e8
    mask = np.zeros(image.shape)
    mask[2, 2] = 1
    result = weigh_stamp(image, 4, 12, 12, 1.1, 8, 2, 500, mask)
    assert result["preferred_model"] == "moving_source"


def test_template_evidence_prior_scale_matters():
    data = np.array([0, 0, 5, 0, 0])
    template = np.array([[0, 0, 1, 0, 0]])
    near, _ = template_evidence(data, np.ones(5), template, 10)
    broad, _ = template_evidence(data, np.ones(5), template, 1e6)
    assert broad[0] < near[0]
