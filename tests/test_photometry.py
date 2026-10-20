import numpy as np
import pytest
from scipy.integrate import quad

from magictools.photometry import forced_photometry, gaussian_psf


def test_pixel_integral_and_total_flux():
    x, y, sigma = 0.23, -0.17, 0.7
    def density(v, center):
        return np.exp(-0.5 * ((v - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    expected = quad(density, -0.5, 0.5, args=(x,))[0] * quad(density, -0.5, 0.5, args=(y,))[0]
    assert gaussian_psf(0, 0, x, y, sigma) == pytest.approx(expected, rel=1e-12)
    yy, xx = np.mgrid[-20:21, -20:21]
    assert gaussian_psf(xx, yy, x, y, sigma).sum() == pytest.approx(1)


@pytest.mark.parametrize("x,y", [(15.3, 14.8), (0.1, 0.3)])
def test_flux_and_background_recovery_with_mask_and_chip_edge(x, y):
    yy, xx = np.mgrid[:31, :31]
    psf = gaussian_psf(xx, yy, x, y, 1.2)
    image = 123.4 * psf + 17.3
    mask = np.zeros(image.shape)
    mask[1, 1] = 1
    image[1, 1] = 1e9  # masked cosmic ray cannot bias the fit
    image[2, 2] = np.nan
    variance = np.full(image.shape, 4.0)
    variance[3, 3] = 0
    result = forced_photometry(image, variance, x, y, 1.2, mask)
    assert result.flux == pytest.approx(123.4)
    assert result.background == pytest.approx(17.3)
    assert result.chi2 < 1e-20
    if x < 1:
        assert result.psf_fraction < 0.6


def test_matches_independent_weighted_matrix_solution():
    rng = np.random.default_rng(31)
    yy, xx = np.mgrid[:11, :11]
    psf = gaussian_psf(xx, yy, 5.2, 4.8, 1.2)
    variance = 1 + rng.random(psf.shape) * 3
    image = -12 * psf + 20 + rng.normal(size=psf.shape) * np.sqrt(variance)
    design = np.column_stack([psf.ravel(), np.ones(psf.size)])
    weight = np.diag(1 / variance.ravel())
    covariance = np.linalg.inv(design.T @ weight @ design)
    expected = covariance @ design.T @ weight @ image.ravel()
    result = forced_photometry(image, variance, 5.2, 4.8, 1.2)
    assert result.flux == pytest.approx(expected[0])
    assert result.background == pytest.approx(expected[1])
    assert result.flux_error == pytest.approx(np.sqrt(covariance[0, 0]))


def test_noise_only_uncertainty_calibration():
    rng = np.random.default_rng(813)
    snrs = [forced_photometry(rng.normal(40, 3, (17, 17)), 9, 8.3, 7.6, 1.2).snr for _ in range(1200)]
    assert abs(np.mean(snrs)) < 0.10
    assert 0.92 < np.std(snrs, ddof=1) < 1.08


@pytest.mark.parametrize("variance,x,y,sigma", [(0, 5, 5, 1), (1, 12, 5, 1), (1, 5, 5, 0), (1, np.nan, 5, 1)])
def test_invalid_inputs(variance, x, y, sigma):
    with pytest.raises(ValueError):
        forced_photometry(np.zeros((11, 11)), variance, x, y, sigma)


def test_no_usable_pixels():
    with pytest.raises(ValueError, match="three valid"):
        forced_photometry(np.zeros((11, 11)), 1, 5, 5, 1, np.ones((11, 11)))
