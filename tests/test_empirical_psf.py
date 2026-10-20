import numpy as np
import pytest

from magictools.photometry import forced_photometry, sampled_psf, validate_psf_kernel


@pytest.fixture
def kernel():
    yy, xx = np.mgrid[-5:6, -5:6]
    # Deliberately asymmetric, non-Gaussian template.
    return np.exp(-((xx - .35)**2 / 2 + yy**2 / 4)) + .15 * np.exp(-((xx + 2)**2 + (yy - 1)**2))


def test_flux_conservation_under_fractional_shifts_and_trails(kernel):
    yy, xx = np.mgrid[:70, :70]
    for x, y, dx, dy in [(30, 30, 0, 0), (30.33, 29.12, 0, 0), (30.33, 29.12, 10.4, -3.1)]:
        template = sampled_psf(xx, yy, x, y, kernel, dx, dy)
        assert template.sum() == pytest.approx(1, abs=1e-12)
        assert template.min() >= 0


def test_integer_shift_matches_independently_placed_kernel(kernel):
    yy, xx = np.mgrid[:40, :40]
    truth = np.zeros((40, 40))
    truth[12:23, 17:28] = kernel / kernel.sum()
    np.testing.assert_allclose(sampled_psf(xx, yy, 22, 17, kernel), truth, atol=1e-15)
    fit = forced_photometry(230 * truth + 17, 4, 22, 17, 1, psf_kernel=kernel)
    assert fit.flux == pytest.approx(230)
    assert fit.background == pytest.approx(17)


def test_fractional_shift_matches_independent_bilinear_superposition(kernel):
    truth = np.zeros((40, 40))
    for dx, dy, weight in [(0, 0, .7*.6), (1, 0, .3*.6), (0, 1, .7*.4), (1, 1, .3*.4)]:
        truth[12+dy:23+dy, 17+dx:28+dx] += weight * kernel / kernel.sum()
    yy, xx = np.indices(truth.shape)
    np.testing.assert_allclose(sampled_psf(xx, yy, 22.3, 17.4, kernel), truth, atol=1e-15)


def test_clipping_and_masks_reduce_information_without_flux_renormalization(kernel):
    yy, xx = np.mgrid[:35, :35]
    center = forced_photometry(200 * sampled_psf(xx, yy, 17, 17, kernel) + 4, 9, 17, 17, 1, psf_kernel=kernel)
    data = 200 * sampled_psf(xx, yy, .2, 17, kernel) + 4
    mask = np.zeros(data.shape); mask[17, 1] = 1
    edge = forced_photometry(data, 9, .2, 17, 1, mask=mask, psf_kernel=kernel)
    assert edge.flux == pytest.approx(200)
    assert edge.psf_fraction < center.psf_fraction
    assert edge.flux_error > center.flux_error


@pytest.mark.parametrize("kernel", [np.ones((4, 3)), np.zeros((5, 5)), np.full((5, 5), np.nan)])
def test_invalid_kernel_rejected(kernel):
    with pytest.raises(ValueError):
        validate_psf_kernel(kernel)
