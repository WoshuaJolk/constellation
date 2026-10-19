"""Known-position PSF fitting with a fitted constant local background."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class Measurement:
    flux: float
    flux_error: float
    background: float
    chi2: float
    dof: int
    n_pixels: int
    psf_fraction: float
    x: float
    y: float

    @property
    def snr(self):
        return self.flux / self.flux_error

    def to_dict(self):
        return {**asdict(self), "snr": self.snr}


def gaussian_psf(x_pixels, y_pixels, x, y, sigma):
    """Pixel-integrated circular Gaussian of unit total flux on the infinite plane.

    Pixel coordinates are zero-based centers. Never renormalize a cutout: missing
    wings, masks and chip edges must reduce information, not change flux units.
    """
    if not np.all(np.isfinite([x, y, sigma])) or sigma <= 0:
        raise ValueError("Position and PSF sigma must be finite; sigma must be positive")
    px = ndtr((np.asarray(x_pixels) + 0.5 - x) / sigma) - ndtr(
        (np.asarray(x_pixels) - 0.5 - x) / sigma
    )
    py = ndtr((np.asarray(y_pixels) + 0.5 - y) / sigma) - ndtr(
        (np.asarray(y_pixels) - 0.5 - y) / sigma
    )
    return px * py


def trailed_psf(x_pixels, y_pixels, x, y, sigma, trail_dx=0, trail_dy=0):
    """Constant-rate, constant-flux exposure, centered on the supplied midpoint."""
    if not np.all(np.isfinite([trail_dx, trail_dy, sigma])) or sigma <= 0:
        raise ValueError("Trail must be finite and PSF sigma positive")
    length = np.hypot(trail_dx, trail_dy)
    if length == 0:
        return gaussian_psf(x_pixels, y_pixels, x, y, sigma)
    count = max(16, int(np.ceil(4 * length / sigma)))
    if count > 256:
        raise ValueError("Trail exceeds 64 PSF sigma; split the model into shorter exposures")
    nodes, weights = leggauss(count)
    return sum(weight / 2 * gaussian_psf(x_pixels, y_pixels, x + node * trail_dx / 2,
                                        y + node * trail_dy / 2, sigma)
               for node, weight in zip(nodes, weights))


def forced_photometry(image, variance, x, y, psf_sigma, mask=None, trail_dx=0, trail_dy=0):
    """Fit image = flux * PSF + background using known independent variances.

    Invalid image/variance pixels and nonzero mask pixels are excluded. The
    centroid must lie within the image; partly clipped PSFs are supported.
    Source Poisson noise, if relevant, must already be in the supplied variance.
    """
    image = np.asarray(image, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("Image must be a 2D array at least 2 by 2")
    if variance.ndim == 0:
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Scalar variance must be finite and positive")
        variance = np.broadcast_to(variance, image.shape)
    if variance.shape != image.shape:
        raise ValueError("Variance must be scalar or match image shape")
    if mask is not None and np.shape(mask) != image.shape:
        raise ValueError("Mask must match image shape")
    if not np.all(np.isfinite([x, y, psf_sigma])) or psf_sigma <= 0:
        raise ValueError("Position and PSF sigma must be finite; sigma must be positive")
    height, width = image.shape
    if not (-0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5):
        raise ValueError("Predicted centroid is outside the image footprint")
    if not np.all(np.isfinite([trail_dx, trail_dy])):
        raise ValueError("Trail offsets must be finite")
    radius = int(np.ceil(5 * psf_sigma + max(abs(trail_dx), abs(trail_dy)) / 2))
    x0, x1 = max(0, int(np.floor(x)) - radius), min(width, int(np.floor(x)) + radius + 2)
    y0, y1 = max(0, int(np.floor(y)) - radius), min(height, int(np.floor(y)) + radius + 2)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    psf = trailed_psf(xs, ys, x, y, psf_sigma, trail_dx, trail_dy)
    data = image[y0:y1, x0:x1]
    var = variance[y0:y1, x0:x1]
    good = np.isfinite(data) & np.isfinite(var) & (var > 0)
    if mask is not None:
        good &= np.asarray(mask)[y0:y1, x0:x1] == 0
    if np.count_nonzero(good) < 3:
        raise ValueError("At least three valid pixels are required")
    p, d, w = psf[good], data[good], 1 / var[good]
    p_mean, d_mean = np.average(p, weights=w), np.average(d, weights=w)
    centered = p - p_mean
    information = np.sum(w * centered**2)
    if not np.isfinite(information) or information <= np.finfo(float).tiny:
        raise ValueError("No usable PSF information remains after masking/background fit")
    flux = np.sum(w * centered * (d - d_mean)) / information
    background = d_mean - flux * p_mean
    chi2 = np.sum(w * (d - flux * p - background)**2)
    if not np.all(np.isfinite([flux, background, chi2])):
        raise ValueError("Non-finite fit; check the image and variance dynamic range")
    return Measurement(float(flux), float(1 / np.sqrt(information)), float(background),
                       float(chi2), int(good.sum() - 2), int(good.sum()), float(p.sum()),
                       float(x), float(y))
