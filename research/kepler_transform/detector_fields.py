"""Finite, nonperiodic Gaussian-pixel likelihood fields with tail bounds.

Zero padding is a computational construction, not a periodic camera model.
Explicit bounds charge the artificial periodic copies and Fourier truncation.
Each exposure can have different known variances, masks and Gaussian widths.
An optional constant background is profiled independently in each exposure.
WCS errors, trailing and non-Gaussian PSFs remain separate.
"""

import numpy as np
from flint import arb, ctx
from scipy.signal import fftconvolve
from scipy.special import ndtr

from research.detection_limit.interval_orbits import upper_float


def finite_pixel_fields(images, variances, sigma_pixels, pixel_scale_radians, *,
                        masks=None, source_half_width_pixels=None,
                        spectral_sigma=9., max_modes=3_000_000,
                        profile_background=False, flux_scales=1.):
    images = np.asarray(images, float)
    if images.ndim != 3 or images.shape[1] != images.shape[2] or images.shape[1] % 2 != 1 or min(images.shape) < 1:
        raise ValueError("Nonempty odd square exposure stamps required")
    frames, n, _ = images.shape
    variances = np.broadcast_to(np.asarray(variances, float), images.shape)
    sigma = np.broadcast_to(np.asarray(sigma_pixels, float), (frames, 2))
    flux_scales = np.broadcast_to(np.asarray(flux_scales, float), (frames,))
    half = (n-1)/2 if source_half_width_pixels is None else float(source_half_width_pixels)
    if (not all(np.all(np.isfinite(x)) for x in [images, variances, sigma, flux_scales])
            or np.any(variances <= 0) or np.any(sigma <= 0)
            or np.any(flux_scales <= 0) or not isinstance(profile_background, bool)
            or not np.isfinite(pixel_scale_radians) or pixel_scale_radians <= 0
            or not np.isfinite(half) or half < 0
            or not np.isfinite(spectral_sigma) or not 3 <= spectral_sigma <= 14
            or isinstance(max_modes, bool) or not isinstance(max_modes, int) or max_modes < 1):
        raise ValueError("Finite data, positive variances/scales, bounded spectral accuracy and mode budget required")
    weight = 1/variances
    if masks is not None:
        mask = np.asarray(masks)
        if mask.shape != images.shape:
            raise ValueError("Mask shape must match data")
        weight = np.where(mask != 0, 0., weight)
    if profile_background:
        weight_sums = np.sum(weight, axis=(1, 2))
        means = np.divide(np.sum(weight*images, axis=(1, 2)), weight_sums,
                          out=np.zeros(frames), where=weight_sums > 0)
        images = images-means[:, None, None]
    specifications = []
    for sx, sy in sigma:
        # Keep the nearest wrapped copy at least 10 sigma from every
        # allowed source/pixel displacement.
        period = max(n, int(np.ceil((n-1)/2+half+.5+10*max(sx, sy))))
        period += 1-period % 2
        cutoff = int(np.ceil(period*spectral_sigma/(2*np.pi*min(sx, sy))))
        specifications.append((period, cutoff))
    if sum((4*k+1)**2 for _, k in specifications) > max_modes:
        raise ValueError("Finite detector Fourier-mode budget exceeded")
    all_visits, all_frequencies, all_n, all_i, reports = [], [], [], [], []
    for visit, ((period, cutoff), widths) in enumerate(zip(specifications, sigma)):
        axis = np.arange(-cutoff, cutoff+1)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        kx, ky = xx*2*np.pi/period, yy*2*np.pi/period
        psf = np.exp(-.5*(widths[0]**2*kx**2+widths[1]**2*ky**2))*np.sinc(kx/(2*np.pi))*np.sinc(ky/(2*np.pi))
        extended = np.arange(-2*cutoff, 2*cutoff+1)
        ey, ex = np.meshgrid(extended, extended, indexing="ij")
        frequencies = np.column_stack((ex.ravel(), ey.ravel()))*2*np.pi/(period*pixel_scale_radians)
        pad = np.zeros((period, period))
        lo = (period-n)//2
        pad[lo:lo+n, lo:lo+n] = weight[visit]*images[visit]
        fft_data = np.fft.fft2(np.fft.ifftshift(pad))
        pad[lo:lo+n, lo:lo+n] = weight[visit]
        fft_weight = np.fft.fft2(np.fft.ifftshift(pad))
        numerator = np.zeros((4*cutoff+1, 4*cutoff+1), complex)
        numerator[cutoff:3*cutoff+1, cutoff:3*cutoff+1] = psf*fft_data[np.ix_(axis % period, axis % period)]/period**2
        information = fftconvolve(psf, psf, mode="full")/period**4*fft_weight[np.ix_(extended % period, extended % period)]
        if profile_background and weight_sums[visit] > 0:
            background_overlap = psf*fft_weight[np.ix_(axis % period, axis % period)]/period**2
            information -= fftconvolve(background_overlap, background_overlap, mode="full")/weight_sums[visit]
        numerator *= flux_scales[visit]
        information *= flux_scales[visit]**2
        with ctx.workprec(128):
            tails, wraps = [], []
            for width in widths:
                s = arb(float(width))
                tails.append((arb(2).sqrt()*arb.pi()*s*cutoff/period).erfc()/(s*(2*arb.pi()).sqrt()))
                gap = arb(period)-arb(n-1)/2-arb(half)-arb(".5")
                wraps.append((gap/(arb(2).sqrt()*s)).erfc())
            delta = tails[0]+tails[1]+tails[0]*tails[1]+sum(wraps, arb(0))
            n_error = delta*sum((abs(arb(float(x))) for x in (weight[visit]*images[visit]).ravel()), arb(0))
            i_error = delta*(2+delta)*sum((arb(float(x)) for x in weight[visit].ravel()), arb(0))
            if profile_background:
                i_error *= 2
            n_error *= arb(float(flux_scales[visit]))
            i_error *= arb(float(flux_scales[visit]))**2
            reports.append({"period_pixels": period, "frequency_cutoff": cutoff,
                            "template_absolute_error_bound": upper_float(delta),
                            "numerator_model_error_bound": upper_float(n_error),
                            "information_model_error_bound": upper_float(i_error)})
        all_visits.append(np.full(len(frequencies), visit, int))
        all_frequencies.append(frequencies)
        all_n.append(numerator.ravel())
        all_i.append(information.ravel())
    report = {"model": "finite detector; unrenormalized pixel-integrated separable Gaussian PSF",
              "source_half_width_pixels": half, "sigma_pixels": sigma.tolist(),
              "pixel_scale_radians": pixel_scale_radians,
              "background": "independent profiled constant per exposure" if profile_background else "known, already subtracted",
              "flux_scales": flux_scales.tolist(),
              "known_independent_gaussian_variances": True,
              "exposures": reports, "periodic_camera_assumed": False,
              "fourier_coefficient_arithmetic_certified": False}
    return (np.concatenate(all_visits), np.concatenate(all_frequencies),
            np.concatenate(all_n), np.concatenate(all_i), report)


def direct_pixel_statistics(images, variances, sigma_pixels, positions, *, masks=None,
                            profile_background=False, flux_scales=1.):
    """Independent real-space Gaussian integrals; no FFT or periodic copies."""
    images, positions = np.asarray(images, float), np.asarray(positions, float)
    if (images.ndim != 3 or images.shape[1] != images.shape[2] or images.shape[1] % 2 != 1
            or positions.ndim != 3 or positions.shape[1:] != (len(images), 2)
            or not np.all(np.isfinite(images)) or not np.all(np.isfinite(positions))):
        raise ValueError("Odd square images and finite (queries, frames, 2) positions required")
    variances = np.broadcast_to(np.asarray(variances, float), images.shape)
    sigma = np.broadcast_to(np.asarray(sigma_pixels, float), (len(images), 2))
    flux_scales = np.broadcast_to(np.asarray(flux_scales, float), (len(images),))
    if (np.any(variances <= 0) or np.any(sigma <= 0) or not np.all(np.isfinite(variances))
            or not np.all(np.isfinite(sigma)) or not isinstance(profile_background, bool)
            or not np.all(np.isfinite(flux_scales)) or np.any(flux_scales <= 0)):
        raise ValueError("Positive finite variances and PSF widths required")
    weight = 1/variances
    if masks is not None:
        if np.asarray(masks).shape != images.shape:
            raise ValueError("Mask shape must match data")
        weight = np.where(np.asarray(masks) != 0, 0., weight)
    axis = np.arange(images.shape[1])-(images.shape[1]-1)/2
    numerator, information = np.zeros(len(positions)), np.zeros(len(positions))
    for visit, (sx, sy) in enumerate(sigma):
        x = positions[:, visit, 0, None]
        y = positions[:, visit, 1, None]
        px = ndtr((axis+.5-x)/sx)-ndtr((axis-.5-x)/sx)
        py = ndtr((axis+.5-y)/sy)-ndtr((axis-.5-y)/sy)
        templates = py[:, :, None]*px[:, None, :]
        templates *= flux_scales[visit]
        numerator += np.einsum("nij,ij->n", templates, weight[visit]*images[visit])
        information += np.einsum("nij,ij->n", templates**2, weight[visit])
        weight_sum = np.sum(weight[visit])
        if profile_background and weight_sum > 0:
            overlap = np.einsum("nij,ij->n", templates, weight[visit])
            numerator -= overlap*np.sum(weight[visit]*images[visit])/weight_sum
            information -= overlap**2/weight_sum
    return numerator, information
