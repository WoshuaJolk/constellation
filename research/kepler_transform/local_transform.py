"""Localized polynomial geometry and phase-corrected 3D NUFFT construction.

For each fixed outer state, fit a quadratic apparent-position map in three
inner state coordinates from 27 exact retarded trajectories. Linear phase is
summed by a type-3 NUFFT; polynomial corrections handle quadratic phase. This
avoids integrating every requested trajectory during construction, but the
physical geometry-fit error is NOT certified. A slice is not a full 6D atlas.
"""

from itertools import product
from math import factorial
from time import perf_counter

import numpy as np
from scipy.signal import convolve2d

from .universal import apparent


LINEAR = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
QUADRATIC = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
POWERS = [(0, 0, 0)]+LINEAR+QUADRATIC


def monomials(points, powers=POWERS):
    return np.prod(np.asarray(points)[:, None, :]**np.asarray(powers)[None], axis=-1)


def check_grid_budget(source, destination, max_grid_cells=8_000_000):
    """Conservative work heuristic, NOT a guarantee on FINUFFT's allocation.

    Type-3 cost grows with the phase-space bounding box, even for few targets.
    Refuse obviously huge boxes before entering the native library.
    """
    bandwidth = np.ptp(source, axis=0)*np.ptp(destination, axis=0)/(2*np.pi)
    estimate = float(np.prod(np.maximum(32., 4*np.ceil(bandwidth)+32)))
    if not np.isfinite(estimate) or estimate > max_grid_cells:
        raise ValueError(f"NUFFT phase-volume budget exceeded: estimated {estimate:.0f} grid cells > {max_grid_cells}; partition the domain")
    return int(estimate)


class LocalTransform:
    def __init__(self, center, basis, geometry, *, correction_order=1, eps=1e-12):
        started = perf_counter()
        self.center, self.basis = np.asarray(center, float), np.asarray(basis, float)
        if (self.center.shape != (6,) or self.basis.shape != (3, 6)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.basis))
                or np.linalg.matrix_rank(self.basis) != 3
                or isinstance(correction_order, bool) or not isinstance(correction_order, int)
                or not 0 <= correction_order <= 4 or not np.isfinite(eps) or not 1e-14 <= eps < 1e-3):
            raise ValueError("Finite center, independent 3x6 basis, order 0..4 and valid NUFFT epsilon required")
        self.geometry, self.order, self.eps = geometry, correction_order, eps
        grid = np.array(list(product([-1., 0., 1.], repeat=3)))
        states = self.center+grid @ self.basis
        emitted, _ = apparent(states, geometry.times, geometry.observers)
        delta = emitted[..., :3]-geometry.observers
        denominator = np.einsum("nti,ti->nt", delta, geometry.normal)
        if np.any(denominator <= 0):
            raise ValueError("Quadratic stencil crosses the gnomonic horizon")
        positions = np.stack([np.einsum("nti,ti->nt", delta, axis)/denominator
                              for axis in (geometry.east, geometry.north)], axis=-1)
        self.position_coefficients = np.linalg.lstsq(monomials(grid), positions.reshape(27, -1),
                                                     rcond=None)[0].reshape(10, len(geometry.times), 2)
        self._compile(started, 27*len(geometry.times))

    @classmethod
    def from_position_coefficients(cls, center, basis, coefficients, geometry, *, correction_order=1, eps=1e-12):
        """Build from a previously constructed position atlas, without orbit sampling."""
        result = cls.__new__(cls)
        result.center, result.basis = np.asarray(center, float), np.asarray(basis, float)
        result.position_coefficients = np.asarray(coefficients, float)
        if (result.center.shape != (6,) or result.basis.shape != (3, 6)
                or result.position_coefficients.shape != (10, len(geometry.times), 2)
                or not all(np.all(np.isfinite(x)) for x in
                           [result.center, result.basis, result.position_coefficients])
                or isinstance(correction_order, bool) or not isinstance(correction_order, int)
                or not 0 <= correction_order <= 4 or not np.isfinite(eps) or not 1e-14 <= eps < 1e-3):
            raise ValueError("Invalid position atlas slice")
        result.geometry, result.order, result.eps = geometry, correction_order, eps
        result._compile(perf_counter(), 0)
        return result

    def _compile(self, started, geometry_pairs):
        geometry, correction_order = self.geometry, self.order
        phases = np.einsum("bmi,mi->bm", self.position_coefficients[:, geometry.visits], geometry.frequencies)
        self.constant, self.linear, self.quadratic = phases[0], phases[1:4].T, phases[4:]
        # exp(i q(z)) = sum_s i^s q(z)^s/s!; combine equal monomials.
        power = {(0, 0, 0): np.ones(len(self.constant))}
        coefficients = {}
        for order in range(correction_order+1):
            for exponent, coefficient in power.items():
                coefficients[exponent] = coefficients.get(exponent, 0)+(1j**order/factorial(order))*coefficient
            following = {}
            for a, value in power.items():
                for b, quadratic in zip(QUADRATIC, self.quadratic):
                    exponent = tuple(x+y for x, y in zip(a, b))
                    following[exponent] = following.get(exponent, 0)+value*quadratic
            power = following
        self.powers = list(coefficients)
        self.coefficients = np.array(list(coefficients.values()))*np.exp(1j*self.constant)
        self.report = {"construction_seconds": perf_counter()-started,
                       "geometry_state_time_pairs": geometry_pairs,
                       "polynomial_terms": len(self.powers), "correction_order": correction_order,
                       "physical_geometry_error_certified": False,
                       "continuous_6d_domain_covered": False,
                       "nufft_requested_epsilon": self.eps}

    def apply(self, data, targets, *, batch_size=4):
        import finufft
        data, targets = np.asarray(data, complex), np.asarray(targets, float)
        if (data.shape != self.constant.shape or not np.all(np.isfinite(data))
                or targets.ndim != 2 or targets.shape[1] != 3 or not len(targets)
                or not np.all(np.isfinite(targets)) or np.max(abs(targets)) > 1
                or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("Finite compatible coefficients and targets in [-1,1]^3 required")
        result = np.zeros(len(targets), complex)
        check_grid_budget(self.linear, targets)
        source = [np.ascontiguousarray(self.linear[:, axis]) for axis in range(3)]
        destination = [np.ascontiguousarray(targets[:, axis]) for axis in range(3)]
        for start in range(0, len(self.powers), batch_size):
            stop = min(start+batch_size, len(self.powers))
            transformed = finufft.nufft3d3(*source,
                np.ascontiguousarray(self.coefficients[start:stop]*data[None]),
                *destination, eps=self.eps, isign=1, nthreads=1)
            result += np.sum(transformed.T*monomials(targets, self.powers[start:stop]), axis=1)
        return result

    def quadratic_direct(self, data, targets, chunk=256):
        """Exact exponential of the fitted quadratic map, without phase truncation."""
        output = []
        phase_coefficients = np.vstack((self.constant, self.linear.T, self.quadratic))
        for start in range(0, len(targets), chunk):
            phase = monomials(targets[start:start+chunk]) @ phase_coefficients
            output.append(np.exp(1j*phase) @ data)
        return np.concatenate(output)

    def adjoint(self, values, targets, *, batch_size=4):
        """Algebraic adjoint of the phase-corrected finite query operator."""
        import finufft
        values, targets = np.asarray(values, complex), np.asarray(targets, float)
        if (targets.ndim != 2 or targets.shape[1] != 3 or not len(targets)
                or values.shape != (len(targets),) or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(targets)) or np.max(abs(targets)) > 1
                or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("Compatible finite query values/targets required")
        result = np.zeros(len(self.constant), complex)
        check_grid_budget(targets, self.linear)
        source = [np.ascontiguousarray(targets[:, axis]) for axis in range(3)]
        destination = [np.ascontiguousarray(self.linear[:, axis]) for axis in range(3)]
        for start in range(0, len(self.powers), batch_size):
            stop = min(start+batch_size, len(self.powers))
            data = monomials(targets, self.powers[start:stop]).T*values[None]
            transformed = finufft.nufft3d3(*source, np.ascontiguousarray(data),
                                          *destination, eps=self.eps, isign=-1, nthreads=1)
            result += np.sum(self.coefficients[start:stop].conj()*transformed, axis=0)
        return result

    def inverse(self, scores, targets, *, damping=0., tolerance=1e-9, max_iterations=200):
        """Regularized Fourier-coefficient recovery; nullspaces remain nullspaces."""
        from scipy.sparse.linalg import LinearOperator, lsqr
        targets, scores = np.asarray(targets, float), np.asarray(scores, complex)
        if (not np.isfinite(damping) or damping < 0 or not 0 < tolerance < 1
                or isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1
                or targets.ndim != 2 or targets.shape[1] != 3 or not len(targets)
                or not np.all(np.isfinite(targets)) or np.max(abs(targets)) > 1
                or scores.shape != (len(targets),) or not np.all(np.isfinite(scores))):
            raise ValueError("Invalid regularized inverse request")
        operator = LinearOperator((len(targets), len(self.constant)), dtype=complex,
            matvec=lambda x: self.apply(x, targets), rmatvec=lambda y: self.adjoint(y, targets))
        result = lsqr(operator, scores, damp=damping, atol=tolerance, btol=tolerance, iter_lim=max_iterations)
        return result[0], {"method": "LSQR on the local phase-corrected Fourier operator",
                           "stop_code": result[1], "iterations": result[2], "residual_norm": result[3],
                           "condition_estimate": result[6], "iteration_limit_reached": result[1] == 7,
                           "damping": damping, "unique_image_reconstruction_claimed": False}

    def phase_tail_bound(self, data):
        """Outward Arb bound on phase-series truncation over [-1,1]^3.

        Applies to the exact represented quadratic model; excludes physical
        geometry interpolation, coefficient construction roundoff and NUFFT
        error. It cannot be reported as a total physical score certificate.
        """
        from flint import arb, ctx
        data = np.asarray(data, complex)
        if data.shape != self.constant.shape or not np.all(np.isfinite(data)):
            raise ValueError("Finite compatible Fourier coefficients required")
        with ctx.workprec(128):
            total = arb(0)
            for index, datum in enumerate(data):
                phase = sum((abs(arb(float(value))) for value in self.quadratic[:, index]), arb(0))
                amplitude = (arb(float(datum.real))**2+arb(float(datum.imag))**2).sqrt()
                total += amplitude*phase**(self.order+1)/factorial(self.order+1)
            bound = float(total.upper())
            return float(np.nextafter(bound, np.inf))


def pixel_fields(images, variances, sigma_pixels, pixel_scale_radians, *, masks=None):
    """Exact Fourier sufficient statistics for a declared periodic pixel model.

    Odd square detectors; a finite bandlimited pixel-integrated Gaussian PSF,
    known zero background, independent known Gaussian variances. Includes all
    frequencies of the SQUARED PSF so varying weights/masks are not replaced by
    constant information. Coordinates are centered on the central pixel.

    Returns visits, frequencies, numerator coefficients, information coefficients.
    Periodic wrap and finite bandwidth are model assumptions, not real-camera
    boundary conditions. Tests compare directly with templates in pixel space.
    """
    images = np.asarray(images, float)
    variances = np.broadcast_to(np.asarray(variances, float), images.shape)
    if (images.ndim != 3 or images.shape[1] != images.shape[2] or images.shape[1] % 2 != 1
            or min(images.shape) < 1 or not np.all(np.isfinite(images))
            or not np.all(np.isfinite(variances)) or np.any(variances <= 0)
            or not np.isfinite(sigma_pixels) or sigma_pixels <= 0
            or not np.isfinite(pixel_scale_radians) or pixel_scale_radians <= 0):
        raise ValueError("Finite odd square images, positive variances and PSF/pixel scales required")
    n, frames = images.shape[1], len(images)
    weight = 1/variances
    if masks is not None:
        mask = np.asarray(masks)
        if mask.shape != images.shape:
            raise ValueError("Mask shape must match images")
        weight = np.where(mask != 0, 0., weight)
    axis = np.arange(-(n//2), n//2+1)
    ky, kx = np.meshgrid(axis*2*np.pi/n, axis*2*np.pi/n, indexing="ij")
    psf_hat = np.exp(-.5*sigma_pixels**2*(kx**2+ky**2))*np.sinc(kx/(2*np.pi))*np.sinc(ky/(2*np.pi))
    squared = convolve2d(psf_hat, psf_hat, mode="full")/n**4
    extended = np.arange(-(n-1), n)
    yy, xx = np.meshgrid(extended, extended, indexing="ij")
    frequencies = np.column_stack((xx.ravel(), yy.ravel()))*2*np.pi/(n*pixel_scale_radians)
    numerator, information = [], []
    for frame in range(frames):
        data_fft = np.fft.fft2(np.fft.ifftshift(images[frame]*weight[frame]))
        weight_fft = np.fft.fft2(np.fft.ifftshift(weight[frame]))
        coeff = np.zeros_like(squared, dtype=complex)
        lo = n//2
        coeff[lo:lo+n, lo:lo+n] = psf_hat*data_fft[np.ix_(axis % n, axis % n)]/n**2
        numerator.append(coeff.ravel())
        information.append((squared*weight_fft[np.ix_(extended % n, extended % n)]).ravel())
    return (np.repeat(np.arange(frames), len(frequencies)), np.tile(frequencies, (frames, 1)),
            np.concatenate(numerator), np.concatenate(information))
