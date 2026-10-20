"""Finite Gaussian detector fields, including affine pixel geometry.

No image reprojection is performed. Pixel centers enter a nonuniform Fourier
sum, preserving the original masks and diagonal inverse variances.
"""

import numpy as np
from flint import arb, ctx
from scipy.signal import fftconvolve

from research.detection_limit.interval_orbits import upper_float
from ..detector_fields import direct_pixel_statistics


def l1_upper(values):
    """Outward upper bound on l1 of stored complex128 coefficients.

    Bound modulus by |Re|+|Im|, then use the standard nonnegative floating-point
    summation bound. This does not certify how the coefficients were computed.
    """
    a = np.asarray(values)
    if not np.all(np.isfinite(a)):
        raise ValueError("Finite coefficients required")
    n = max(1, 2 * a.size)
    u = np.finfo(float).eps / 2
    if n * u >= .01:
        raise ValueError("Summation bound exceeds supported size")
    total = float(np.sum(np.abs(a.real), dtype=np.float64) +
                  np.sum(np.abs(a.imag), dtype=np.float64))
    with ctx.workprec(128):
        return upper_float(arb(total) / (1 - arb(n + 2) * arb(u)))


def sum_bounds(values):
    with ctx.workprec(128):
        return upper_float(sum((arb(float(x)) for x in values), arb(0)))


def _geometry_bounds(matrix, sigma, offset, n, half):
    with ctx.workprec(128):
        b = [[arb(float(v)) for v in row] for row in matrix]
        s = [arb(float(v)) for v in sigma]
        cov = [[sum((b[i][k] * b[j][k] * s[k]**2 for k in range(2)), arb(0))
                for j in range(2)] for i in range(2)]
        trace = cov[0][0] + cov[1][1]
        disc = ((cov[0][0] - cov[1][1])**2 + 4 * cov[0][1]**2).sqrt()
        small, large = (trace - disc) / 2, (trace + disc) / 2
        area = abs(b[0][0] * b[1][1] - b[0][1] * b[1][0])
        if not small > 0 or not area > 0:
            raise ValueError("Nonsingular affine geometry and positive PSF covariance required")
        extent = [arb(float(half)) + abs(arb(float(offset[i]))) +
                  arb(n) / 2 * sum((abs(x) for x in b[i]), arb(0)) for i in range(2)]
        return small, large, area, extent


def template_bound(matrix, sigma, offset, n, half, period, cutoff):
    """Gaussian lattice tail + periodic-copy bound over the whole source box."""
    with ctx.workprec(128):
        small, large, area, extent = _geometry_bounds(matrix, sigma, offset, n, half)
        p = arb(float(period))
        a = small * (2 * arb.pi() / p)**2 / 2
        total_1d = 1 + (arb.pi() / a).sqrt()
        tail_1d = (arb.pi() / a).sqrt() * (a.sqrt() * cutoff).erfc()
        spectral = area / p**2 * 2 * total_1d * tail_1d
        wraps = []
        for d in extent:
            gap = p - d
            if not gap > 0:
                raise ValueError("Period does not separate artificial copies")
            ratio = (-(p**2 + 2 * p * gap) / (2 * large)).exp()
            wraps.append(2 * (-gap**2 / (2 * large)).exp() / (1 - ratio))
        # area / sqrt(det covariance) = 1/(sigma_x sigma_y).
        peak = 1 / (2 * arb.pi() * arb(float(sigma[0])) * arb(float(sigma[1])))
        wrapped = peak * (wraps[0] + wraps[1] + wraps[0] * wraps[1])
        return {"fourier_template_error_bound": upper_float(spectral),
                "periodic_copy_template_error_bound": upper_float(wrapped),
                "template_error_bound": upper_float(spectral + wrapped)}


class CommonDetector:
    """One common angular grid for finite, individually calibrated exposures.

    Coordinates and matrices use arcseconds. `matrix[j]` maps a detector pixel
    displacement (x,y) to the common angular plane; `offset[j]` is its stamp
    center. `source_half_width` declares the entire allowed source square.
    """

    def __init__(self, variances, sigma_pixels, matrices, *, offsets=None,
                 masks=None, flux_scales=1., profile_background=True,
                 source_half_width=4., spectral_sigma=8., max_bytes=1_000_000_000):
        v = np.asarray(variances, float)
        if v.ndim != 3 or v.shape[1] != v.shape[2] or v.shape[1] % 2 != 1 or min(v.shape) < 1:
            raise ValueError("Positive odd-square (exposure,y,x) variance arrays required")
        self.frames, self.n = v.shape[:2]
        self.shape_pixels = v.shape
        self.sigma = np.broadcast_to(np.asarray(sigma_pixels, float), (self.frames, 2)).copy()
        self.matrices = np.broadcast_to(np.asarray(matrices, float), (self.frames, 2, 2)).copy()
        self.offsets = (np.zeros((self.frames, 2)) if offsets is None
                        else np.broadcast_to(np.asarray(offsets, float), (self.frames, 2)).copy())
        self.scales = np.broadcast_to(np.asarray(flux_scales, float), (self.frames,)).copy()
        if (not all(np.all(np.isfinite(x)) for x in
                    [v, self.sigma, self.matrices, self.offsets, self.scales])
                or np.any(v <= 0) or np.any(self.sigma <= 0) or np.any(self.scales <= 0)
                or not isinstance(profile_background, bool)
                or not np.isfinite(source_half_width) or source_half_width <= 0
                or not np.isfinite(spectral_sigma) or not 3 <= spectral_sigma <= 12):
            raise ValueError("Finite geometry/data, positive variances/scales and valid source box required")
        self.mask = np.zeros(v.shape, bool) if masks is None else np.asarray(masks) != 0
        if self.mask.shape != v.shape:
            raise ValueError("Mask shape mismatch")
        self.variances = v.copy()
        self.weight = np.where(self.mask, 0., 1 / v)
        self.half, self.profile = float(source_half_width), profile_background
        bounds = [_geometry_bounds(b, s, o, self.n, self.half)
                  for b, s, o in zip(self.matrices, self.sigma, self.offsets)]
        with ctx.workprec(128):
            sufficient = max(upper_float(max(extent, key=upper_float) + 10 * large.sqrt())
                             for _, large, _, extent in bounds)
            self.period = float(np.ceil(sufficient))
            self.cutoff = max(int(np.ceil(upper_float(
                arb(self.period) * arb(float(spectral_sigma)) / (2 * arb.pi() * small.sqrt()))))
                for small, _, _, _ in bounds)
        self.side = 4 * self.cutoff + 1
        required = self.frames * self.side**2 * 16 * 2
        if required > max_bytes:
            raise ValueError(f"Two Fourier-field arrays exceed allocation budget: {required}")
        self.spacing = 2 * np.pi / self.period
        self.template_reports = [template_bound(b, s, o, self.n, self.half, self.period, self.cutoff)
                                 for b, s, o in zip(self.matrices, self.sigma, self.offsets)]
        self._tail_reports = {self.cutoff: self.template_reports}
        axis = np.arange(self.n) - (self.n - 1) / 2
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        self.pixels = np.column_stack((xx.ravel(), yy.ravel()))
        small_axis = np.arange(-self.cutoff, self.cutoff + 1) * self.spacing
        kx, ky = np.meshgrid(small_axis, small_axis, indexing="ij")
        self.small_k = np.stack((kx, ky), axis=-1)
        self.psf = []
        for b, sigma in zip(self.matrices, self.sigma):
            pixel_k = self.small_k @ b
            transfer = np.exp(-.5 * np.sum((pixel_k * sigma)**2, axis=-1))
            transfer *= np.prod(np.sinc(pixel_k / (2 * np.pi)), axis=-1)
            self.psf.append(abs(np.linalg.det(b)) / self.period**2 * transfer)

    def numerator_bounds(self, images, cutoff):
        """Bound a smaller numerator grid without constructing discarded modes."""
        images = np.asarray(images, float)
        if (images.shape != self.shape_pixels or not np.all(np.isfinite(images))
                or isinstance(cutoff, bool) or not isinstance(cutoff, int)
                or not 1 <= cutoff <= self.cutoff):
            raise ValueError("Finite matching images and an admissible integer cutoff required")
        if cutoff not in self._tail_reports:
            self._tail_reports[cutoff] = [
                template_bound(b, s, o, self.n, self.half, self.period, cutoff)
                for b, s, o in zip(self.matrices, self.sigma, self.offsets)]
        errors = []
        for j, report in enumerate(self._tail_reports[cutoff]):
            w, d = self.weight[j], images[j].copy()
            if self.profile and np.sum(w) > 0:
                d -= np.sum(w * d) / np.sum(w)
            with ctx.workprec(128):
                errors.append(upper_float(arb(report["template_error_bound"]) *
                                          arb(l1_upper(w * d)) * arb(float(self.scales[j]))))
        return {"cutoff": cutoff, "template_reports": self._tail_reports[cutoff],
                "numerator_model_error_bound": sum_bounds(errors), "per_exposure_error_bounds": errors}

    def choose_numerator_cutoff(self, images, absolute_budget):
        if not np.isfinite(absolute_budget) or absolute_budget <= 0:
            raise ValueError("Positive finite numerator error budget required")
        if self.numerator_bounds(images, self.cutoff)["numerator_model_error_bound"] > absolute_budget:
            raise ValueError("Initial spectral cutoff cannot meet numerator error budget")
        low, high = 1, self.cutoff
        while low < high:
            mid = (low + high) // 2
            if self.numerator_bounds(images, mid)["numerator_model_error_bound"] <= absolute_budget:
                high = mid
            else:
                low = mid + 1
        return low, self.numerator_bounds(images, low)

    def prepare_numerator(self, images, cutoff):
        """Compute only the modes a certified numerator actually needs."""
        import finufft
        report = self.numerator_bounds(images, cutoff)
        side = 2 * cutoff + 1
        result = np.empty((self.frames, side, side), complex)
        crop = slice(self.cutoff - cutoff, self.cutoff + cutoff + 1)
        for j in range(self.frames):
            w, d = self.weight[j], np.asarray(images[j], float).copy()
            if self.profile and np.sum(w) > 0:
                d -= np.sum(w * d) / np.sum(w)
            points = (self.pixels @ self.matrices[j].T + self.offsets[j]) * self.spacing
            modes = finufft.nufft2d1(np.ascontiguousarray(points[:, 0]),
                                     np.ascontiguousarray(points[:, 1]),
                                     np.ascontiguousarray((w * d).ravel().astype(complex)),
                                     n_modes=(side, side), eps=1e-13, isign=-1, nthreads=1)
            result[j] = self.psf[j][crop, crop] * modes * self.scales[j]
        return result, report

    def prepare(self, images, *, include_information=True):
        """Return numerator/information fields and exact-model error budgets."""
        import finufft
        images = np.asarray(images, float)
        if images.shape != self.shape_pixels or not np.all(np.isfinite(images)):
            raise ValueError("Finite images matching the detector plan required")
        if not isinstance(include_information, bool):
            raise ValueError("include_information must be boolean")
        numerator = np.zeros((self.frames, self.side, self.side), complex)
        information = np.zeros_like(numerator) if include_information else None
        reports = []
        c = self.cutoff
        for j in range(self.frames):
            w, d = self.weight[j], images[j].copy()
            total = float(np.sum(w))
            if self.profile and total > 0:
                d -= np.sum(w * d) / total
            points = (self.pixels @ self.matrices[j].T + self.offsets[j]) * self.spacing
            rows = [(w * d).ravel(), w.ravel()] if include_information else [(w * d).ravel()]
            values = np.ascontiguousarray(np.stack(rows).astype(complex))
            modes = finufft.nufft2d1(np.ascontiguousarray(points[:, 0]),
                                     np.ascontiguousarray(points[:, 1]), values,
                                     n_modes=(self.side, self.side), eps=1e-13, isign=-1, nthreads=1)
            core = (slice(c, 3 * c + 1),) * 2
            psf = self.psf[j]
            numerator[j][core] = psf * modes[0][core] * self.scales[j]
            if include_information:
                information[j] = fftconvolve(psf, psf, mode="full") * modes[1] * self.scales[j]**2
                if self.profile and total > 0:
                    overlap = psf * modes[1][core]
                    information[j] -= fftconvolve(overlap, overlap, mode="full") / total * self.scales[j]**2
            with ctx.workprec(128):
                delta = arb(self.template_reports[j]["template_error_bound"])
                nerr = delta * arb(l1_upper(w * d)) * arb(float(self.scales[j]))
                ierr = delta * (2 + delta) * arb(l1_upper(w)) * arb(float(self.scales[j]))**2
                if self.profile:
                    ierr *= 2
                reports.append({**self.template_reports[j],
                                "numerator_model_error_bound": upper_float(nerr),
                                "information_model_error_bound": upper_float(ierr)})
        report = {"period_arcsec": self.period, "template_cutoff": self.cutoff,
                  "field_side": self.side, "fourier_spacing_inverse_arcsec": self.spacing,
                  "source_half_width_arcsec": self.half,
                  "allocated_field_bytes": numerator.nbytes + (information.nbytes if include_information else 0),
                  "information_computed": include_information,
                  "exposures": reports, "periodic_camera_assumed": False,
                  "image_resampling_performed": False,
                  "wcs_model": "declared affine maps into one tangent plane",
                  "floating_point_and_wcs_errors_certified": False}
        return numerator, information, report

    def pixel_positions(self, angular_positions):
        p = np.asarray(angular_positions, float)
        if p.ndim != 3 or p.shape[1:] != (self.frames, 2) or not np.all(np.isfinite(p)):
            raise ValueError("Finite (query,exposure,2) angular positions required")
        if np.any(abs(p) > self.half):
            raise ValueError("Source positions leave the declared detector approximation box")
        return np.einsum("fij,qfj->qfi", np.linalg.inv(self.matrices), p - self.offsets)

    def direct(self, images, angular_positions):
        return direct_pixel_statistics(images, self.variances, self.sigma,
                                       self.pixel_positions(angular_positions), masks=self.mask,
                                       profile_background=self.profile, flux_scales=self.scales)


def trim_fields(numerator, information, absolute_budgets):
    """Remove outer mode squares only when their coefficient l1 fits a budget."""
    n, i = np.asarray(numerator), np.asarray(information)
    if n.shape != i.shape or n.ndim != 3 or n.shape[1] != n.shape[2] or n.shape[1] % 2 != 1:
        raise ValueError("Compatible odd square coefficient fields required")
    if len(absolute_budgets) != 2 or any(not np.isfinite(x) or x <= 0 for x in absolute_budgets):
        raise ValueError("Positive finite absolute field-error budgets required")
    half = n.shape[1] // 2
    def error_at(keep):
        outside = np.ones(n.shape[1:], bool)
        outside[half - keep:half + keep + 1, half - keep:half + keep + 1] = False
        return [l1_upper(a[:, outside]) for a in (n, i)]
    low, high = min(1, half), half  # Retain at least three modes for the field API.
    while low < high:
        mid = (low + high) // 2
        if all(e <= b for e, b in zip(error_at(mid), absolute_budgets)):
            high = mid
        else:
            low = mid + 1
    best, bounds = low, error_at(low)
    selection = (slice(None), slice(half - best, half + best + 1),
                 slice(half - best, half + best + 1))
    return (np.ascontiguousarray(n[selection]), np.ascontiguousarray(i[selection]),
            {"original_side": n.shape[1], "retained_side": 2 * best + 1,
             "discarded_numerator_l1_bound": bounds[0], "discarded_information_l1_bound": bounds[1]})
