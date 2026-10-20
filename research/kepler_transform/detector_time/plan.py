"""Time compression at a detector's actual bandwidth, with tracked pointings.

Only the original analytically certified circular-observer domains are used.
Real ephemerides must not be silently substituted into this certificate.
"""

from time import perf_counter
from math import prod

import numpy as np
from flint import arb, ctx

from research.detection_limit.interval_orbits import upper_float
from ..certify_global import partitioned_gradients
from ..chart_experiments import setup
from ..factorization import ARCSEC
from ..global_kepler import ContinuousKeplerPhase
from ..source_butterfly import cardinal
from ..temporal.transform import TimePlan, FourierEvaluator
from .fields import l1_upper


def domain_source_box(case):
    chart, center, widths, geometry, _ = setup(case)
    span = float(max(abs(geometry.times)))
    with ctx.workprec(128):
        gradients, proof = partitioned_gradients(chart, center, widths, 0., span, 1.,
                                                 partition_depth=6)
        bounds = [upper_float(sum((abs(v.real) for v in axis), arb(0)) / arb(ARCSEC))
                  for axis in gradients]
    return {"case": case, "relative_reference_position_bound_arcsec": bounds,
            "time_half_span_days": span, "root_certificate": proof,
            "pointing_model": "stamp origin tracks the mathematical center orbit"}


class DetectorTimePlan(TimePlan):
    """General Fourier spacing, with each exposure centered on the reference orbit."""

    def __init__(self, case, times, shape, spacing_inverse_arcsec, *,
                 tolerance=1e-9, rho=2., max_order=2048, compression_backend="dense"):
        started = perf_counter()
        if (case not in ("near", "distant", "stress") or len(shape) != 2
                or any(not isinstance(x, int) or isinstance(x, bool) or x < 3 or x % 2 == 0
                       for x in shape)):
            raise ValueError("Known domain and odd mode dimensions >=3 required")
        if (not np.isfinite(spacing_inverse_arcsec) or spacing_inverse_arcsec <= 0
                or not np.isfinite(tolerance) or not 0 < tolerance < 1
                or not np.isfinite(rho) or rho <= 1
                or not isinstance(max_order, int) or not 2 <= max_order <= 4096
                or compression_backend not in ("dense", "nufft")):
            raise ValueError("Valid spacing, tolerance, ellipse, order budget and backend required")
        self.shape, self.times = tuple(shape), np.array(times, float, copy=True)
        self.chart, self.center, self.widths, geometry, _ = setup(case)
        self.span = float(max(abs(geometry.times)))
        if (self.times.ndim != 1 or not len(self.times) or not np.all(np.isfinite(self.times))
                or np.any(abs(self.times) > self.span)):
            raise ValueError("Times must lie in the certified interval")
        self.spacing = np.array([spacing_inverse_arcsec / ARCSEC] * 2)
        axes = [np.arange(-(n // 2), n // 2 + 1) for n in shape]
        xx, yy = np.meshgrid(*axes, indexing="ij")
        self.frequencies = np.column_stack((xx.ravel(), yy.ravel())) * self.spacing
        bandwidth = float(np.max(abs(self.frequencies)))
        with ctx.workprec(128):
            gradients, root = partitioned_gradients(self.chart, self.center, self.widths,
                                                    0., self.span, rho, partition_depth=6)
            d = sum((abs(v.imag) for ax in gradients for v in ax), arb(0)) * arb(bandwidth)
            for p in range(2, max_order + 1):
                bound = 2 * (1 + arb(2 * p - 1).sqrt()) * d.exp() * arb(rho)**(-(p - 1)) / (arb(rho) - 1)
                if bound < arb(tolerance):
                    break
            else:
                raise ValueError("Required detector-bandwidth order exceeds budget")
            self.order = p
            self.proof = {"case": case, "chart": self.chart.to_dict(), "center": self.center.tolist(),
                          "widths": self.widths.tolist(), "time_half_span_days": self.span,
                          "frequency_bound_radian_inverse": bandwidth, "ellipse_parameter": rho,
                          "time_log_modulus_bound": upper_float(d), "order": p,
                          "exact_arithmetic_kernel_error_bound": upper_float(bound),
                          "requested_tolerance": tolerance, "root_certificate": root,
                          "floating_point_error_certified": False}
        self.phase = ContinuousKeplerPhase(self.chart, self.center, self.widths,
                                           time_half_span=self.span, frequency_scale=bandwidth)
        alpha = (np.arange(p) + .5) * np.pi / p
        nodes, weights = np.cos(alpha), (-1.)**np.arange(p) * np.sin(alpha)
        self.node_times = nodes * self.span
        self.compression_backend, self.numerical_eps = compression_backend, 1e-12
        if compression_backend == "dense":
            self.interpolation = cardinal(self.times / self.span, nodes, weights)
        else:
            import finufft
            theta = np.arccos(self.times / self.span)
            self.reflected_times = np.r_[-theta, theta]
            self.moment_plan = finufft.Plan(1, (2 * p - 1,), n_trans=prod(shape),
                                            eps=self.numerical_eps, isign=1, nthreads=1)
            self.moment_plan.setpts(self.reflected_times)
        self.center_input = self.positions(np.zeros((1, 6)), self.times)[0]
        self.center_nodes = self.positions(np.zeros((1, 6)), self.node_times)[0]
        # Each field's angular coordinate already subtracts the center orbit.
        self.modulation = np.ones((len(self.times), prod(shape)), complex)
        self.proof["construction_seconds"] = perf_counter() - started

    def direct_plan(self, normalized_queries, **kwargs):
        positions = self.positions(normalized_queries, self.times) - self.center_input
        return FourierEvaluator(positions, self.shape, self.spacing, **kwargs)

    def score_error_bound(self, coefficients):
        with ctx.workprec(128):
            return upper_float(arb(self.proof["exact_arithmetic_kernel_error_bound"]) *
                               arb(l1_upper(coefficients)))


def statistic_intervals(numerator, information, numerator_error, information_error):
    """Propagate absolute field errors into S/N and profiled flux intervals.

    No significance/false-alarm calibration is inferred. Refuse unresolved
    nonpositive information rather than clipping it into a detection.
    """
    n, i = np.asarray(numerator, float), np.asarray(information, float)
    if (n.shape != i.shape or not np.all(np.isfinite(n + i))
            or not all(np.isfinite(e) and e >= 0 for e in [numerator_error, information_error])):
        raise ValueError("Finite statistics and nonnegative finite errors required")
    lo, hi = i - information_error, i + information_error
    resolved = lo > 0
    snr = np.full((*n.shape, 2), [-np.inf, np.inf])
    flux = snr.copy()
    if np.any(resolved):
        a, b = n[resolved] - numerator_error, n[resolved] + numerator_error
        corners = np.stack((a / np.sqrt(lo[resolved]), a / np.sqrt(hi[resolved]),
                            b / np.sqrt(lo[resolved]), b / np.sqrt(hi[resolved])))
        snr[resolved] = np.stack((corners.min(axis=0), corners.max(axis=0)), axis=-1)
        corners = np.stack((a / lo[resolved], a / hi[resolved], b / lo[resolved], b / hi[resolved]))
        flux[resolved] = np.stack((corners.min(axis=0), corners.max(axis=0)), axis=-1)
    return {"positive_information_resolved": resolved, "snr_interval": snr,
            "flux_interval": flux, "interval_endpoint_roundoff_certified": False}
