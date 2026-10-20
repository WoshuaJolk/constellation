"""Compress source time, retain Fourier modes, and never grid orbit space.

The exact-arithmetic interpolation guarantee is separate from FFT, orbit-solver,
and NUFFT roundoff. Only the existing circular-observer benchmark domains are
supported. Inputs are periodic, common-tangent-plane trigonometric fields.
"""

from math import prod
from time import perf_counter

import numpy as np
from flint import arb, ctx

from research.detection_limit.interval_orbits import upper_float
from ..certify_global import partitioned_gradients
from ..chart_experiments import setup
from ..compiled_reference import projected_chart
from ..factorization import ARCSEC
from ..global_kepler import ContinuousKeplerPhase
from ..source_butterfly import cardinal, node_rule


def certificate(case, tolerance=1e-8, rho=2., max_order=512):
    """Fresh interval proof over the entire original box and time ellipse."""
    if case not in ("near", "distant", "stress"):
        raise ValueError("Unknown certified domain")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Tolerance must lie in (0,1)")
    if not np.isfinite(rho) or rho <= 1:
        raise ValueError("Ellipse parameter must exceed one")
    if isinstance(max_order, bool) or not isinstance(max_order, int) or not 2 <= max_order <= 512:
        raise ValueError("Maximum order must be an integer in 2..512")
    chart, center, widths, geometry, _ = setup(case)
    span = float(max(abs(geometry.times)))
    started = perf_counter()
    with ctx.workprec(128):
        gradients, root_record = partitioned_gradients(
            chart, center, widths, 0., span, rho, partition_depth=6)
        # Integrate normalized orbit derivatives along the real segment from
        # the box center to q. All six normalized displacements are <= 1.
        log_modulus = sum((abs(v.imag) for axis in gradients for v in axis),
                          arb(0)) * arb(float(1 / ARCSEC))
        for order in range(2, max_order + 1):
            bound = (2 * (1 + arb(2 * order - 1).sqrt()) * log_modulus.exp()
                     * arb(rho)**(-(order - 1)) / (arb(rho) - 1))
            if bound <= arb(tolerance):
                break
        else:
            raise ValueError("Certified time order exceeds budget; domain not reduced")
        result = {
            "case": case, "chart": chart.to_dict(),
            "center": center.tolist(), "widths": widths.tolist(),
            "time_half_span_days": span, "frequency_bound": 1 / ARCSEC,
            "ellipse_parameter": rho, "time_log_modulus_bound": upper_float(log_modulus),
            "order": order, "tolerance": tolerance,
            "exact_arithmetic_kernel_error_bound": upper_float(bound),
            "root_certificate": root_record,
            "floating_point_error_certified": False,
        }
    result["construction_seconds"] = perf_counter() - started
    return result


def image_coefficients(images):
    """Exact input convention: periodic trigonometric interpolation of samples."""
    images = np.asarray(images)
    if images.ndim != 3 or min(images.shape) < 1 or not np.all(np.isfinite(images)):
        raise ValueError("A finite (exposure, x, y) image stack is required")
    return np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1)) / prod(images.shape[-2:])


def _fields(data, visits, shape):
    data = np.asarray(data, complex)
    if data.shape != (visits, *shape) or not np.all(np.isfinite(data)):
        raise ValueError("Finite coefficient fields with the plan's shape required")
    return np.ascontiguousarray(data)


class FourierEvaluator:
    """Apply fields at fixed sky coordinates; also used by direct baselines."""

    def __init__(self, positions, shape, spacing, *, backend="nufft",
                 eps=1e-12, max_cache_bytes=600_000_000):
        positions = np.asarray(positions, float)
        if (positions.ndim != 3 or positions.shape[-1] != 2 or min(positions.shape) < 1
                or not np.all(np.isfinite(positions))):
            raise ValueError("Finite (query, time, 2) sky coordinates required")
        if backend not in ("nufft", "dense", "stream"):
            raise ValueError("Unknown Fourier evaluation backend")
        if not np.isfinite(eps) or not 0 < eps < 1:
            raise ValueError("NUFFT tolerance must lie in (0,1)")
        self.shape, self.spacing = tuple(shape), np.asarray(spacing, float)
        if (len(self.shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2
                                      for n in self.shape)
                or self.spacing.shape != (2,) or not np.all(np.isfinite(self.spacing))
                or np.any(self.spacing <= 0)):
            raise ValueError("Two integer mode dimensions and positive spacings required")
        self.queries, self.visits = positions.shape[:2]
        self.backend = backend
        modes = [np.arange(-(n // 2), (n - 1) // 2 + 1) for n in self.shape]
        xx, yy = np.meshgrid(*modes, indexing="ij")
        self.frequencies = np.column_stack((xx.ravel(), yy.ravel())) * self.spacing
        self.persistent_array_bytes = 0
        if backend == "dense":
            size = self.queries * self.visits * prod(self.shape) * 16
            if size > max_cache_bytes:
                raise ValueError(f"Dense response cache exceeds byte budget: {size}")
            self.kernel = np.empty((self.queries, self.visits * prod(self.shape)), complex)
            for visit in range(self.visits):
                self.kernel[:, visit * prod(self.shape):(visit + 1) * prod(self.shape)] = (
                    np.exp(1j * positions[:, visit] @ self.frequencies.T))
            self.persistent_array_bytes = self.kernel.nbytes
        elif backend == "nufft":
            import finufft
            self.plans = []
            # Retain point arrays: the FINUFFT guru interface owns pointers.
            self.points = []
            for visit in range(self.visits):
                xy = positions[:, visit] * self.spacing
                x, y = [np.ascontiguousarray(xy[:, axis]) for axis in range(2)]
                plan = finufft.Plan(2, self.shape, eps=eps, isign=1, nthreads=1)
                plan.setpts(x, y)
                self.plans.append(plan)
                self.points.append((x, y))
                self.persistent_array_bytes += x.nbytes + y.nbytes
            # Native library workspaces are not counted in the above NumPy total.
        else:
            self.positions = positions.copy()
            self.persistent_array_bytes = self.positions.nbytes

    def apply(self, coefficients):
        data = _fields(coefficients, self.visits, self.shape)
        if self.backend == "dense":
            return self.kernel @ data.ravel()
        result = np.zeros(self.queries, complex)
        for visit in range(self.visits):
            if self.backend == "nufft":
                result += self.plans[visit].execute(data[visit])
            else:
                result += np.exp(1j * self.positions[:, visit] @ self.frequencies.T) @ data[visit].ravel()
        return result

    def adjoint(self, values):
        """Dense/stream algebraic adjoint, for correctness and inverse research."""
        values = np.asarray(values, complex)
        if values.shape != (self.queries,) or not np.all(np.isfinite(values)):
            raise ValueError("Finite vector of query values required")
        if self.backend == "dense":
            return (self.kernel.conj().T @ values).reshape(self.visits, *self.shape)
        if self.backend != "stream":
            raise ValueError("Use the dense or stream backend for the algebraic adjoint")
        return np.array([np.exp(-1j * self.positions[:, v] @ self.frequencies.T).T @ values
                         for v in range(self.visits)]).reshape(self.visits, *self.shape)


class TimePlan:
    """A certified continuous orbit field represented by p Fourier images.

    Setup and transform need no query orbits. Query compilation is optional;
    its complete cost must be charged when query points change.
    """

    def __init__(self, case, times, shape=(8, 8), *, tolerance=1e-8, rho=2.,
                 max_order=512, compression_backend="dense", numerical_eps=1e-12):
        if (len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2
                                   for n in shape)):
            raise ValueError("Two integer mode dimensions >=2 required")
        self.shape = tuple(shape)
        if compression_backend not in ("dense", "nufft"):
            raise ValueError("Compression backend must be dense or nufft")
        if not np.isfinite(numerical_eps) or not 0 < numerical_eps < 1:
            raise ValueError("Numerical epsilon must lie in (0,1)")
        self.compression_backend, self.numerical_eps = compression_backend, numerical_eps
        self.times = np.array(times, dtype=float, copy=True)
        if self.times.ndim != 1 or not len(self.times) or not np.all(np.isfinite(self.times)):
            raise ValueError("Finite nonempty exposure times required")
        self.proof = certificate(case, tolerance, rho, max_order)
        self.chart, self.center, self.widths, _, _ = setup(case)
        self.span, self.order = self.proof["time_half_span_days"], self.proof["order"]
        if np.any(abs(self.times) > self.span):
            raise ValueError("Exposure times leave the certified interval")
        self.spacing = np.nextafter(
            np.array([1 / ((n // 2) * ARCSEC) for n in self.shape]), 0.)
        modes = [np.arange(-(n // 2), (n - 1) // 2 + 1) for n in self.shape]
        xx, yy = np.meshgrid(*modes, indexing="ij")
        self.frequencies = np.column_stack((xx.ravel(), yy.ravel())) * self.spacing
        if np.any(abs(self.frequencies) > self.proof["frequency_bound"]):
            raise ValueError("Rounded frequency grid exceeds the certified bandwidth")
        self.phase = ContinuousKeplerPhase(
            self.chart, self.center, self.widths, time_half_span=self.span,
            frequency_scale=1 / ARCSEC)
        nodes, weights = node_rule(self.order)
        self.node_times = nodes * self.span
        if compression_backend == "dense":
            self.interpolation = cardinal(self.times / self.span, nodes, weights)
        else:
            import finufft
            theta = np.arccos(self.times / self.span)
            self.reflected_times = np.r_[-theta, theta]
            self.moment_plan = finufft.Plan(
                1, (2 * self.order - 1,), n_trans=prod(self.shape),
                eps=numerical_eps, isign=1, nthreads=1)
            self.moment_plan.setpts(self.reflected_times)
        self.center_input = self.positions(np.zeros((1, 6)), self.times)[0]
        self.center_nodes = self.positions(np.zeros((1, 6)), self.node_times)[0]
        self.modulation = np.exp(1j * self.center_input @ self.frequencies.T)

    def positions(self, normalized_queries, times):
        q = np.asarray(normalized_queries, float)
        if (q.ndim != 2 or q.shape[1] != 6 or not len(q)
                or not np.all(np.isfinite(q)) or np.any(abs(q) > 1)):
            raise ValueError("Normalized orbit queries must lie in the certified six-dimensional box")
        return projected_chart(self.chart, self.center + q * self.widths,
                               self.phase.geometry(np.asarray(times, float)))

    def transform(self, coefficients):
        data = _fields(coefficients, len(self.times), self.shape)
        modulated = data.reshape(len(self.times), -1) * self.modulation
        if self.compression_backend == "dense":
            result = self.interpolation.T @ modulated
        else:
            from scipy.fft import dct
            reflected = np.ascontiguousarray(np.tile(modulated.T, (1, 2)) / 2)
            moments = self.moment_plan.execute(reflected)[:, self.order - 1:]
            result = (dct(moments, type=3, axis=1) / self.order).T
        return np.ascontiguousarray(result.reshape(self.order, *self.shape))

    def adjoint_compress(self, values):
        values = _fields(values, self.order, self.shape)
        if self.compression_backend == "dense":
            result = self.interpolation @ values.reshape(self.order, -1)
        else:
            import finufft
            from scipy.fft import dct
            moments = dct(values.reshape(self.order, -1).T, type=2, axis=1) / self.order
            moments[:, 0] *= .5
            modes = np.zeros((prod(self.shape), 2 * self.order - 1), complex)
            modes[:, self.order - 1:] = moments
            reflected = finufft.nufft1d2(self.reflected_times, modes, eps=self.numerical_eps,
                                        isign=-1, nthreads=1)
            result = reflected.reshape(prod(self.shape), 2, len(self.times)).sum(axis=1).T / 2
        return (result * self.modulation.conj()).reshape(len(self.times), *self.shape)

    def query_plan(self, normalized_queries, **kwargs):
        positions = self.positions(normalized_queries, self.node_times) - self.center_nodes
        return FourierEvaluator(positions, self.shape, self.spacing, **kwargs)

    def direct_plan(self, normalized_queries, **kwargs):
        return FourierEvaluator(self.positions(normalized_queries, self.times),
                                self.shape, self.spacing, **kwargs)

    def evaluate(self, fields, normalized_queries, **kwargs):
        """Fresh arbitrary queries, including orbit solves and Fourier plan setup."""
        return self.query_plan(normalized_queries, **kwargs).apply(fields)

    def counts(self, queries):
        n, p, k, q = len(self.times), self.order, prod(self.shape), queries
        return {
            "exposures": n, "time_order": p, "fourier_modes": k, "queries": q,
            "input_values": n * k, "compressed_field_values": p * k,
            "compression_complex_multiply_adds": n * p * k,
            "compression_backend": self.compression_backend,
            "dense_compression_count_is_executed": self.compression_backend == "dense",
            "plan_center_orbit_pairs": n + p,
            "fresh_query_orbit_pairs": q * p,
            "fresh_direct_orbit_pairs": q * n,
            "dense_compressed_query_entries": q * p * k,
            "dense_direct_cache_entries": q * n * k,
            "interpolation_real_entries": n * p if self.compression_backend == "dense" else 0,
            "modulation_complex_entries": n * k,
            "orbit_grid_entries": 0,
        }
