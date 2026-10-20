"""Executable cascade with a whole-domain inverse certificate.

This is a transfer building block, not a complete archive transform. Its
exact-map domain and (optionally) holomorphy are certified; numerical orbit
propagation and floating-point matrix application are not.
"""

from time import perf_counter

import numpy as np
from flint import acb, arb

from research.detection_limit.interval_orbits import upper_float
from .cascade_domain import _state_box, certify_domain
from .cascade_transfer import CascadeTransfer
from .reception_flow import enclose
from .segment_field import TensorGrid
from .tensor_prolongation import TensorProlongation


class CertifiedCascadeTransfer:
    def __init__(self, observer, parent, start, stop, *, child_order=None,
                 child_cells=None, certificate_options=None, refine=True,
                 max_entries=20_000_000, max_work_nodes=1_000_000,
                 child_grid_factory=None, **kwargs):
        begin = perf_counter()
        if parent.axes != tuple(range(6)):
            raise ValueError("A full six-coordinate parent grid is required")
        proof = certify_domain(observer, parent.center, parent.widths, start, stop,
                               **(certificate_options or {}))
        if child_grid_factory is None:
            child = TensorGrid(proof["child_center"], proof["child_widths"],
                               parent.cells if child_cells is None else child_cells,
                               parent.orders if child_order is None else child_order)
        else:
            if child_order is not None or child_cells is not None:
                raise ValueError("Specify a child grid factory or explicit grid orders/cells")
            child = child_grid_factory(np.array(proof["child_center"]), np.array(proof["child_widths"]))
            if (child.axes != tuple(range(6))
                    or not np.array_equal(child.center, proof["child_center"])
                    or not np.array_equal(child.widths, proof["child_widths"])):
                raise ValueError("Child factory must retain the full certified box")
        if max(parent.size, child.size) > max_work_nodes:
            raise MemoryError("Transfer grid budget exceeded before allocation")
        work = child
        self.prolongation = None
        if refine:
            # Count-balanced dyadic cell grids are nested. Refuse incompatible
            # partitions instead of silently multiplying their cell counts.
            work = TensorGrid(child.center, child.widths,
                              tuple(max(a, b) for a, b in zip(parent.cells, child.cells)),
                              tuple(max(a, b) for a, b in zip(parent.orders, child.orders)))
            if work.cells != child.cells or work.orders != child.orders:
                self.prolongation = TensorProlongation(
                    child, work, max_entries=max_entries, max_work_nodes=max_work_nodes)
        chart, target = observer.chart(start), observer.chart(stop)
        def rebase(q):
            if np.any(abs((q-parent.center)/parent.widths)
                      > np.array(proof["normalized_enclosing_radii"])):
                raise ValueError("Numerical geometry left the certified enclosing box")
            return chart.rebase(q, stop-start, target)
        used = 0 if self.prolongation is None else self.prolongation.entries
        self.remap = CascadeTransfer(parent, work, rebase,
                                     max_entries=max_entries-used, **kwargs)
        self.parent, self.child, self.work_grid = parent, child, work
        self.shape = (parent.size, child.size)
        self.entries = used+self.remap.entries
        self.array_bytes = self.remap.array_bytes+(
            0 if self.prolongation is None else self.prolongation.array_bytes)
        self.certificate = proof
        self.report = dict(self.remap.report)
        self.report.update({
            "stored_entries": self.entries, "array_bytes": self.array_bytes,
            "separable_refinement": None if self.prolongation is None else self.prolongation.report,
            "intermediate_domains_certified": True,
            "complex_intermediate_inverses_certified": proof["complex_inverse_certified"],
            "certificate": proof,
            "total_construction_seconds": perf_counter()-begin,
            "floating_point_evaluation_certified": False,
            "full_objective_complete": False,
        })

    def __matmul__(self, values):
        if self.prolongation is not None:
            values = self.prolongation @ values
        return self.remap @ values

    def adjoint(self, values):
        result = self.remap.adjoint(values)
        return result if self.prolongation is None else self.prolongation.adjoint(result)

    def conj(self):
        return self

    @property
    def T(self):
        class Transpose:
            def __init__(self, parent):
                self.parent = parent

            def __matmul__(self, values):
                return self.parent.adjoint(values)
        return Transpose(self)


def fourier_modulus_bound(proof, observer, frequency_bounds, *, max_step=.03125,
                          max_range_pieces=16):
    """Bound child-angle Fourier fields on all complex hybrid inverse domains.

    The bound is per unit source-coefficient l1 norm. It covers a field of the
    form sum d_k exp(i*k.theta_child), not arbitrary detector/WCS fields.
    """
    if not proof["complex_inverse_certified"]:
        raise ValueError("A complex inverse-domain certificate is required")
    frequency_bounds = np.asarray(frequency_bounds, float)
    if (frequency_bounds.shape != (2,) or not np.all(np.isfinite(frequency_bounds))
            or np.any(frequency_bounds < 0)):
        raise ValueError("Two finite nonnegative angular frequency bounds required")
    initial = _state_box(proof["parent_center"], proof["parent_widths"],
                         proof["complex_normalized_enclosing_radii"],
                         proof["complex_normalized_imaginary_radii"])
    pieces, attempts = 1, 0
    begin = perf_counter()
    while True:
        exponents = []
        try:
            for j in range(pieces):
                values = initial.copy()
                if pieces > 1:
                    real = initial[0].real
                    values[0] = acb(real.mid()+real.rad()*arb(-1+(2*j+1)/pieces, 1/pieces),
                                    initial[0].imag)
                attempts += 1
                endpoint, _, _ = enclose(values, proof["start"], proof["stop"], observer,
                                         max_step=max_step)
                exponents.append(sum(arb(float(k))*abs(endpoint[a].imag)
                                     for k, a in zip(frequency_bounds, (3, 4))).upper())
            break
        except ValueError:
            if pieces*2 > max_range_pieces:
                raise
            pieces *= 2
    return {
        "log_modulus_per_source_l1": upper_float(max(exponents)),
        "flow_calls_including_refusals": attempts, "range_pieces": pieces,
        "seconds": perf_counter()-begin, "exact_arithmetic_uniform_bound": True,
    }


def interpolation_error(orders, ellipse, log_modulus, permutation=(3, 4, 5, 2, 1, 0)):
    """Uniform exact-arithmetic error of the six-stage transfer at parent nodes.

    Child nodal values must be exact samples of the analytic field. Input error
    is separately amplified by the reported product of Lebesgue constants.
    This does not include interpolation of the final parent field at queries.
    """
    orders = tuple(orders)
    if (len(orders) != 6 or any(isinstance(p, bool) or not isinstance(p, (int, np.integer))
                               or p < 1 for p in orders)
            or sorted(permutation) != list(range(6))
            or not np.isfinite(ellipse) or ellipse <= 1
            or not np.isfinite(log_modulus)):
        raise ValueError("Six positive integer orders, valid permutation and finite analytic bound required")
    r = arb(float(ellipse))
    modulus = arb(float(log_modulus)).exp()
    error, amplification = arb(0), arb(1)
    for axis in permutation:
        p = int(orders[axis])
        lebesgue = arb(1) if p <= 2 else arb(2*p-2).sqrt()
        local = (1+lebesgue)*2*modulus*r**(-p)/(1-1/r)
        error = lebesgue*error+local
        amplification *= lebesgue
    return {
        "log_uniform_error_per_source_l1": upper_float(error.log()),
        "input_error_amplification": upper_float(amplification),
        "exact_arithmetic_only": True,
        "floating_point_evaluation_certified": False,
    }


def sufficient_uniform_order(ellipse, log_modulus, tolerance=1e-8, *, max_order=100000):
    """Choose a sufficient order without constructing an unaffordable grid."""
    if not np.isfinite(tolerance) or tolerance <= 0 or max_order < 1:
        raise ValueError("Positive finite tolerance and order budget required")
    log_target = arb(float(tolerance)).log()

    def passes(p):
        bound = interpolation_error((p,)*6, ellipse, log_modulus)
        return arb(bound["log_uniform_error_per_source_l1"]) <= log_target
    low, high = 0, 1
    while not passes(high):
        low, high = high, min(2*high, max_order)
        if high == low:
            raise ValueError("Sufficient interpolation order exceeds the explicit budget")
    while high-low > 1:
        middle = (low+high)//2
        if passes(middle):
            high = middle
        else:
            low = middle
    bound = interpolation_error((high,)*6, ellipse, log_modulus)
    return {
        "order": high, "tolerance": tolerance, **bound,
        "uniform_single_cell_grid_nodes": high**6,
        "cascade_weights_on_equal_uniform_grids": 6*high**7,
        "grid_constructed": False,
    }


def cadence_count(times, bandwidth, *, prefactor=1.):
    """Check the conditional anisotropic node-count inequality at any cadence.

    G(h)=C B^2 (1+B h)^2 (1+B h^2)^2 is a dimensionless resolution model,
    not an established resolution guarantee for an arbitrary physical archive.
    """
    times = np.asarray(times, float)
    if (times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))
            or np.any(np.diff(times) < 0) or not np.isfinite(bandwidth)
            or bandwidth <= 0 or not np.isfinite(prefactor) or prefactor <= 0):
        raise ValueError("Finite sorted times, positive bandwidth and prefactor required")
    def size(span):
        return prefactor*bandwidth**2*(1+bandwidth*span)**2*(1+bandwidth*span**2)**2
    queue, total, depth = [(0, len(times))], 0., 0
    while queue:
        following = []
        for first, last in queue:
            total += size(float(times[last-1]-times[first]))
            if last-first > 1:
                middle = (first+last)//2
                following.extend([(first, middle), (middle, last)])
        queue = following
        depth += 1
    leaf_cost = size(0.)
    root_cost = size(float(times[-1]-times[0]))
    bound = (2*len(times)-1)*leaf_cost+depth*(root_cost-leaf_cost)
    return {"exposures": len(times), "levels": depth, "modeled_total": total,
            "modeled_upper_bound": bound, "root_count_model": root_cost,
            "resolution_law_is_assumed": True,
            "full_objective_complete": False}
