"""Bottom-up continuous Kepler fields with certified transition covers.

This is a complete executable segment recurrence, not a claim that its
accuracy/cost meets the FKT objective. Parent construction never resums images.
The finite interpolation graph has an exact algebraic adjoint and a defined
Tikhonov inverse. Coverage certificates refer to exact geometric maps; numerical
rebasing and sparse matrix arithmetic still incur uncertified roundoff.
"""

from dataclasses import dataclass
from itertools import product
import math
from time import perf_counter

import numpy as np
from flint import arb
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, lsmr

from .reception_flow import box_balls, enclose, float_box, rhs
from .source_butterfly import cardinal


def lobatto_rule(order):
    if order == 1:
        return np.array([0.]), np.array([1.])
    nodes = np.cos(np.pi*np.arange(order)/(order-1))
    nodes[0], nodes[-1] = 1., -1.
    weights = (-1.)**np.arange(order)
    weights[[0, -1]] *= .5
    return nodes, weights


@dataclass
class TensorGrid:
    center: np.ndarray
    widths: np.ndarray
    cells: tuple
    order: int
    axes: tuple = (0, 1, 2, 3, 4, 5)

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.widths = np.asarray(self.widths, float)
        self.orders = ((self.order,)*len(self.axes) if np.isscalar(self.order)
                       else tuple(self.order))
        if (self.center.shape != (6,) or self.widths.shape != (6,)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.widths))
                or np.any(self.widths <= 0) or len(self.cells) != len(self.axes)
                or any(isinstance(n, bool) or int(n) != n or n < 1 for n in self.cells)
                or len(set(self.axes)) != len(self.axes) or any(i not in range(6) for i in self.axes)
                or len(self.orders) != len(self.axes)
                or any(isinstance(p, bool) or int(p) != p or not 1 <= p <= 64 for p in self.orders)):
            raise ValueError("Valid finite tensor grid, cells and orders [1,64] required")
        self.cells = tuple(int(n) for n in self.cells)
        self.orders = tuple(int(p) for p in self.orders)
        self.shape = tuple(n*p for n, p in zip(self.cells, self.orders))
        self.size = math.prod(self.shape)
        self.stencil_size = math.prod(self.orders)

    def points(self):
        coordinates = []
        for axis, count, order in zip(self.axes, self.cells, self.orders):
            nodes, _ = lobatto_rule(order)
            normalized = ((2*np.arange(count)[:, None]+1+nodes)/count-1).ravel()
            coordinates.append(self.center[axis]+self.widths[axis]*normalized)
        result = np.tile(self.center, (self.size, 1))
        result[:, self.axes] = np.stack(np.meshgrid(*coordinates, indexing="ij"), -1).reshape(-1, len(self.axes))
        return result

    def matrix(self, parameters, *, max_entries=20_000_000):
        q = np.asarray(parameters, float)
        if q.ndim != 2 or q.shape[1] != 6 or not np.all(np.isfinite(q)):
            raise ValueError("Finite (N,6) query parameters required")
        if len(q)*self.stencil_size > max_entries:
            raise MemoryError("Interpolation stencil budget exceeded before allocation")
        per_axis, cell_indices = [], []
        for axis, count, order in zip(self.axes, self.cells, self.orders):
            nodes, bary = lobatto_rule(order)
            z = (q[:, axis]-self.center[axis])/self.widths[axis]
            # A strict check: no silent clipping/extrapolation past the cover.
            if np.any(abs(z) > 1):
                raise ValueError("Query leaves the certified field box")
            local = (z+1)*count/2
            cell = np.minimum(np.floor(local).astype(int), count-1)
            per_axis.append(cardinal(2*(local-cell)-1, nodes, bary))
            cell_indices.append(cell)
        offsets = np.array(list(product(*(range(p) for p in self.orders))), dtype=int)
        weights = np.ones((len(q), len(offsets)))
        columns = np.zeros_like(weights, dtype=np.int64)
        stride = 1
        for j in range(len(self.axes)-1, -1, -1):
            weights *= per_axis[j][:, offsets[:, j]]
            columns += (cell_indices[j][:, None]*self.orders[j]+offsets[None, :, j])*stride
            stride *= self.shape[j]
        rows = np.arange(len(q)+1, dtype=np.int64)*len(offsets)
        return csr_matrix((weights.ravel(), columns.ravel(), rows), shape=(len(q), self.size))


@dataclass
class Segment:
    indices: np.ndarray
    epoch: float
    grid: TensorGrid
    children: tuple = ()
    transfers: tuple = ()
    leaf_kernel: np.ndarray = None
    proof_work: int = 0


@dataclass
class ExactExposure:
    """Stored exposure model evaluated at one parent's nodal states."""
    exposure: int
    kernel: np.ndarray

    @property
    def array_bytes(self):
        return self.kernel.nbytes


class SegmentTransform:
    """Data-independent graph mapping Fourier exposure fields to a full box.

    ``grid_policy(center,widths,half_span,is_leaf)`` returns counts of cells,
    six internally and two at an individual-exposure leaf. Fixed order is the
    number of Chebyshev-Lobatto nodes per cell and active coordinate. A callable
    order policy may return different orders for each coordinate and segment.

    Cadence is split by sample count and actual spans are recorded. No balanced
    physical-span or asymptotic near-linear claim is inferred from that split.
    """

    def __init__(self, observer, center, widths, times, frequencies=None, *,
                 grid_policy=None, order=2, max_nodes=500_000,
                 max_entries=20_000_000, max_step=.25, root_epoch=None,
                 transfer_backend="tensor", cascade_padding_rate=.25,
                 exact_leaves=False, leaf_kernel_builder=None, source_counts=None):
        started = perf_counter()
        self.observer = observer
        self.times = np.asarray(times, float)
        self.leaf_kernel_builder = leaf_kernel_builder
        self.frequencies = None if frequencies is None else [np.asarray(k, float) for k in frequencies]
        if (self.times.ndim != 1 or not len(self.times) or not np.all(np.isfinite(self.times))
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Strictly increasing finite times and nonempty (M,2) frequencies required")
        if self.frequencies is not None:
            if (leaf_kernel_builder is not None or source_counts is not None
                    or len(self.frequencies) != len(self.times)
                    or any(k.ndim != 2 or k.shape[1] != 2 or not len(k) or not np.all(np.isfinite(k))
                           for k in self.frequencies)):
                raise ValueError("Specify Fourier frequencies OR a custom kernel and source counts")
            self.source_counts = [len(k) for k in self.frequencies]
        else:
            if (not callable(leaf_kernel_builder) or source_counts is None
                    or len(source_counts) != len(self.times)
                    or any(isinstance(n, bool) or int(n) != n or n <= 0 for n in source_counts)):
                raise ValueError("Custom kernels require one positive integer source count per exposure")
            self.source_counts = list(map(int, source_counts))
        if any(isinstance(n, bool) or int(n) != n or n < 1 for n in [max_nodes, max_entries]):
            raise ValueError("Positive integer node and entry budgets required")
        self.max_entries, self.max_nodes = max_entries, max_nodes
        self.max_step, self.order = max_step, order
        if transfer_backend not in ["tensor", "cascade"]:
            raise ValueError("Unknown transfer backend")
        if not np.isfinite(cascade_padding_rate) or cascade_padding_rate < 0:
            raise ValueError("Nonnegative finite padding rate required")
        self.transfer_backend, self.cascade_padding_rate = transfer_backend, cascade_padding_rate
        self.exact_leaves = bool(exact_leaves)
        self.cascade_reports = []
        self.grid_policy = grid_policy or (lambda c, w, h, leaf: (1,)*(2 if leaf else 6))
        self.nodes, self.total_nodes, self.total_entries, self.rebased_states = [], 0, 0, 0
        self.source_offsets = np.r_[0, np.cumsum(self.source_counts)]
        epoch = float((self.times[0]+self.times[-1])/2 if root_epoch is None else root_epoch)
        if not np.isfinite(epoch):
            raise ValueError("Finite root reception epoch required")
        if len(self.times) == 1 and epoch != self.times[0]:
            raise ValueError("A single-exposure root must use its reception epoch")
        # Even a one-exposure leaf must describe a regular physical root box.
        rhs(arb(epoch), box_balls(center, widths), observer)
        self.root = self._geometry(np.arange(len(self.times)), epoch,
                                   np.asarray(center, float), np.asarray(widths, float))
        # Geometry/cover construction is complete before any data is accepted.
        self.construction_seconds = perf_counter()-started
        self.input_size, self.output_size = int(self.source_offsets[-1]), self.root.grid.size
        self.stats = self._stats()

    def _charge(self, nodes=0, entries=0):
        if self.total_nodes+nodes > self.max_nodes or self.total_entries+entries > self.max_entries:
            raise MemoryError("Total segment graph budget exceeded before allocation")
        self.total_nodes += nodes
        self.total_entries += entries

    def _kernel(self, exposure, positions):
        matrix = (np.exp(1j*positions @ self.frequencies[exposure].T) if self.frequencies is not None
                  else np.asarray(self.leaf_kernel_builder(exposure, positions), complex))
        if matrix.shape != (len(positions), self.source_counts[exposure]) or not np.all(np.isfinite(matrix)):
            raise ValueError("Custom exposure kernel returned invalid shape or nonfinite entries")
        return matrix

    def _geometry(self, indices, epoch, center, widths):
        leaf = len(indices) == 1
        span = float(max(abs(self.times[indices]-epoch)))
        cells = tuple(self.grid_policy(center.copy(), widths.copy(), span, leaf))
        order = self.order(center.copy(), widths.copy(), span, leaf) if callable(self.order) else self.order
        grid = TensorGrid(center, widths, cells, order, (3, 4) if leaf else tuple(range(6)))
        self._charge(nodes=grid.size)
        segment = Segment(indices, epoch, grid)
        self.nodes.append(segment)
        if leaf:
            self._charge(entries=grid.size*self.source_counts[indices[0]])
            # An exposure is evaluated at its own reception epoch; its four
            # hidden/rate coordinates drop out exactly.
            positions = grid.points()[:, [3, 4]]
            segment.leaf_kernel = self._kernel(indices[0], positions)
            return segment
        cut = len(indices)//2
        children, transfers = [], []
        points = grid.points()
        parent_chart = self.observer.chart(epoch)
        initial = box_balls(center, widths)
        for selected in [indices[:cut], indices[cut:]]:
            child_epoch = float((self.times[selected[0]]+self.times[selected[-1]])/2)
            enclosure, _, proof = enclose(initial, epoch, child_epoch, self.observer,
                                          max_step=self.max_step)
            child_center, child_widths = float_box(enclosure)
            if self.transfer_backend == "cascade" and len(selected) > 1:
                child_widths[3:5] *= 1+self.cascade_padding_rate*abs(child_epoch-epoch)
            child = self._geometry(selected, child_epoch, child_center, child_widths)
            if self.exact_leaves and len(selected) == 1:
                j = selected[0]
                self._charge(entries=grid.size*self.source_counts[j])
                rebased = parent_chart.rebase(points, child_epoch-epoch, self.observer.chart(child_epoch))
                self.rebased_states += len(points)
                matrix = ExactExposure(j, self._kernel(j, rebased[:, [3, 4]]))
            elif self.transfer_backend == "cascade" and len(selected) > 1:
                from .cascade_transfer import CascadeTransfer
                rebase = lambda q: parent_chart.rebase(q, child_epoch-epoch,
                                                       self.observer.chart(child_epoch))
                matrix = CascadeTransfer(grid, child.grid, rebase,
                                         max_entries=self.max_entries-self.total_entries)
                self._charge(entries=matrix.entries)
                self.rebased_states += matrix.geometry_points
                self.cascade_reports.append(matrix.report)
            else:
                self._charge(entries=grid.size*child.grid.stencil_size)
                rebased = parent_chart.rebase(points, child_epoch-epoch, self.observer.chart(child_epoch))
                self.rebased_states += len(points)
                matrix = child.grid.matrix(rebased, max_entries=self.max_entries)
            children.append(child)
            transfers.append(matrix)
            segment.proof_work += proof["steps"]
        segment.children, segment.transfers = tuple(children), tuple(transfers)
        return segment

    def _stats(self):
        memory = sum((node.leaf_kernel.nbytes if node.leaf_kernel is not None else
                      sum(m.array_bytes if hasattr(m, "array_bytes") else
                          m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in node.transfers))
                     for node in self.nodes)
        return {"exposures": len(self.times), "segments": len(self.nodes),
                "total_grid_nodes": self.total_nodes, "root_grid_nodes": self.root.grid.size,
                "stored_entries": self.total_entries, "array_bytes": memory,
                "rebased_states": self.rebased_states,
                "cover_flow_steps": sum(node.proof_work for node in self.nodes),
                "construction_seconds": self.construction_seconds,
                "parent_values_use_only_child_fields": True,
                "geometric_covers_exact_arithmetic_certified": True,
                "interpolation_accuracy_certified": False,
                "floating_point_evaluation_certified": False,
                "transfer_backend": self.transfer_backend,
                "exact_leaf_model_evaluation": self.exact_leaves,
                "leaf_model": "fourier" if self.frequencies is not None else "custom",
                "cascade_intermediate_domains_certified": False,
                "cascade_reports": self.cascade_reports,
                "full_objective_complete": False,
                "levels_by_node": [{"samples": len(n.indices),
                                   "half_span": float(max(abs(self.times[n.indices]-n.epoch))),
                                   "grid_shape": n.grid.shape} for n in self.nodes]}

    def transform(self, data):
        data = np.asarray(data, complex)
        if data.shape != (self.input_size,) or not np.all(np.isfinite(data)):
            raise ValueError("One finite complex coefficient per source required")

        def apply(node):
            if node.leaf_kernel is not None:
                j = node.indices[0]
                return node.leaf_kernel @ data[self.source_offsets[j]:self.source_offsets[j+1]]
            return sum((matrix.kernel @ data[self.source_offsets[matrix.exposure]:
                                             self.source_offsets[matrix.exposure+1]]
                        if isinstance(matrix, ExactExposure) else matrix @ apply(child))
                       for matrix, child in zip(node.transfers, node.children))

        return SegmentField(self, apply(self.root))

    def adjoint(self, root_values):
        root_values = np.asarray(root_values, complex)
        if root_values.shape != (self.output_size,) or not np.all(np.isfinite(root_values)):
            raise ValueError("One finite value per root grid node required")
        data = np.zeros(self.input_size, complex)

        def apply(node, values):
            if node.leaf_kernel is not None:
                j = node.indices[0]
                data[self.source_offsets[j]:self.source_offsets[j+1]] = node.leaf_kernel.conj().T @ values
            else:
                for matrix, child in zip(node.transfers, node.children):
                    if isinstance(matrix, ExactExposure):
                        j = matrix.exposure
                        data[self.source_offsets[j]:self.source_offsets[j+1]] += matrix.kernel.conj().T @ values
                    else:
                        apply(child, matrix.conj().T @ values)
        apply(self.root, root_values)
        return data

    def operator(self):
        return LinearOperator((self.output_size, self.input_size), dtype=complex,
                              matvec=lambda d: self.transform(d).values, rmatvec=self.adjoint)

    def regularized_inverse(self, values, *, damping, tolerance=1e-10, maxiter=200):
        """Solve min ||A d-values||_2^2 + damping^2 ||d||_2^2.

        This inverts the approximate, finite nodal operator with an explicitly
        chosen Euclidean measure. It does not invert the full continuous
        physical likelihood. The report includes iterations and normal residual.
        """
        if not np.isfinite(damping) or damping <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Positive finite damping and tolerance required")
        if isinstance(maxiter, bool) or int(maxiter) != maxiter or maxiter < 1:
            raise ValueError("Positive integer iteration budget required")
        values = np.asarray(values, complex)
        if values.shape != (self.output_size,) or not np.all(np.isfinite(values)):
            raise ValueError("Finite root nodal target required")
        started = perf_counter()
        operator = self.operator()
        answer = lsmr(operator, values, damp=damping, atol=tolerance, btol=tolerance, maxiter=maxiter)
        x, stop, iterations = answer[:3]
        residual = operator @ x-values
        normal = operator.rmatvec(residual)+damping*damping*x
        return x, {"damping": damping, "iterations": int(iterations), "stop_code": int(stop),
                   "normal_residual_norm": float(np.linalg.norm(normal)),
                   "data_residual_norm": float(np.linalg.norm(residual)),
                   "seconds": perf_counter()-started,
                   "converged": bool(stop in [0, 1, 2, 4, 5]),
                   "exact_unregularized_inverse": False,
                   "inverse_of": "finite interpolated root-nodal operator"}


class SegmentField:
    def __init__(self, transform, values):
        self.transform, self.values = transform, values

    def evaluate(self, parameters):
        grid = self.transform.root.grid
        q = np.asarray(parameters, float)
        if q.ndim != 2 or q.shape[1] != 6:
            raise ValueError("Queries must have shape (N,6)")
        if (not np.all(np.isfinite(q))
                or np.any(abs(q-grid.center) > grid.widths)):
            raise ValueError("Query leaves the full six-coordinate root domain")
        batch = max(1, self.transform.max_entries//grid.stencil_size)
        return np.concatenate([grid.matrix(q[j:j+batch], max_entries=self.transform.max_entries) @ self.values
                               for j in range(0, len(q), batch)]) if len(q) else np.empty(0, complex)

    def evaluate_adjoint(self, parameters, values):
        q, values = np.asarray(parameters, float), np.asarray(values, complex)
        if values.shape != (len(q),) or not np.all(np.isfinite(values)):
            raise ValueError("One finite value per query required")
        grid = self.transform.root.grid
        if (q.ndim != 2 or q.shape[1] != 6 or not np.all(np.isfinite(q))
                or np.any(abs(q-grid.center) > grid.widths)):
            raise ValueError("Query leaves the full six-coordinate root domain")
        matrix = self.transform.root.grid.matrix(q, max_entries=self.transform.max_entries)
        return self.transform.adjoint(matrix.conj().T @ values)
