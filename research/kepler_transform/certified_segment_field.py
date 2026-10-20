"""Full segment recurrence using proved cascade domains and refinement.

Accuracy and archive-scale cost are still research obligations. The new
guarantee concerns exact geometric coverage and intermediate inverse maps,
not floating-point calculations or an automatically sufficient grid rule.
"""

from time import perf_counter

import numpy as np
from flint import arb

from research.detection_limit.interval_orbits import upper_float
from .certified_cascade import CertifiedCascadeTransfer
from .reception_flow import box_balls, enclose, float_box
from .segment_field import ExactExposure, Segment, SegmentTransform, TensorGrid


class CertifiedSegmentTransform(SegmentTransform):
    def __init__(self, *args, certificate_options=None, **kwargs):
        self.certificate_options = dict(certificate_options or {})
        if "transfer_backend" in kwargs or "cascade_padding_rate" in kwargs:
            raise ValueError("Certified geometry replaces backend/padding selection")
        super().__init__(*args, transfer_backend="cascade", **kwargs)

    def _grid(self, indices, epoch, center, widths):
        leaf = len(indices) == 1
        span = float(max(abs(self.times[indices]-epoch)))
        cells = tuple(self.grid_policy(center.copy(), widths.copy(), span, leaf))
        order = self.order(center.copy(), widths.copy(), span, leaf) if callable(self.order) else self.order
        return TensorGrid(center, widths, cells, order, (3, 4) if leaf else tuple(range(6)))

    def _geometry(self, indices, epoch, center, widths, existing_grid=None):
        grid = self._grid(indices, epoch, center, widths) if existing_grid is None else existing_grid
        self._charge(nodes=grid.size)
        segment = Segment(indices, epoch, grid)
        self.nodes.append(segment)
        if len(indices) == 1:
            self._charge(entries=grid.size*self.source_counts[indices[0]])
            segment.leaf_kernel = self._kernel(indices[0], grid.points()[:, [3, 4]])
            return segment
        cut = len(indices)//2
        children, transfers = [], []
        parent_chart = self.observer.chart(epoch)
        for selected in [indices[:cut], indices[cut:]]:
            child_epoch = float((self.times[selected[0]]+self.times[selected[-1]])/2)
            if len(selected) > 1:
                matrix = CertifiedCascadeTransfer(
                    self.observer, grid, epoch, child_epoch,
                    certificate_options=self.certificate_options,
                    child_grid_factory=lambda c, w: self._grid(selected, child_epoch, c, w),
                    max_entries=self.max_entries-self.total_entries,
                    max_work_nodes=self.max_nodes)
                self._charge(entries=matrix.entries)
                child = self._geometry(selected, child_epoch, matrix.child.center,
                                       matrix.child.widths, matrix.child)
                self.rebased_states += matrix.remap.geometry_points
                self.cascade_reports.append(matrix.report)
                segment.proof_work += (matrix.certificate["point_flow_steps"]
                                       +matrix.certificate["derivative_flow_steps"])
            else:
                enclosure, _, proof = enclose(box_balls(center, widths), epoch, child_epoch,
                                              self.observer, max_step=self.max_step)
                cc, cw = float_box(enclosure)
                child = self._geometry(selected, child_epoch, cc, cw)
                points = grid.points()
                rebased = parent_chart.rebase(points, child_epoch-epoch, self.observer.chart(child_epoch))
                self.rebased_states += len(points)
                if self.exact_leaves:
                    j = selected[0]
                    self._charge(entries=grid.size*self.source_counts[j])
                    matrix = ExactExposure(j, self._kernel(j, rebased[:, [3, 4]]))
                else:
                    self._charge(entries=grid.size*child.grid.stencil_size)
                    matrix = child.grid.matrix(rebased, max_entries=self.max_entries)
                segment.proof_work += proof["steps"]
            children.append(child)
            transfers.append(matrix)
        segment.children, segment.transfers = tuple(children), tuple(transfers)
        return segment

    def _stats(self):
        stats = super()._stats()
        stats.update({
            "transfer_backend": "certified_cascade_with_separable_refinement",
            "cascade_intermediate_domains_certified": True,
            "complex_cascade_inverses_certified": all(
                r["complex_intermediate_inverses_certified"] for r in self.cascade_reports),
            "interpolation_accuracy_certified": False,
        })
        return stats


class CertifiedAtlasTransform:
    """Cover the original box, rebuilding every failed patch's full graph.

    Budgets apply to the sum of retained graphs. Failed attempts and their
    time are retained in the report. An incomplete atlas is never returned.
    """

    def __init__(self, observer, center, widths, times, frequencies=None, *,
                 axes=(0, 1), max_patches=64, max_nodes=500_000,
                 max_entries=20_000_000, **options):
        box_balls(center, widths)
        if (not axes or len(set(axes)) != len(axes) or any(a not in range(6) for a in axes)
                or not isinstance(max_patches, int) or max_patches < 1
                or max_nodes < 1 or max_entries < 1):
            raise ValueError("Valid subdivision axes and positive graph budgets required")
        self.center, self.widths = np.asarray(center, float), np.asarray(widths, float)
        self.transforms, refused = [], []
        queue = [(self.center, self.widths, 0)]
        nodes, entries = 0, 0
        begin = perf_counter()
        while queue:
            if nodes >= max_nodes or entries >= max_entries:
                raise MemoryError("The retained atlas exhausted its shared graph budget")
            c, w, depth = queue.pop()
            attempt_start = perf_counter()
            try:
                transform = CertifiedSegmentTransform(
                    observer, c, w, times, frequencies, max_nodes=max_nodes-nodes,
                    max_entries=max_entries-entries, **options)
            except ValueError as error:
                refused.append({"depth": depth, "reason": str(error),
                                "seconds": perf_counter()-attempt_start})
                if len(self.transforms)+len(queue)+2 > max_patches:
                    raise ValueError("Full graph cover exceeds its patch budget") from error
                axis = axes[depth % len(axes)]
                half = arb(float(w[axis]))/2
                for sign in [-1, 1]:
                    ec = arb(float(c[axis]))+sign*half
                    nc, nw = c.copy(), w.copy()
                    nc[axis] = float(ec.mid())
                    nw[axis] = upper_float(half+abs(ec-arb(float(nc[axis]))))
                    queue.append((nc, nw, depth+1))
                continue
            self.transforms.append(transform)
            nodes += transform.total_nodes
            entries += transform.total_entries
        self.input_size = self.transforms[0].input_size
        self.offsets = np.r_[0, np.cumsum([t.output_size for t in self.transforms])]
        self.output_size = int(self.offsets[-1])
        self.stats = {
            "original_center": self.center.tolist(), "original_widths": self.widths.tolist(),
            "whole_original_box_covered_by_outward_bisections": True,
            "patches": len(self.transforms), "retained_nodes": nodes,
            "patch_domains": [{"center": t.root.grid.center.tolist(),
                               "widths": t.root.grid.widths.tolist()} for t in self.transforms],
            "stored_entries": entries, "refused_graph_attempts": refused,
            "construction_seconds_including_refusals": perf_counter()-begin,
            "graphs": [t.stats for t in self.transforms],
            "interpolation_accuracy_certified": False,
            "floating_point_evaluation_certified": False, "full_objective_complete": False,
        }

    def transform(self, data):
        return AtlasField(self, [t.transform(data) for t in self.transforms])

    def adjoint(self, values):
        values = np.asarray(values, complex)
        if values.shape != (self.output_size,) or not np.all(np.isfinite(values)):
            raise ValueError("Finite values for all atlas root nodes required")
        return sum(t.adjoint(values[a:b]) for t, a, b in
                   zip(self.transforms, self.offsets[:-1], self.offsets[1:]))

    operator = SegmentTransform.operator
    regularized_inverse = SegmentTransform.regularized_inverse

    def owners(self, parameters):
        q = np.asarray(parameters, float)
        if (q.ndim != 2 or q.shape[1] != 6 or not np.all(np.isfinite(q))
                or np.any(abs(q-self.center) > self.widths)):
            raise ValueError("Queries must lie in the original six-coordinate domain")
        owners = np.full(len(q), -1)
        for index, t in enumerate(self.transforms):
            grid = t.root.grid
            belongs = np.all(abs(q-grid.center) <= grid.widths, axis=1)
            owners[(owners < 0) & belongs] = index
        if np.any(owners < 0):
            raise ValueError("Numerical query lookup could not locate a covering patch")
        return q, owners


class AtlasField:
    def __init__(self, transform, fields):
        self.transform, self.fields = transform, fields
        self.values = np.concatenate([f.values for f in fields])

    def evaluate(self, parameters):
        q, owners = self.transform.owners(parameters)
        result = np.empty(len(q), complex)
        for index, field in enumerate(self.fields):
            chosen = owners == index
            if np.any(chosen):
                result[chosen] = field.evaluate(q[chosen])
        return result

    def evaluate_adjoint(self, parameters, values):
        q, owners = self.transform.owners(parameters)
        values = np.asarray(values, complex)
        if values.shape != (len(q),) or not np.all(np.isfinite(values)):
            raise ValueError("One finite value per query required")
        result = np.zeros(self.transform.input_size, complex)
        for index, field in enumerate(self.fields):
            chosen = owners == index
            if np.any(chosen):
                result += field.evaluate_adjoint(q[chosen], values[chosen])
        return result
