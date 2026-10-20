"""Plan and optionally execute a hierarchy of locally sufficient Kepler grids.

Incoming transfer requirements choose each child's stored grid. This avoids a
circular refine/rebase budget: every transfer starts with nodal data on its
own certified working grid. No internal transfer re-reads exposure data.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import math
import tempfile
from time import perf_counter

import numpy as np
from flint import arb

from research.detection_limit.interval_orbits import upper_float
from .cascade_domain import certify_domain
from .cascade_transfer import CascadeTransfer
from .local_resolution import PERMUTATION, local_grid
from .segment_field import TensorGrid


def transfer_cost(parent_shape, child_shape, child_orders, permutation=PERMUTATION,
                  max_iterations=40):
    """Exact array-operation counts; geometry bound uses the solver's cap.

    The bound counts rebase calls on individual states. It does not assert
    constant arithmetic cost for numerical/validated orbit evaluation.
    """
    if (any(len(x) != 6 for x in (parent_shape, child_shape, child_orders))
            or sorted(permutation) != list(range(6))
            or any(type(v) is not int or v < 1 for x in
                   (parent_shape, child_shape, child_orders) for v in x)):
        raise ValueError("Six positive integer sizes/orders and a valid permutation required")
    shape, rows, entries = list(child_shape), [], []
    for axis in permutation:
        shape[axis] = parent_shape[axis]
        rows.append(math.prod(shape))
        entries.append(rows[-1]*child_orders[axis])
    return {"intermediate_nodes": rows, "stage_entries": entries,
            "stored_entries": sum(entries), "multiply_add_pairs": sum(entries),
            "peak_intermediate_nodes": max([math.prod(child_shape)]+rows),
            "rebase_state_evaluations_upper_bound": 13+max_iterations*sum(rows[:-1])+rows[-1],
            "coefficient_values_bytes": 8*sum(entries),
            "fits_64bit_sparse_indices": max(entries+rows+[math.prod(child_shape)]) < 2**63,
            "csr_bytes_if_64bit_indices_fit": (16*sum(entries)+8*sum(n+1 for n in rows)
                                               if max(entries+rows+[math.prod(child_shape)]) < 2**63 else None),
            "geometry_iteration_cap": max_iterations,
            "cost_per_rebase_state_not_assumed_constant": True}


@dataclass
class PlannedNode:
    first: int
    last: int
    epoch: float
    center: list
    widths: list
    grid: dict
    nodal_budget: float
    children: tuple = ()
    edges: tuple = ()
    error: float = 0.

    def record(self):
        return {"first": self.first, "last": self.last, "epoch": self.epoch,
                "center": list(self.center), "widths": list(self.widths),
                "grid": self.grid, "nodal_budget": self.nodal_budget,
                "nodal_error_bound_per_source_l1": self.error,
                "children": [x.record() for x in self.children], "edges": list(self.edges)}


def _down(value):
    answer = float(value.lower())
    if answer <= 0 or not np.isfinite(answer):
        raise ValueError("Error budget cannot be represented by a positive finite float")
    # Float conversion is round-to-nearest; step down to preserve allocation.
    return float(np.nextafter(answer, 0.))


class LocalHierarchyPlan:
    def __init__(self, observer, center, widths, times, frequency_bounds, *,
                 tolerance=1e-8, neighborhood_ellipses=(1.05, 2., 4., 8., 16.),
                 max_step=.03125, max_order=64, certificate_options=None, progress=None,
                 cache_directory=None):
        started = perf_counter()
        self.observer = observer
        self.progress = progress
        self.cache_directory = None if cache_directory is None else Path(cache_directory)
        self.cached_grids = 0
        self.cached_planning_seconds = 0.
        if self.cache_directory is not None:
            self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.source_digest = hashlib.sha256(b"".join(
            str(p).encode()+p.read_bytes() for p in sorted(list(Path(__file__).parent.glob("*.py"))+
                [Path(__file__).parents[2]/"src/magictools/orbits.py"])
        )).hexdigest()
        self.times = np.asarray(times, float)
        if (self.times.ndim != 1 or not len(self.times) or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0) or not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("Strictly increasing finite times and positive tolerance required")
        self.frequency_bounds = list(frequency_bounds)
        self.options = {"neighborhood_ellipses": neighborhood_ellipses,
                        "max_step": max_step, "max_order": max_order}
        self.certificate_options = {"max_step": max_step, **(certificate_options or {})}
        self.nodes, self.edges = [], []
        epoch = float((self.times[0]+self.times[-1])/2)
        identity = certify_domain(observer, center, widths, epoch, epoch, **self.certificate_options)
        root_grid = self._grid(identity, [self.times[0], self.times[-1]], _down(arb(tolerance)/2))
        root_budget = _down(arb(tolerance)/(2*arb(root_grid["input_error_amplification"])))
        self.root = self._node(0, len(self.times), epoch, list(center), list(widths),
                               root_grid, root_budget)
        self.root_identity_certificate = identity
        self.error = upper_float(arb(root_grid["error_bound"])+
                                 arb(root_grid["input_error_amplification"])*self.root.error)
        if not arb(self.error) <= arb(tolerance):
            raise ValueError("Complete hierarchy error budget was not met")
        self.stats = {"exposures": len(self.times), "tree_nodes": len(self.nodes),
                      "transfers": len(self.edges), "root_nodes": root_grid["nodes"],
                      "total_stored_field_nodes": sum(x.grid["nodes"] for x in self.nodes),
                      "total_transfer_entries": sum(e["cost"]["stored_entries"] for e in self.edges),
                      "total_transfer_multiply_add_pairs": sum(e["cost"]["multiply_add_pairs"] for e in self.edges),
                      "total_geometry_state_evaluations_bound": sum(e["cost"]["rebase_state_evaluations_upper_bound"] for e in self.edges),
                      "sum_intermediate_nodes": sum(sum(e["cost"]["intermediate_nodes"]) for e in self.edges),
                      "peak_field_or_intermediate_nodes": max([x.grid["nodes"] for x in self.nodes]+
                                                              [e["cost"]["peak_intermediate_nodes"] for e in self.edges]),
                      "coefficient_storage_bytes": 8*sum(e["cost"]["stored_entries"] for e in self.edges),
                      "source_preparation_included": False,
                      "query_stencil_nodes": math.prod(root_grid["orders"]),
                      "uniform_error_per_source_l1": self.error, "requested_tolerance": tolerance,
                      "exact_arithmetic_full_tree_interpolation_certified": True,
                      "plan_construction_seconds_including_refusals": perf_counter()-started,
                      "cached_grids_reused": self.cached_grids,
                      "planning_seconds_charging_cached_work": perf_counter()-started+self.cached_planning_seconds,
                      "original_grid_planning_seconds_including_reused_work": sum(
                          n.grid["construction_seconds"] for n in self.nodes),
                      "grid_allocated": False, "floating_point_evaluation_certified": False,
                      "archive_cost_scaling_proved": False, "full_objective_complete": False}

    def _grid(self, proof, time_range, tolerance):
        from flint import ctx
        # Source-addressed restart cache. Every retained certificate is replayed
        # independently by the campaign verifier; cache hits do not replace it.
        # Observer identity must be explicit: caching is limited to the built-in
        # canonical circular model, whose configuration is fixed by its source.
        from .reception_flow import CircularObserver
        use_cache = (self.cache_directory is not None and type(self.observer) is CircularObserver
                     and np.array_equal(self.observer.axes, np.eye(3))
                     and all(self.observer.dual_axes[i][j] == arb(int(i == j))
                             for i in range(3) for j in range(3)))
        request = {"source_digest": self.source_digest, "precision": ctx.prec,
                   "circular_observer_rate": self.observer.rate if use_cache else None,
                   "geometry": {k: proof[k] for k in ["parent_center", "parent_widths",
                                                       "child_center", "child_widths", "start", "stop"]},
                   "time_range": list(time_range), "frequency_bounds": self.frequency_bounds,
                   "tolerance": tolerance, "options": self.options}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = None if not use_cache else self.cache_directory/(key+".json")
        if path is not None and path.exists():
            saved = json.loads(path.read_text())
            serialized = json.dumps(saved["grid"], sort_keys=True)
            if saved["key"] != key or hashlib.sha256(serialized.encode()).hexdigest() != saved["grid_sha256"]:
                raise ValueError("Corrupt local grid restart cache")
            self.cached_grids += 1
            seconds = saved["grid"]["construction_seconds"]
            self.cached_planning_seconds += seconds
            if self.progress is not None:
                self.progress({"cache_reuse_seconds": seconds})
            return saved["grid"]
        grid = local_grid(proof, self.observer, time_range, self.frequency_bounds,
                          tolerance, **self.options)
        if path is not None:
            serialized = json.dumps(grid, sort_keys=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
                json.dump({"key": key, "grid_sha256": hashlib.sha256(
                    serialized.encode()).hexdigest(), "grid": grid}, stream, allow_nan=False)
            Path(stream.name).replace(path)
        return grid

    def _node(self, first, last, epoch, center, widths, grid, budget):
        node = PlannedNode(first, last, epoch, center, widths, grid, budget)
        self.nodes.append(node)
        if self.progress is not None:
            self.progress({"first": first, "last": last, "grid_nodes": grid["nodes"],
                           "orders": grid["orders"], "cells": grid["cells"]})
        if last-first == 1:
            return node
        children, edges, errors = [], [], []
        cut = (first+last)//2
        for a, b in ((first, cut), (cut, last)):
            child_epoch = float((self.times[a]+self.times[b-1])/2)
            proof = certify_domain(self.observer, center, widths, epoch, child_epoch,
                                   **self.certificate_options)
            child_grid = self._grid(proof, [self.times[a], self.times[b-1]], _down(arb(budget)/2))
            child_budget = _down(arb(budget)/(2*arb(child_grid["input_error_amplification"])))
            child = self._node(a, b, child_epoch, proof["child_center"], proof["child_widths"],
                               child_grid, child_budget)
            cost = transfer_cost(grid["shape"], child_grid["shape"], child_grid["orders"])
            edge = {"first": a, "last": b, "certificate": proof, "cost": cost}
            self.edges.append(edge)
            errors.append(arb(child_grid["error_bound"])+
                          arb(child_grid["input_error_amplification"])*child.error)
            children.append(child)
            edges.append(edge)
        node.children, node.edges = tuple(children), tuple(edges)
        # Child coefficient blocks are disjoint. Per *total* source l1 norm,
        # their weighted average is bounded by max, not a gratuitous sum.
        node.error = upper_float(max(errors))
        if not arb(node.error) <= arb(budget):
            raise ValueError("A subtree exceeded its nodal error budget")
        return node

    def record(self):
        return {"stats": self.stats, "times": self.times.tolist(),
                "frequency_bounds": self.frequency_bounds, "local_options": self.options,
                "certificate_options": self.certificate_options,
                "root_identity_certificate": self.root_identity_certificate,
                "root": self.root.record()}

    def materialize(self, frequencies, *, max_nodes=1_000_000, max_entries=20_000_000):
        """Allocate only after a complete plan and all global budgets pass."""
        if (self.stats["total_stored_field_nodes"] > max_nodes
                or self.stats["peak_field_or_intermediate_nodes"] > max_nodes
                or self.stats["total_transfer_entries"] > max_entries):
            raise MemoryError("Sufficient hierarchy exceeds allocation budgets; plan retained")
        return LocalHierarchy(self, frequencies, max_entries=max_entries)


class LocalHierarchy:
    def __init__(self, plan, frequencies, *, max_entries):
        self.plan, self.max_entries = plan, max_entries
        self.frequencies = [np.asarray(k, float) for k in frequencies]
        K = np.asarray(plan.frequency_bounds)
        if (len(self.frequencies) != len(plan.times) or any(
                k.ndim != 2 or k.shape[1] != 2 or not len(k) or not np.isfinite(k).all()
                or np.any(abs(k) > K) for k in self.frequencies)):
            raise ValueError("Exposure frequencies must lie inside the planned bounds")
        leaf_entries = sum(node.grid["nodes"]*len(self.frequencies[node.first])
                           for node in plan.nodes if not node.children)
        if leaf_entries+plan.stats["total_transfer_entries"] > max_entries:
            raise MemoryError("Combined leaf kernel and transfer budget exceeded before allocation")
        self.offsets = np.r_[0, np.cumsum([len(k) for k in self.frequencies])]
        self.grids, self.transfers, self.kernels = {}, {}, {}
        started = perf_counter()
        for node in plan.nodes:
            grid = TensorGrid(node.center, node.widths, tuple(node.grid["cells"]), tuple(node.grid["orders"]))
            self.grids[id(node)] = grid
            if not node.children:
                # Kernel work is charged separately; no exposure data at internal nodes.
                if grid.size*len(self.frequencies[node.first]) > max_entries:
                    raise MemoryError("Leaf kernel budget exceeded")
                self.kernels[id(node)] = np.exp(1j*grid.points()[:, [3, 4]] @ self.frequencies[node.first].T)
        for node in plan.nodes:
            for child, edge in zip(node.children, node.edges):
                proof = edge["certificate"]
                chart, target = plan.observer.chart(node.epoch), plan.observer.chart(child.epoch)
                def rebase(q, chart=chart, target=target, node=node, child=child, proof=proof):
                    if np.any(abs((q-np.asarray(node.center))/node.widths)
                              > np.array(proof["normalized_enclosing_radii"])):
                        raise ValueError("Numerical geometry left its certified enclosing box")
                    return chart.rebase(q, child.epoch-node.epoch, target)
                self.transfers[id(child)] = CascadeTransfer(self.grids[id(node)], self.grids[id(child)],
                                                            rebase, max_entries=max_entries)
        self.stats = {**plan.stats, "grid_allocated": True,
                      "materialization_seconds": perf_counter()-started,
                      "actual_transfer_entries": sum(t.entries for t in self.transfers.values()),
                      "actual_geometry_state_evaluations": sum(t.geometry_points for t in self.transfers.values()),
                      "leaf_kernel_entries": sum(k.size for k in self.kernels.values())}

    def transform(self, data):
        data = np.asarray(data)
        if data.shape != (self.offsets[-1],):
            raise ValueError("One coefficient per planned exposure frequency required")
        def visit(node):
            if not node.children:
                return self.kernels[id(node)] @ data[self.offsets[node.first]:self.offsets[node.first+1]]
            return sum(self.transfers[id(child)] @ visit(child) for child in node.children)
        return visit(self.plan.root)

    def adjoint(self, values):
        values = np.asarray(values)
        if values.shape != (self.plan.root.grid["nodes"],):
            raise ValueError("One adjoint value per root node required")
        result = np.zeros(self.offsets[-1], dtype=complex)
        def visit(node, v):
            if not node.children:
                result[self.offsets[node.first]:self.offsets[node.first+1]] = self.kernels[id(node)].conj().T @ v
            else:
                for child in node.children:
                    visit(child, self.transfers[id(child)].adjoint(v))
        visit(self.plan.root, values)
        return result

    def evaluate(self, values, queries):
        return self.grids[id(self.plan.root)].matrix(queries, max_entries=self.max_entries) @ values


class LocalAtlasPlan:
    """Retain a sufficient complete tree on every patch of an outward cover."""
    def __init__(self, observer, center, widths, times, frequency_bounds, *,
                 max_patches=64, initial_patches=None, progress=None, **options):
        from .verify_constructive_campaign import verify_rectangular_cover
        started = perf_counter()
        if initial_patches is None:
            initial_patches = [{"center": list(center), "widths": list(widths)}]
        verify_rectangular_cover(center, widths, initial_patches)
        pending = [(np.asarray(p["center"], float), np.asarray(p["widths"], float), 0)
                   for p in initial_patches]
        if not isinstance(max_patches, int) or max_patches < len(pending):
            raise ValueError("Positive atlas budget must cover the initial partition")
        self.plans, self.failed = [], []
        cached_seconds = 0.
        def forward(record):
            nonlocal cached_seconds
            cached_seconds += record.get("cache_reuse_seconds", 0.)
            if progress is not None:
                progress(record)
        while pending:
            c, w, depth = pending.pop()
            begin = perf_counter()
            try:
                plan = LocalHierarchyPlan(observer, c, w, times, frequency_bounds,
                                          progress=forward, **options)
                self.plans.append(plan)
            except ValueError as error:
                self.failed.append({"center": c.tolist(), "widths": w.tolist(),
                                    "reason": str(error), "seconds": perf_counter()-begin})
                if progress is not None:
                    progress({"refused_patch": self.failed[-1]})
                if len(pending)+len(self.plans)+2 > max_patches:
                    raise ValueError("Sufficient full-coverage atlas exceeds budget; no partial certificate") from error
                axis, half = depth % 2, arb(float(w[depth % 2]))/2
                for sign in (-1, 1):
                    exact = arb(float(c[axis]))+sign*half
                    nc, nw = c.copy(), w.copy()
                    nc[axis] = float(exact.mid())
                    nw[axis] = upper_float(half+abs(exact-arb(float(nc[axis]))))
                    pending.append((nc, nw, depth+1))
        patches = [{"center": p.root.center, "widths": p.root.widths} for p in self.plans]
        verify_rectangular_cover(center, widths, patches)
        sum_keys = ["root_nodes", "total_stored_field_nodes", "total_transfer_entries",
                    "total_transfer_multiply_add_pairs", "total_geometry_state_evaluations_bound",
                    "sum_intermediate_nodes", "coefficient_storage_bytes"]
        self.stats = {k: sum(p.stats[k] for p in self.plans) for k in sum_keys}
        self.stats.update({"patches": len(patches), "failed_patch_attempts": len(self.failed),
                           "uniform_error_per_source_l1": max(p.error for p in self.plans),
                           "original_box_covered_with_exact_rationals": True,
                           "total_planning_seconds_including_refusals": perf_counter()-started,
                           "total_planning_seconds_charging_cached_work": perf_counter()-started+cached_seconds,
                           "peak_field_or_intermediate_nodes": max(p.stats["peak_field_or_intermediate_nodes"] for p in self.plans),
                           "grid_allocated": False, "floating_point_evaluation_certified": False,
                           "archive_cost_scaling_proved": False, "full_objective_complete": False})
        self.center, self.widths, self.patches = list(center), list(widths), patches

    def record(self):
        return {"original_center": self.center, "original_widths": self.widths,
                "patch_domains": self.patches, "stats": self.stats,
                "refused_attempts": self.failed, "plans": [p.record() for p in self.plans]}
