"""Source-only demodulated interpolation over a continuous output box.

The complete output box is represented before any query points are supplied.
See global_theory.md for the conditional exact-arithmetic error theorem.
No sampled diagnostic is promoted to a uniform Kepler certificate.
"""

from itertools import product
from math import log, prod
from time import perf_counter

import numpy as np


def node_rule(order):
    if isinstance(order, bool) or not isinstance(order, int) or not 2 <= order <= 512:
        raise ValueError("Chebyshev order must be an integer in 2..512")
    angles = (np.arange(order) + .5) * np.pi / order
    return np.cos(angles), (-1.) ** np.arange(order) * np.sin(angles)


def cardinal(points, nodes, weights):
    points = np.asarray(points, float)
    difference = points[:, None] - nodes
    matches = difference == 0
    ordinary = ~np.any(matches, axis=1)
    result = np.zeros_like(difference)
    result[ordinary] = weights / difference[ordinary]
    result[ordinary] /= np.sum(result[ordinary], axis=1)[:, None]
    for row in np.flatnonzero(np.any(matches, axis=1)):
        result[row] = matches[row].astype(float)
    return result


def tensor_weights(points, nodes, weights):
    factors = [cardinal(points[:, axis], nodes, weights) for axis in range(points.shape[1])]
    result = factors[0]
    for factor in factors[1:]:
        result = (result[:, :, None] * factor[:, None, :]).reshape(len(points), -1)
    return result


def contract(values, matrices, *, adjoint=False):
    result = values
    axes = range(len(matrices)-1, -1, -1) if adjoint else range(len(matrices))
    for axis in axes:
        matrix = matrices[axis].conj().T if adjoint else matrices[axis]
        result = np.moveaxis(np.tensordot(matrix, result, axes=(1, axis)), 0, axis)
    return result


def box_center(index, depth):
    return -1 + (2 * np.asarray(index, float) + 1) / (2 ** depth)


def bin_indices(points, depth):
    return np.minimum(np.floor((points + 1) * (2 ** depth) / 2).astype(np.int64), 2 ** depth - 1)


def groups(points, depth):
    keys, inverse = np.unique(bin_indices(points, depth), axis=0, return_inverse=True)
    return {tuple(key): np.flatnonzero(inverse == i) for i, key in enumerate(keys)}


def theoretical_counts(output_dimension, source_dimension, depth, order):
    integers = [output_dimension, source_dimension, depth]
    if any(isinstance(x, bool) or not isinstance(x, int) for x in integers):
        raise ValueError("Integer dimensions, depth and order required")
    if not 1 <= output_dimension <= 8 or not 1 <= source_dimension <= 4 or not 0 <= depth <= 20:
        raise ValueError("Dimensions or depth outside supported limits")
    try:
        orders = (order,)*source_dimension if isinstance(order, int) else tuple(order)
    except TypeError as error:
        raise ValueError("Integer interpolation orders required") from error
    if len(orders) != source_dimension or any(isinstance(p, bool) or not isinstance(p, int) or p < 2 for p in orders):
        raise ValueError("Integer interpolation orders >=2 required")
    rank = prod(orders)
    pairs = [2 ** (output_dimension*l + source_dimension*(depth-l)) for l in range(depth+1)]
    edges = sum(pairs[1:]) * (2 ** source_dimension)
    leaves = 2 ** (output_dimension * depth)
    return {
        "output_dimension": output_dimension, "source_dimension": source_dimension,
        "depth": depth, "order": order, "orders": orders, "rank": rank,
        "full_tree_pairs_by_level": pairs, "full_tree_pairs": sum(pairs),
        "full_tree_transfer_edges": edges, "continuous_output_leaves": leaves,
        "full_tree_gate_bytes": 16 * rank * edges,
        "final_field_bytes": 16 * rank * leaves,
        "full_tree_tensor_contraction_multiply_adds": edges * rank * sum(orders),
    }


def conditional_error_bound(depth, order, source_dimension, rho, residual_bound):
    """Log-domain evaluation of the deliberately conservative theorem.

    rho and residual_bound are mathematical hypotheses, NOT inferred from
    samples or certified by this routine. This returns no floating-point proof.
    """
    if (not isinstance(depth, int) or depth < 0 or not isinstance(source_dimension, int)
            or source_dimension < 1 or not np.isfinite(rho) or rho <= 1
            or not np.isfinite(residual_bound) or residual_bound < 0):
        raise ValueError("Valid depth, dimensions, ellipse and residual bound required")
    node_rule(order)
    one = float(order ** 2)
    log_local = (log(2) + residual_bound - (order-1)*log(rho) - log(rho-1)
                 + log(1+one) + log(sum(one**j for j in range(source_dimension))))
    log_lambda = source_dimension * log(one)
    log_amplification = np.logaddexp.reduce(np.arange(depth+1)*log_lambda)
    value = float(log_local + log_amplification)
    return {"log_absolute_kernel_error_bound": value,
            "absolute_kernel_error_bound": float(np.exp(value)) if value < 700 else None,
            "hypotheses_verified": False, "floating_point_arithmetic_certified": False}


class SourceButterfly:
    def __init__(self, phase, sources, *, output_dimension, depth=1, order=8,
                 max_gate_bytes=512_000_000, max_pairs=200_000, moment_backend="auto"):
        start = perf_counter()
        sources = np.asarray(sources, float)
        if (sources.ndim != 2 or min(sources.shape) < 1 or not np.all(np.isfinite(sources))
                or np.any(abs(sources) > 1)):
            raise ValueError("Finite source points in [-1,1]^s required")
        count = theoretical_counts(output_dimension, sources.shape[1], depth, order)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in [max_gate_bytes, max_pairs]):
            raise ValueError("Positive integer construction budgets required")
        self.phase, self.sources = phase, sources.copy()
        self.dimension, self.source_dimension = output_dimension, sources.shape[1]
        self.depth, self.order, self.rank = depth, order, count["rank"]
        self.orders = tuple(count["orders"])
        self.rules = [node_rule(p) for p in self.orders]
        self.nodes = [rule[0] for rule in self.rules]
        if moment_backend not in ["auto", "dense", "nufft"]:
            raise ValueError("Moment backend must be auto, dense or nufft")
        source_groups = groups(sources, depth)
        source_levels = [sorted(source_groups)]
        for _ in range(depth):
            source_levels.append(sorted({tuple(v//2 for v in key) for key in source_levels[-1]}))
        actual_pairs = [2 ** (output_dimension*l) * len(source_levels[l]) for l in range(depth+1)]
        edges = sum(2 ** (output_dimension*l) * len(source_levels[l-1]) for l in range(1, depth+1))
        self.factorized_gates = hasattr(phase, "factorized_gate") and self.source_dimension == 3
        gate_size = self.orders[0]*sum(self.orders[1:]) if self.factorized_gates else self.rank
        gate_bytes = 16 * gate_size * edges
        live_field_bytes = 32 * self.rank * max(actual_pairs)
        backends = {key: ("nufft" if self.source_dimension <= 3
                    and (moment_backend == "nufft" or (moment_backend == "auto" and len(indices)*self.rank > 100_000))
                    else "dense") for key, indices in source_groups.items()}
        if moment_backend == "nufft" and self.source_dimension > 3:
            raise ValueError("Fast cosine moments support at most three source dimensions")
        initial_bytes = sum((8*len(indices)*self.rank if backends[key] == "dense" else
                             8*len(indices)*self.source_dimension*2**self.source_dimension)
                            for key, indices in source_groups.items())+16*len(sources)
        final_center_bytes = 16 * gate_size * count["continuous_output_leaves"]
        storage = gate_bytes + live_field_bytes + initial_bytes + final_center_bytes
        if sum(actual_pairs) > max_pairs or storage > max_gate_bytes:
            raise ValueError(
                f"Global hierarchy resource budget exceeded: pairs={sum(actual_pairs)}, "
                f"principal_complex_array_bytes={storage}; domain not reduced")
        self.grid = np.stack(np.meshgrid(*self.nodes, indexing="ij"), axis=-1).reshape(-1, self.source_dimension)
        self.report = {**count, "occupied_pairs_by_level": actual_pairs,
                       "occupied_transfer_edges": edges, "gate_bytes": gate_bytes,
                       "two_field_storage_bound_bytes": live_field_bytes,
                       "initial_interpolation_bytes": initial_bytes,
                       "final_center_bytes": final_center_bytes,
                       "principal_complex_array_bytes": storage,
                       "python_overhead_and_temporary_arrays_included": False,
                       "initial_moment_backends": {name: sum(v == name for v in backends.values())
                                                   for name in ["dense", "nufft"]},
                       "complete_output_box_represented": True,
                       "exact_spatial_frequency_gate_separation": self.factorized_gates,
                       "uniform_kepler_error_certified": False,
                       "full_objective_complete": False}
        self.kernel_entries = 0
        self.interpolation = [[
            cardinal((nodes + sign)/2, nodes, weights).T for sign in [-1, 1]]
            for nodes, weights in self.rules]
        self.initial = {}
        root = np.zeros((1, output_dimension))
        initial_phase = self._kernel(root, sources)[0]
        from .chebyshev_moments import CosineMoments, DenseMoments
        for key, indices in source_groups.items():
            local = (sources[indices] - box_center(key, depth)) * 2**depth
            moments = (CosineMoments(local, self.orders) if backends[key] == "nufft"
                       else DenseMoments(local, self.rules, self.orders))
            self.initial[key] = indices, moments, initial_phase[indices]
        self.layers = []
        for level in range(1, depth+1):
            target_nodes = list(product(range(2**level), repeat=output_dimension))
            child_depth = depth-level+1
            by_parent = {}
            for child in source_levels[level-1]:
                parent = tuple(v//2 for v in child)
                by_parent.setdefault(parent, []).append(child)
            layer = {}
            for target in target_nodes:
                parent_target = tuple(v//2 for v in target)
                centers = np.array([box_center(target, level), box_center(parent_target, level-1)])
                for source, children in by_parent.items():
                    gates = []
                    for child in children:
                        if self.factorized_gates:
                            child_center = box_center(child, child_depth)
                            locations = [c+n/(2**child_depth) for c, n in zip(child_center, self.nodes)]
                            gate = phase.factorized_gate(centers, locations)
                        else:
                            locations = box_center(child, child_depth) + self.grid/(2**child_depth)
                            phases = self._phases(centers, locations)
                            gate = np.exp(1j*(phases[0]-phases[1])).reshape(self.orders)
                        gates.append((child, gate))
                    layer[target, source] = gates
            self.layers.append(layer)
        # The final demodulation center gates can be reused for every input.
        self.final_centers = {}
        for key in product(range(2**depth), repeat=output_dimension):
            self.final_centers[key] = (phase.factorized_gate(box_center(key, depth)[None], self.nodes)
                if self.factorized_gates else self._kernel(box_center(key, depth)[None], self.grid)[0])
        self.report.update(construction_seconds=perf_counter()-start,
                           construction_kernel_entries=self.kernel_entries,
                           kernel_entry_counter_excludes_factorized_gates=self.factorized_gates,
                           factorized_gate_complex_entries=(gate_bytes+final_center_bytes)//16
                           if self.factorized_gates else 0)

    def _phases(self, targets, sources):
        result = np.asarray(self.phase(targets, sources), float)
        if result.shape != (len(targets), len(sources)) or not np.all(np.isfinite(result)):
            raise ValueError("Phase oracle returned incompatible/nonfinite values")
        self.kernel_entries += result.size
        return result

    def _kernel(self, targets, sources):
        return np.exp(1j*self._phases(targets, sources))

    def center_kernel(self, key):
        value = self.final_centers[key]
        return value.dense() if self.factorized_gates else value

    def apply_gate(self, gate, value, *, adjoint=False):
        if self.factorized_gates:
            return gate.apply(value, adjoint=adjoint)
        return (gate.conj() if adjoint else gate)*value

    def transform(self, data):
        data = np.asarray(data, complex)
        if data.shape != (len(self.sources),) or not np.all(np.isfinite(data)):
            raise ValueError("Finite coefficient for every input source required")
        current = {((0,)*self.dimension, key): moments.forward(phase*data[indices])
                   for key, (indices, moments, phase) in self.initial.items()}
        for layer in self.layers:
            following = {}
            for (target, source), gates in layer.items():
                previous_target = tuple(v//2 for v in target)
                value = np.zeros(self.orders, complex)
                for child, gate in gates:
                    matrices = [self.interpolation[i][v % 2] for i, v in enumerate(child)]
                    value += contract(self.apply_gate(gate, current[previous_target, child]), matrices)
                following[target, source] = value
            current = following
        return ContinuousField(self, {key: current[key, (0,)*self.source_dimension]
                                      for key in self.final_centers})

    def adjoint(self, targets, values, *, chunk=128):
        targets = self._targets(targets)
        values = np.asarray(values, complex)
        if values.shape != (len(targets),) or not np.all(np.isfinite(values)):
            raise ValueError("Finite value for each adjoint query required")
        current = {(key, (0,)*self.source_dimension): np.zeros(
            self.orders, complex) for key in self.final_centers}
        for key, indices in groups(targets, self.depth).items():
            for start in range(0, len(indices), chunk):
                rows = indices[start:start+chunk]
                kernel = self._kernel(targets[rows], self.grid) / self.center_kernel(key)
                current[key, (0,)*self.source_dimension] += (kernel.conj().T @ values[rows]).reshape(
                    self.orders)
        for layer in reversed(self.layers):
            previous = {}
            for (target, source), gates in layer.items():
                previous_target = tuple(v//2 for v in target)
                for child, gate in gates:
                    matrices = [self.interpolation[i][v % 2] for i, v in enumerate(child)]
                    value = self.apply_gate(gate, contract(current[target, source], matrices, adjoint=True),
                                            adjoint=True)
                    key = previous_target, child
                    if key in previous:
                        previous[key] += value
                    else:
                        previous[key] = value
            current = previous
        result = np.zeros(len(self.sources), complex)
        for key, (indices, moments, phase) in self.initial.items():
            result[indices] = phase.conj()*moments.adjoint(current[(0,)*self.dimension, key])
        return result

    def _targets(self, targets):
        targets = np.asarray(targets, float)
        if (targets.ndim != 2 or targets.shape[1] != self.dimension or not len(targets)
                or not np.all(np.isfinite(targets)) or np.any(abs(targets) > 1)):
            raise ValueError("Finite queries in the entire declared output box required")
        return targets


class ContinuousField:
    def __init__(self, plan, coefficients):
        self.plan, self.coefficients = plan, coefficients

    def evaluate(self, targets, *, chunk=128, accelerated=False):
        targets = self.plan._targets(targets)
        result = np.empty(len(targets), complex)
        for key, indices in groups(targets, self.plan.depth).items():
            if accelerated:
                if not hasattr(self.plan.phase, "evaluate_expansion"):
                    raise ValueError("This phase oracle has no accelerated expansion evaluator")
                result[indices] = self.plan.phase.evaluate_expansion(
                    targets[indices], self.plan.nodes, self.coefficients[key],
                    box_center(key, self.plan.depth))
                continue
            for start in range(0, len(indices), chunk):
                rows = indices[start:start+chunk]
                kernel = self.plan._kernel(targets[rows], self.plan.grid) / self.plan.center_kernel(key)
                result[rows] = kernel @ self.coefficients[key].ravel()
        return result
