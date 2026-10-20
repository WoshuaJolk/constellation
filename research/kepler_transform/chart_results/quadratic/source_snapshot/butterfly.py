"""Sampled interpolative butterfly with explicit construction accounting.

This is an experimental finite operator. Sampled residuals do NOT certify its
accuracy, including when every sampled block check passes. The kernel callback
receives only requested row/column indices; construction never needs a dense
global matrix. The transfer graph supplies its exact algebraic adjoint.
"""

from time import perf_counter

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.sparse.linalg import LinearOperator, lsqr


def tree(points, depth):
    """Nested balanced geometric partitions, retaining every supplied index."""
    points = np.asarray(points, float)
    levels = [[np.arange(len(points))]]
    for _ in range(depth):
        next_level = []
        for indices in levels[-1]:
            axis = np.argmax(np.ptp(points[indices], axis=0))
            order = indices[np.argsort(points[indices, axis], kind="stable")]
            middle = len(order)//2
            next_level.extend([order[:middle], order[middle:]])
        levels.append(next_level)
    return levels


class Butterfly:
    def __init__(self, kernel, row_points, column_points, *, rank_cap=16,
                 tolerance=1e-10, leaf_size=8, seed=20261020):
        started = perf_counter()
        rows, columns = np.asarray(row_points, float), np.asarray(column_points, float)
        if (rows.ndim != 2 or columns.ndim != 2 or min(rows.shape+columns.shape) < 1
                or not np.all(np.isfinite(rows)) or not np.all(np.isfinite(columns))):
            raise ValueError("Finite nonempty two-dimensional point arrays required")
        if (isinstance(rank_cap, bool) or not isinstance(rank_cap, int) or rank_cap < 1
                or isinstance(leaf_size, bool) or not isinstance(leaf_size, int) or leaf_size < 1
                or not np.isfinite(tolerance) or not 0 < tolerance < 1):
            raise ValueError("Positive integer rank/leaf limits and 0<tolerance<1 required")
        self.shape = len(rows), len(columns)
        depth = max(0, int(np.floor(np.log2(min(self.shape)/leaf_size))))
        row_tree, column_tree = tree(rows, depth), tree(columns, depth)
        rng = np.random.default_rng(seed)
        self.levels, self.leaves = [], []
        ranks, defects, limited = [], [], 0
        self.kernel_entries = 0

        def evaluate(r, c):
            self.kernel_entries += len(r)*len(c)
            result = np.asarray(kernel(r, c), complex)
            if result.shape != (len(r), len(c)) or not np.all(np.isfinite(result)):
                raise ValueError("Kernel returned wrong shape or nonfinite entries")
            return result

        def compress(target, sources):
            nonlocal limited
            if len(sources) <= rank_cap:
                # A true identity transfer: no kernel evaluation is necessary.
                return sources, None
            n_sample = min(len(target), 2*rank_cap+8)
            sample = np.sort(rng.choice(target, n_sample, replace=False))
            matrix = evaluate(sample, sources)
            _, upper, permutation = qr(matrix, mode="economic", pivoting=True,
                                        check_finite=False)
            tail = np.r_[np.cumsum(np.sum(abs(upper)**2, axis=1)[::-1])[::-1], 0.]
            eligible = np.flatnonzero(tail <= tolerance**2*tail[0])
            requested = max(1, int(eligible[0]))
            rank = min(requested, rank_cap)
            limited += requested > rank_cap
            ranks.append(rank)
            transfer = np.zeros((rank, len(sources)), complex)
            transfer[:, permutation[:rank]] = np.eye(rank)
            if rank < len(sources):
                transfer[:, permutation[rank:]] = solve_triangular(
                    upper[:rank, :rank], upper[:rank, rank:], check_finite=False)
            skeleton = sources[permutation[:rank]]
            available = np.setdiff1d(target, sample, assume_unique=False)
            # Withheld rows detect some failures, but cannot rule out all of them.
            check = np.sort(rng.choice(available, min(8, len(available)), replace=False))
            if len(check):
                truth = evaluate(check, sources)
                predicted = truth[:, permutation[:rank]] @ transfer
                defects.append(float(np.linalg.norm(truth-predicted)/max(1e-300, np.linalg.norm(truth))))
            return skeleton, transfer

        previous = None
        for level in range(depth+1):
            current = {}
            for a, target in enumerate(row_tree[level]):
                for b, source in enumerate(column_tree[depth-level]):
                    if level == 0:
                        candidates = source
                        split = None
                    else:
                        left = previous[a//2, 2*b][0]
                        right = previous[a//2, 2*b+1][0]
                        candidates = np.r_[left, right]
                        split = len(left)
                    skeleton, transfer = compress(target, candidates)
                    current[a, b] = skeleton, transfer, source if level == 0 else split
            self.levels.append(current)
            previous = current
        for a, target in enumerate(row_tree[-1]):
            self.leaves.append((target, evaluate(target, previous[a, 0][0])))
        stored = sum(block.size for _, block in self.leaves)
        stored += sum(value[1].size for level in self.levels for value in level.values()
                      if value[1] is not None)
        self.report = {
            "construction_seconds": perf_counter()-started,
            "depth": depth, "rank_cap": rank_cap, "requested_tolerance": tolerance,
            "kernel_entries": self.kernel_entries, "dense_kernel_entries": int(np.prod(self.shape)),
            "kernel_entry_ratio": self.kernel_entries/int(np.prod(self.shape)),
            "stored_complex_numbers": stored,
            "storage_ratio": stored/int(np.prod(self.shape)),
            "rank_limited_blocks": int(limited),
            "maximum_rank": max(ranks, default=0),
            "maximum_withheld_block_relative_residual": max(defects, default=0.),
            "continuous_error_certified": False,
            "finite_operator_error_certified": False,
            "accuracy_status": "sampled diagnostics only; external validation required",
        }

    def apply(self, values, *, adjoint=False):
        values = np.asarray(values, complex)
        if (values.ndim not in (1, 2) or values.shape[0] != self.shape[0 if adjoint else 1]
                or not np.all(np.isfinite(values))):
            raise ValueError("Finite vector or matrix with operator-compatible leading dimension required")
        extra = values.shape[1:]
        if not adjoint:
            previous = None
            for depth, level in enumerate(self.levels):
                current = {}
                for (a, b), (_, transfer, source) in level.items():
                    vector = (values[source] if depth == 0 else
                              np.concatenate((previous[a//2, 2*b], previous[a//2, 2*b+1]), axis=0))
                    current[a, b] = vector if transfer is None else transfer @ vector
                previous = current
            result = np.zeros((self.shape[0],)+extra, complex)
            for a, (target, block) in enumerate(self.leaves):
                result[target] = block @ previous[a, 0]
            return result
        current = { (a, 0): block.conj().T @ values[target]
                    for a, (target, block) in enumerate(self.leaves) }
        result = np.zeros((self.shape[1],)+extra, complex)
        for depth in range(len(self.levels)-1, -1, -1):
            previous = {}
            for (a, b), (_, transfer, source) in self.levels[depth].items():
                vector = current[a, b]
                if transfer is not None:
                    vector = transfer.conj().T @ vector
                if depth == 0:
                    result[source] += vector
                else:
                    for key, part in (((a//2, 2*b), vector[:source]),
                                      ((a//2, 2*b+1), vector[source:])):
                        if key not in previous:
                            previous[key] = part.copy()
                        else:
                            previous[key] += part
            current = previous
        return result

    def inverse(self, scores, *, damping=0., tolerance=1e-10, max_iterations=1000):
        """Minimum-norm LSQR (or Tikhonov) for THIS approximate finite operator.

        This does not restore a time-summation nullspace, certify conditioning,
        or assert convergence when an iteration limit is reached.
        """
        if (not np.isfinite(damping) or damping < 0 or not 0 < tolerance < 1
                or isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
                or max_iterations < 1):
            raise ValueError("Nonnegative finite damping, valid tolerance and iteration limit required")
        scores = np.asarray(scores, complex)
        if scores.shape != (self.shape[0],) or not np.all(np.isfinite(scores)):
            raise ValueError("One finite score per output row required")
        operator = LinearOperator(self.shape, matvec=self.apply,
                                  rmatvec=lambda x: self.apply(x, adjoint=True), dtype=complex)
        result = lsqr(operator, scores, damp=damping, atol=tolerance, btol=tolerance,
                      iter_lim=max_iterations)
        return result[0], {"method": "LSQR on approximate finite operator",
                           "damping": damping, "stop_code": result[1], "iterations": result[2],
                           "residual_norm": result[3], "condition_estimate": result[6],
                           "iteration_limit_reached": result[1] == 7,
                           "unique_image_reconstruction_claimed": False}
