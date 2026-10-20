"""Experimental six-dimensional cascade remapping of a Kepler field.

Inspired by Purser & Leslie (1991), DOI
10.1175/1520-0493(1991)119<2492:AEIPFH>2.0.CO;2.

Mixed-coordinate inverses here are numerical, with checked residuals at every
constructed grid node. Their existence/uniqueness on whole intermediate boxes
is NOT certified. This backend must never inherit the tensor backend's uniform
error certificate. It uses the full physical rebase, not split approximate
Kepler dynamics.
"""

import math

import numpy as np
from scipy.sparse import csr_matrix

from .segment_field import lobatto_rule
from .source_butterfly import cardinal


def axis_nodes(grid, axis):
    index = grid.axes.index(axis)
    count = grid.cells[index]
    nodes, _ = lobatto_rule(grid.orders[index])
    return grid.center[axis]+grid.widths[axis]*(
        ((2*np.arange(count)[:, None]+1+nodes)/count-1).ravel())


class CascadeTransfer:
    """Map child grid values to parent grid with six 1D interpolation stages."""

    def __init__(self, parent, child, rebase, *, permutation=(3, 4, 5, 2, 1, 0),
                 max_entries=20_000_000, max_iterations=40, tolerance=2e-9):
        if parent.axes != tuple(range(6)) or child.axes != tuple(range(6)):
            raise ValueError("Cascade needs two full six-dimensional grids")
        if sorted(permutation) != list(range(6)):
            raise ValueError("Permutation must contain each coordinate once")
        self.shape = (parent.size, child.size)
        self.stages = []
        self.geometry_points = 0
        self.maximum_residual = 0.
        self.maximum_iterations = 0
        self.entries = 0
        self.permutation = tuple(permutation)
        self.parent, self.child = parent, child
        grids = [axis_nodes(child, a) for a in range(6)]
        root = parent.center.copy()
        mapped = rebase(root[None])[0]
        self.geometry_points += 1
        # A single center Jacobian preconditions fixed-Jacobian Newton.
        h = 1e-3
        samples = np.tile(root, (12, 1))
        for a in range(6):
            samples[2*a, a] += h*parent.widths[a]
            samples[2*a+1, a] -= h*parent.widths[a]
        responses = rebase(samples)
        self.geometry_points += 12
        jacobian = ((responses[::2]-responses[1::2])/(2*h)).T/child.widths[:, None]
        updated = []
        previous_shape = tuple(len(g) for g in grids)
        for axis in permutation:
            grids[axis] = axis_nodes(parent, axis)
            current_shape = tuple(len(g) for g in grids)
            rows = math.prod(current_shape)
            order = child.orders[axis]
            entries = rows*order
            if self.entries+entries > max_entries:
                raise MemoryError("Cascade intermediate-grid budget exceeded before allocation")
            updated.append(axis)
            unknown = [a for a in range(6) if a not in updated]
            hybrid = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 6)
            guess = np.tile(root, (rows, 1))
            guess[:, updated] = hybrid[:, updated]
            if unknown:
                inverse = np.linalg.inv(jacobian[np.ix_(unknown, unknown)])
                # Predict from the center and the already fixed input axes.
                normalized_fixed = (guess[:, updated]-root[updated])/parent.widths[updated]
                desired = (hybrid[:, unknown]-mapped[unknown])/child.widths[unknown]
                correction = desired-normalized_fixed @ jacobian[np.ix_(unknown, updated)].T
                guess[:, unknown] += (correction @ inverse.T)*parent.widths[unknown]
                for iteration in range(max_iterations):
                    projected = rebase(guess)
                    self.geometry_points += rows
                    residual = (projected[:, unknown]-hybrid[:, unknown])/child.widths[unknown]
                    maximum = float(np.max(abs(residual)))
                    if maximum < tolerance:
                        self.maximum_residual = max(self.maximum_residual, maximum)
                        self.maximum_iterations = max(self.maximum_iterations, iteration+1)
                        break
                    guess[:, unknown] -= (residual @ inverse.T)*parent.widths[unknown]
                else:
                    raise ValueError("Mixed-coordinate inverse did not converge at every node")
            else:
                projected = rebase(guess)
                self.geometry_points += rows
            z = (projected[:, axis]-child.center[axis])/child.widths[axis]
            if np.any(abs(z) > 1):
                raise ValueError(f"Cascade intermediate grid leaves child box on axis {axis}; "
                                 "no extrapolation or sample dropping allowed")
            count = child.cells[axis]
            local = (z+1)*count/2
            cell = np.minimum(np.floor(local).astype(int), count-1)
            nodes, bary = lobatto_rule(order)
            weights = cardinal(2*(local-cell)-1, nodes, bary)
            index = np.array(np.unravel_index(np.arange(rows), current_shape)).T
            strides = np.array([math.prod(previous_shape[a+1:]) for a in range(6)])
            base = index @ strides-index[:, axis]*strides[axis]
            source = (base[:, None]
                      +(cell[:, None]*order+np.arange(order))*strides[axis])
            matrix = csr_matrix((weights.ravel(), source.ravel(),
                                 np.arange(rows+1)*order),
                                shape=(rows, math.prod(previous_shape)))
            self.stages.append(matrix)
            self.entries += entries
            previous_shape = current_shape
        self.array_bytes = sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in self.stages)
        self.report = {"stored_entries": self.entries, "array_bytes": self.array_bytes,
                       "geometry_points_including_iterations": self.geometry_points,
                       "maximum_mixed_inverse_scaled_residual": self.maximum_residual,
                       "maximum_mixed_inverse_iterations": self.maximum_iterations,
                       "intermediate_domains_certified": False,
                       "whole_box_error_certified": False}

    def __matmul__(self, values):
        result = values
        for matrix in self.stages:
            result = matrix @ result
        return result

    def adjoint(self, values):
        result = values
        for matrix in reversed(self.stages):
            result = matrix.conj().T @ result
        return result

    def conj(self):
        # All interpolation matrices are real.
        return self

    @property
    def T(self):
        class Transpose:
            def __init__(self, parent):
                self.parent = parent

            def __matmul__(self, values):
                return self.parent.adjoint(values)
        return Transpose(self)
