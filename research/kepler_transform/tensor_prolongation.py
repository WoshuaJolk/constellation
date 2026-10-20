"""Separable refinement of an existing piecewise tensor polynomial.

Each destination cell must lie inside a source cell and have at least its
polynomial degree. The represented polynomial is then unchanged in exact
arithmetic. This does not re-read images or re-evaluate any orbit.
"""

import math

import numpy as np
from scipy.sparse import csr_matrix

from .segment_field import lobatto_rule
from .source_butterfly import cardinal


class TensorProlongation:
    def __init__(self, source, target, *, max_entries=20_000_000,
                 max_work_nodes=1_000_000):
        if (source.axes != target.axes or not np.array_equal(source.center, target.center)
                or not np.array_equal(source.widths, target.widths)
                or any(b % a for a, b in zip(source.cells, target.cells))
                or any(b < a for a, b in zip(source.orders, target.orders))):
            raise ValueError("Refinement needs the same box, nested cells and nondecreasing orders")
        if max_work_nodes < 1 or max_entries < 1:
            raise ValueError("Positive work-node and matrix-entry budgets required")
        self.source, self.target = source, target
        self.shape = target.size, source.size
        self.stages, self.entries, self.work = [], 0, 0
        current_shape = list(source.shape)
        largest = source.size
        for axis in range(len(source.axes)):
            rows = target.shape[axis]
            current_shape[axis] = rows
            count = math.prod(current_shape)
            needed = rows*source.orders[axis]
            if count > max_work_nodes or self.entries+needed > max_entries:
                raise MemoryError("Separable refinement budget exceeded before allocation")
            destination_nodes, _ = lobatto_rule(target.orders[axis])
            z = ((2*np.arange(target.cells[axis])[:, None]+1+destination_nodes)
                 /target.cells[axis]-1).ravel()
            local = (z+1)*source.cells[axis]/2
            # Preserve each source polynomial's one-sided endpoint values:
            # duplicated Lobatto nodes on a discontinuous cell boundary must
            # not both be reassigned to the cell on the right.
            ratio = target.cells[axis]//source.cells[axis]
            cell = np.repeat(np.arange(target.cells[axis])//ratio, target.orders[axis])
            source_nodes, bary = lobatto_rule(source.orders[axis])
            weights = cardinal(2*(local-cell)-1, source_nodes, bary)
            columns = cell[:, None]*source.orders[axis]+np.arange(source.orders[axis])
            matrix = csr_matrix((weights.ravel(), columns.ravel(),
                                 np.arange(rows+1)*source.orders[axis]),
                                shape=(rows, source.shape[axis]))
            self.stages.append(matrix)
            self.entries += matrix.nnz
            self.work += matrix.nnz*(count//rows)
            largest = max(largest, count)
        self.array_bytes = sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in self.stages)
        self.report = {
            "source_nodes": source.size, "target_nodes": target.size,
            "largest_intermediate_nodes": largest, "stored_entries": self.entries,
            "multiply_add_pairs_per_application": self.work,
            "same_piecewise_polynomial_in_exact_arithmetic": True,
        }

    def __matmul__(self, values):
        values = np.asarray(values)
        if values.shape != (self.source.size,):
            raise ValueError("One value per source tensor node required")
        result = values.reshape(self.source.shape)
        for axis, matrix in enumerate(self.stages):
            front = np.moveaxis(result, axis, 0)
            shape = front.shape
            front = (matrix @ front.reshape(shape[0], -1)).reshape((matrix.shape[0],)+shape[1:])
            result = np.moveaxis(front, 0, axis)
        return result.ravel()

    def adjoint(self, values):
        values = np.asarray(values)
        if values.shape != (self.target.size,):
            raise ValueError("One value per target tensor node required")
        result = values.reshape(self.target.shape)
        for axis in reversed(range(len(self.stages))):
            matrix = self.stages[axis]
            front = np.moveaxis(result, axis, 0)
            shape = front.shape
            front = (matrix.conj().T @ front.reshape(shape[0], -1)).reshape(
                (matrix.shape[1],)+shape[1:])
            result = np.moveaxis(front, 0, axis)
        return result.ravel()
