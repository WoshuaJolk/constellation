"""Fast tensor Chebyshev moments using reflection, a NUFFT and DCTs.

This is an established Fourier/Chebyshev identity, not a Kepler approximation.
FINUFFT's requested epsilon is a numerical tolerance, not an interval proof.
"""

from itertools import product

import numpy as np
from scipy.fft import dct


class CosineMoments:
    def __init__(self, points, orders, *, eps=1e-13):
        points = np.asarray(points, float)
        if (points.ndim != 2 or not 1 <= points.shape[1] <= 3
                or len(orders) != points.shape[1] or not np.all(np.isfinite(points))
                or np.any(abs(points) > 1+8*np.finfo(float).eps)):
            raise ValueError("Normalized 1D, 2D or 3D cosine-moment points required")
        self.orders, self.eps = tuple(orders), eps
        self.count, self.dimension = points.shape
        self.signs = np.array(list(product([-1., 1.], repeat=self.dimension)))
        theta = np.arccos(np.clip(points, -1, 1))
        reflected = (self.signs[:, None, :]*theta[None]).reshape(-1, self.dimension)
        self.coordinates = [np.ascontiguousarray(reflected[:, i]) for i in range(self.dimension)]
        self.modes = tuple(2*p-1 for p in self.orders)
        self.positive = tuple(slice(p-1, None) for p in self.orders)

    def forward(self, values):
        import finufft
        values = np.asarray(values, complex)
        if values.shape != (self.count,):
            raise ValueError("One weight for each cosine-moment point required")
        reflected = np.ascontiguousarray(np.tile(values, len(self.signs))/len(self.signs))
        transform = getattr(finufft, f"nufft{self.dimension}d1")
        modes = transform(*self.coordinates, reflected, n_modes=self.modes,
                          eps=self.eps, isign=1, nthreads=1)
        result = modes[self.positive].copy()
        for axis, p in enumerate(self.orders):
            result = dct(result, type=3, axis=axis)/p
        return result

    def adjoint(self, values):
        import finufft
        result = np.asarray(values, complex)
        if result.shape != self.orders:
            raise ValueError("Tensor of cosine-moment coefficients required")
        for axis in range(self.dimension-1, -1, -1):
            result = dct(result, type=2, axis=axis)/self.orders[axis]
            selection = [slice(None)]*self.dimension
            selection[axis] = 0
            result[tuple(selection)] *= .5
        modes = np.zeros(self.modes, complex)
        modes[self.positive] = result
        transform = getattr(finufft, f"nufft{self.dimension}d2")
        reflected = transform(*self.coordinates, modes, eps=self.eps, isign=-1, nthreads=1)
        return reflected.reshape(len(self.signs), self.count).sum(axis=0)/len(self.signs)


class DenseMoments:
    def __init__(self, points, rules, orders):
        # Deferred import avoids a module import cycle.
        from .source_butterfly import cardinal
        factors = [cardinal(points[:, i], *rule) for i, rule in enumerate(rules)]
        result = factors[0]
        for factor in factors[1:]:
            result = (result[:, :, None]*factor[:, None, :]).reshape(len(points), -1)
        self.matrix, self.orders = result, tuple(orders)

    def forward(self, values):
        return (self.matrix.T @ values).reshape(self.orders)

    def adjoint(self, values):
        return self.matrix.conj() @ values.ravel()
