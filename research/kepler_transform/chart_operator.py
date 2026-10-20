"""Finite-query forward/adjoint pair and regularized inverse of a chart cover.

This pair uses the exact exponential of the retained position polynomial.
It does not mix a Taylor-truncated forward map with a different adjoint.
The inverse recovers Fourier-field coefficients subject to the operator's
nullspace, not arbitrary original images or unique physical orbital states.
"""

from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

from .chart_cover import polynomial_positions, positions_nufft
from .factorization import ARCSEC


class ChartOperator:
    def __init__(self, cover, parameters, *, eps=1e-12):
        started = perf_counter()
        self.geometry, self.eps = cover.geometry, eps
        parameters = np.asarray(parameters, float)
        if (parameters.ndim != 2 or parameters.shape[1] != 6 or not len(parameters)
                or not np.all(np.isfinite(parameters)) or not np.isfinite(eps) or not 1e-14 <= eps < 1e-3):
            raise ValueError("Finite (N,6) queries and valid numerical epsilon required")
        normalized = (parameters-cover.center)/cover.widths
        if np.any(abs(normalized) > 1):
            raise ValueError("Query outside declared cover")
        self.positions = np.empty((len(parameters), len(self.geometry.times), 2))
        assigned = np.zeros(len(parameters), bool)
        for leaf in cover.leaves:
            bounds = np.asarray(leaf["normalized_bounds"])
            selected = np.all((normalized >= bounds[:, 0]) &
                              ((normalized < bounds[:, 1]) | (bounds[:, 1] == 1)), axis=1)
            rows = np.flatnonzero(selected)
            if not len(rows):
                continue
            if leaf["status"] != "resolved":
                raise ValueError("Finite operator query in unresolved cell")
            if np.any(assigned[rows]):
                raise RuntimeError("Query assigned more than once")
            atlas = cover.atlases[leaf["path"]]
            self.positions[rows] = polynomial_positions(atlas, (parameters[rows]-atlas.center)/atlas.widths)
            assigned[rows] = True
        if not np.all(assigned):
            raise RuntimeError("Query was not assigned")
        self.shape = (len(parameters), len(self.geometry.visits))
        self.report = {"construction_seconds": perf_counter()-started,
                       "cached_position_bytes": self.positions.nbytes,
                       "queries_in_certified_geometry_cells": True,
                       "numerical_transform_error_certified": False}

    def apply(self, coefficients):
        coefficients = np.asarray(coefficients, complex)
        if coefficients.shape != (self.shape[1],) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Finite compatible coefficients required")
        return positions_nufft(self.geometry, self.positions, coefficients, eps=self.eps)

    def adjoint(self, values):
        import finufft
        values = np.asarray(values, complex)
        if values.shape != (self.shape[0],) or not np.all(np.isfinite(values)):
            raise ValueError("Finite compatible query values required")
        result = np.zeros(self.shape[1], complex)
        for visit in np.unique(self.geometry.visits):
            selected = self.geometry.visits == visit
            source = self.positions[:, visit]/ARCSEC
            destination = self.geometry.frequencies[selected]*ARCSEC
            result[selected] = finufft.nufft2d3(
                np.ascontiguousarray(source[:, 0]), np.ascontiguousarray(source[:, 1]),
                np.ascontiguousarray(values), np.ascontiguousarray(destination[:, 0]),
                np.ascontiguousarray(destination[:, 1]), eps=self.eps, isign=-1, nthreads=1)
        return result

    def inverse(self, scores, *, damping=0., tolerance=1e-9, max_iterations=200):
        scores = np.asarray(scores, complex)
        if (scores.shape != (self.shape[0],) or not np.all(np.isfinite(scores))
                or not np.isfinite(damping) or damping < 0
                or not np.isfinite(tolerance) or not 0 < tolerance < 1
                or isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1):
            raise ValueError("Compatible finite scores and valid inverse controls required")
        operator = LinearOperator(self.shape, dtype=complex, matvec=self.apply, rmatvec=self.adjoint)
        solution = lsqr(operator, scores, damp=damping, atol=tolerance, btol=tolerance, iter_lim=max_iterations)
        return solution[0], {"method": "regularized LSQR on finite chart-polynomial Fourier operator",
                             "damping": damping, "iterations": solution[2], "stop_code": solution[1],
                             "residual_norm": solution[3], "condition_estimate": solution[6],
                             "iteration_limit_reached": solution[1] == 7,
                             "unique_image_reconstruction_claimed": False}
