"""Adaptive, explicit domain coverage for the observer-chart construction.

Certificates cover continuous geometry. Fast Fourier arithmetic is NOT
interval-certified. Budget failures keep their full cells, and strict
evaluation refuses queries in unresolved cells.
"""

import copy
from itertools import product
from time import perf_counter

import numpy as np
from flint import arb, ctx

from research.detection_limit import interval_orbits as flow
from .chart_atlas import ChartAtlas, scalar_position
from .continuous import verify_partition
from .factorization import ARCSEC
from .local_transform import POWERS, monomials


OUTER_POWERS = list(product(range(3), repeat=3))


def polynomial_positions(atlas, normalized, chunk=4096):
    """Full 6D retained polynomial, for arbitrary (unstructured) queries."""
    normalized = np.asarray(normalized, float)
    if (normalized.ndim != 2 or normalized.shape[1] != 6 or not np.all(np.isfinite(normalized))
            or np.any(abs(normalized) > 1)):
        raise ValueError("Finite normalized six-dimensional queries in the atlas required")
    powers = getattr(atlas, "outer_powers", OUTER_POWERS)
    coefficients = np.array([[atlas.coefficients[a+b] for b in POWERS] for a in powers])
    result = np.empty((len(normalized), len(atlas.geometry.times), 2))
    for start in range(0, len(normalized), chunk):
        points = normalized[start:start+chunk]
        outer = monomials(points[:, :3], powers)
        inner = monomials(points[:, 3:])
        features = (outer[:, :, None]*inner[:, None, :]).reshape(len(points), -1)
        result[start:start+chunk] = (features @ coefficients.reshape(len(powers)*len(POWERS), -1)).reshape(len(points), -1, 2)
    return result


def positions_nufft(geometry, positions, data, eps=1e-12):
    """Independent per-frame accumulation using already calculated geometry."""
    import finufft
    result = np.zeros(len(positions), complex)
    for visit in np.unique(geometry.visits):
        selected = geometry.visits == visit
        frequency = geometry.frequencies[selected]*ARCSEC
        destination = positions[:, visit]/ARCSEC
        result += finufft.nufft2d3(
            np.ascontiguousarray(frequency[:, 0]), np.ascontiguousarray(frequency[:, 1]),
            np.ascontiguousarray(data[selected]), np.ascontiguousarray(destination[:, 0]),
            np.ascontiguousarray(destination[:, 1]), eps=eps, isign=1, nthreads=1)
    return result


def physical_kernel(geometry, states):
    """Fresh exact geometry cache, preserving the supplied exposure axes."""
    kernel = copy.copy(geometry)
    kernel.states = np.asarray(states, float)
    kernel.positions = np.full((len(states), len(geometry.times), 2), np.nan)
    kernel.geometry_pairs, kernel.entries, kernel.geometry_seconds = 0, 0, 0.
    return kernel


def enclosing_cell(center, widths, bounds):
    """Outward physical boxes cover the exact dyadic subdivision of the root."""
    c, w = [], []
    with ctx.workprec(128):
        for origin, radius, (lo, hi) in zip(center, widths, bounds):
            left = arb(float(origin))+arb(float(radius))*arb(float(lo))
            right = arb(float(origin))+arb(float(radius))*arb(float(hi))
            midpoint = float(((left+right)/2).mid())
            half = flow.upper_float(abs(left-arb(midpoint)).union(abs(right-arb(midpoint))))
            c.append(midpoint)
            w.append(half)
    return np.array(c), np.array(w)


class ChartCover:
    def __init__(self, chart, center, widths, geometry, *, position_tolerance,
                 max_leaves=64, max_seconds=600., progress=None, outer_degree=2, phase_tolerance=None,
                 translated_certificate=False):
        started = perf_counter()
        self.chart, self.geometry = chart, geometry
        self.center, self.widths = np.asarray(center, float), np.asarray(widths, float)
        if (self.center.shape != (6,) or self.widths.shape != (6,)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.widths))
                or np.any(self.widths <= 0) or self.center[0] <= self.widths[0]
                or not np.isfinite(position_tolerance) or position_tolerance <= 0
                or isinstance(max_leaves, bool) or not isinstance(max_leaves, int) or not 1 <= max_leaves <= 4096
                or not np.isfinite(max_seconds) or max_seconds <= 0
                or isinstance(outer_degree, bool) or outer_degree not in (2, 4)
                or not isinstance(translated_certificate, bool) or (phase_tolerance is not None and
                    (not np.isfinite(phase_tolerance) or not 0 < phase_tolerance < 1))):
            raise ValueError("Finite positive-distance box, tolerance and bounded construction budget required")
        self.position_tolerance = position_tolerance
        self.max_leaves, self.max_seconds, self.outer_degree = max_leaves, max_seconds, outer_degree
        self.phase_tolerance, self.phase_orders = phase_tolerance, {}
        self.translated_certificate = translated_certificate
        atlas_type = ChartAtlas
        if outer_degree == 4:
            from .quartic_atlas import QuarticChartAtlas
            atlas_type = QuarticChartAtlas
        self.leaves, self.history = [], []
        self.atlases, self.certificates = {}, {}

        def evaluate(bounds, path):
            c, w = enclosing_cell(self.center, self.widths, bounds)
            record = {"path": path, "normalized_bounds": bounds.tolist(),
                      "center": c.tolist(), "widths": w.tolist(), "status": "unresolved"}
            cell_start = perf_counter()
            try:
                atlas = atlas_type(chart, c, w, geometry, translated=translated_certificate)
                certificate = atlas.certify(max_position_error=position_tolerance)
                direction = np.max(certificate["directional_remainder_radians"], axis=(0, 2))
                record["split_axis"] = int(np.argmax(direction))
                record["screen_directional_max_radians"] = direction.tolist()
                if certificate["status"] == "certified":
                    error = float(np.max(certificate["position_error_radians"]))
                    record["position_error_bound_radians"] = error
                    if error <= position_tolerance:
                        record["status"] = "resolved"
                        record["atlas_sha256"] = atlas.digest
                        self.atlases[path], self.certificates[path] = atlas, certificate
                    else:
                        record["reason"] = "Nodal/coefficient remainder exceeds requested tolerance"
                else:
                    record["reason"] = "Variational screen cannot meet requested tolerance"
                record["nodal_orbit_time_enclosures"] = certificate["nodal_orbit_time_enclosures"]
                record["variational_orbit_time_enclosures"] = certificate["variational_orbit_time_enclosures"]
                record["ordinary_geometry_state_time_pairs"] = atlas.report["geometry_state_time_pairs"]
            except ValueError as failure:
                record["reason"] = str(failure)
                record["split_axis"] = int(np.argmax(bounds[:2, 1]-bounds[:2, 0]))
            record["construction_seconds"] = perf_counter()-cell_start
            self.history.append(record.copy())
            return record

        self.leaves.append(evaluate(np.tile([-1., 1.], (6, 1)), ""))
        while len(self.leaves) < max_leaves and perf_counter()-started < max_seconds:
            unresolved = [i for i, leaf in enumerate(self.leaves) if leaf["status"] != "resolved"]
            if not unresolved:
                break
            # Largest remaining normalized volume prevents starving a branch.
            index = max(unresolved, key=lambda i: np.prod(np.diff(self.leaves[i]["normalized_bounds"], axis=1)))
            parent = self.leaves[index]
            bounds, axis = np.asarray(parent["normalized_bounds"]), parent["split_axis"]
            middle = (bounds[axis, 0]+bounds[axis, 1])/2
            if middle in bounds[axis] or len(parent["path"]) >= 80:
                parent["reason"] = "Subdivision resolution/depth limit"
                break
            left, right = bounds.copy(), bounds.copy()
            left[axis, 1], right[axis, 0] = middle, middle
            children = [evaluate(left, parent["path"]+f"{axis}L"),
                        evaluate(right, parent["path"]+f"{axis}R")]
            self.leaves[index:index+1] = children
            if progress is not None:
                progress(self.report(perf_counter()-started))
        self.construction_seconds = perf_counter()-started
        verify_partition(self.report())

    def report(self, elapsed=None):
        return {"method": "adaptive observer-chart geometry cover",
                "chart": self.chart.to_dict(), "center": self.center.tolist(), "widths": self.widths.tolist(),
                "position_tolerance_radians": self.position_tolerance,
                "outer_degree": self.outer_degree, "max_leaves": self.max_leaves,
                "max_seconds": self.max_seconds,
                "phase_tolerance": self.phase_tolerance,
                "translated_certificate": self.translated_certificate,
                "construction_seconds": getattr(self, "construction_seconds", elapsed),
                "leaves": self.leaves, "history": self.history,
                "cell_evaluations": len(self.history),
                "all_cells_resolved": all(x["status"] == "resolved" for x in self.leaves),
                "continuous_geometry_cover": True,
                "all_arithmetic_certified": False, "full_objective_complete": False}

    def apply(self, data, parameters, *, allow_direct_fallback=False, route="auto"):
        """Evaluate every query or fail. No cell/query is silently discarded.

        Structured queries share 3D transforms across time. Arbitrary queries
        use the same certified 6D polynomial with per-frame NUFFTs; this is a
        cheaper geometry engine, not a global six-dimensional fast transform.
        Direct fallback is explicit and receives no continuous-error certificate.
        """
        started = perf_counter()
        data, parameters = np.asarray(data, complex), np.asarray(parameters, float)
        if (data.shape != (len(self.geometry.visits),) or not np.all(np.isfinite(data))
                or parameters.ndim != 2 or parameters.shape[1] != 6 or not len(parameters)
                or not np.all(np.isfinite(parameters)) or route not in ("auto", "polynomial", "slices")):
            raise ValueError("Finite compatible fields, query parameters and valid route required")
        normalized = (parameters-self.center)/self.widths
        if np.any(abs(normalized) > 1):
            raise ValueError("Query lies outside the declared root domain")
        output, assigned = np.empty(len(parameters), complex), np.zeros(len(parameters), bool)
        counts = {"slice_queries": 0, "polynomial_queries": 0, "direct_queries": 0,
                  "slice_transforms": 0, "resource_fallbacks": 0, "phase_budget_fallbacks": 0,
                  "phase_orders_used": {}}
        for leaf in self.leaves:
            bounds = np.array(leaf["normalized_bounds"])
            selected = np.all((normalized >= bounds[:, 0]) &
                              ((normalized < bounds[:, 1]) | (bounds[:, 1] == 1)), axis=1)
            rows = np.flatnonzero(selected)
            if not len(rows):
                continue
            if np.any(assigned[rows]):
                raise RuntimeError("Internal cover assigned a query twice")
            assigned[rows] = True
            if leaf["status"] != "resolved":
                if not allow_direct_fallback:
                    raise ValueError(f"Query in unresolved cell {leaf['path']!r}: {leaf['reason']}")
                output[rows] = physical_kernel(self.geometry, self.chart.states(parameters[rows])).nufft(data)
                counts["direct_queries"] += len(rows)
                continue
            atlas = self.atlases[leaf["path"]]
            local = (parameters[rows]-atlas.center)/atlas.widths
            if np.any(abs(local) > 1):
                raise ValueError("Query conversion escaped the outward cell enclosure")
            outers, inverse, population = np.unique(local[:, :3], axis=0, return_inverse=True, return_counts=True)
            sliced = route == "slices" or (route == "auto" and np.min(population) >= 512)
            order = 1
            if sliced and self.phase_tolerance is not None:
                from .phase_budget import choose_order
                key = (atlas.digest, self.phase_tolerance)
                if key not in self.phase_orders:
                    self.phase_orders[key] = choose_order(atlas, self.phase_tolerance)
                order = self.phase_orders[key]["selected_order"]
                if order is None:
                    sliced = False
                    order = 1
                    counts["phase_budget_fallbacks"] += 1
            grouped = sliced or (len(outers) <= 32 and len(outers)*512 <= len(rows))
            if grouped:
                for index, outer in enumerate(outers):
                    subset = np.flatnonzero(inverse == index)
                    plan = atlas.slice(outer, correction_order=order)
                    if not sliced:
                        from .comparison_experiments import polynomial_nufft
                        output[rows[subset]] = polynomial_nufft(plan, data, local[subset, 3:])
                        counts["polynomial_queries"] += len(subset)
                        continue
                    try:
                        output[rows[subset]] = plan.apply(data, local[subset, 3:])
                        counts["slice_queries"] += len(subset)
                        counts["slice_transforms"] += 1
                        counts["phase_orders_used"][str(order)] = counts["phase_orders_used"].get(str(order), 0)+len(subset)
                    except ValueError as error:
                        if "phase-volume budget" not in str(error):
                            raise
                        from .comparison_experiments import polynomial_nufft
                        output[rows[subset]] = polynomial_nufft(plan, data, local[subset, 3:])
                        counts["polynomial_queries"] += len(subset)
                        counts["resource_fallbacks"] += 1
            else:
                output[rows] = positions_nufft(self.geometry, polynomial_positions(atlas, local), data)
                counts["polynomial_queries"] += len(rows)
        if not np.all(assigned):
            raise RuntimeError("Internal cover left queries unassigned")
        return output, {**counts, "application_seconds": perf_counter()-started,
                        "queries": len(parameters), "all_queries_evaluated": True,
                        "full_score_error_certified": False}


def audit_scores(chart, geometry, parameters, data, approximations):
    """Slow, independent, fully interval score audit for selected queries.

    Integrates physical dynamics and evaluates all Fourier phases in Arb.
    Includes arithmetic errors, but costs O(selected queries * modes).
    This verifier is not counted as a fast whole-domain score certificate.
    """
    parameters, data = np.asarray(parameters, float), np.asarray(data, complex)
    approximations = np.asarray(approximations, complex)
    if (parameters.ndim != 2 or parameters.shape[1] != 6 or not len(parameters)
            or data.shape != (len(geometry.visits),) or approximations.shape != (len(parameters),)
            or not all(np.all(np.isfinite(x)) for x in [parameters, data, approximations])
            or np.any(parameters[:, 0] <= 0)):
        raise ValueError("Compatible finite query, Fourier data and approximation arrays required")
    results = []
    with ctx.workprec(128):
        for parameters_i, value in zip(parameters, approximations):
            positions = [scalar_position(chart, [arb(float(x)) for x in parameters_i], float(time),
                geometry.observers[j], geometry.normal[j], geometry.east[j], geometry.north[j])
                for j, time in enumerate(geometry.times)]
            real, imaginary = arb(0), arb(0)
            for visit, frequency, datum in zip(geometry.visits, geometry.frequencies, data):
                phase = sum((arb(float(k))*x for k, x in zip(frequency, positions[visit])), arb(0))
                sine, cosine = phase.sin_cos()
                real += arb(float(datum.real))*cosine-arb(float(datum.imag))*sine
                imaginary += arb(float(datum.real))*sine+arb(float(datum.imag))*cosine
            error = flow._norm([real-arb(float(value.real)), imaginary-arb(float(value.imag))])
            results.append({"real_interval": [flow.lower_float(real), flow.upper_float(real)],
                            "imaginary_interval": [flow.lower_float(imaginary), flow.upper_float(imaginary)],
                            "absolute_error_upper": flow.upper_float(error)})
    return results
