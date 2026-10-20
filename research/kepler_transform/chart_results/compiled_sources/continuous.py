"""Validated continuous score atlas for a six-dimensional initial-state box.

Each leaf encloses the real Fourier response at EVERY initial state in that
box. It is a piecewise-constant representation with a rigorous local error,
not interpolation of sampled orbits. This conservative construction is used to
test whether continuous coverage is computationally affordable. It does not
certify the separate sampled butterfly.
"""

from fractions import Fraction
from time import perf_counter

import numpy as np
from flint import arb, ctx

from research.detection_limit.interval_orbits import apparent, lower_float, upper_float, MU, C_AU_DAY


def score_interval(state, times, observers, normal, east, north, visits, frequencies,
                   coefficients, *, light_time_max=1.):
    """Exact represented model coefficients; validated ODE, light time and phase."""
    total = arb(0)
    for visit in np.unique(visits):
        position, _ = apparent(state, float(times[visit]), observers[visit],
                               light_time_max=light_time_max)
        direction = [value-arb(float(observer)) for value, observer in zip(position, observers[visit])]
        dot = lambda axis: sum((x*arb(float(a)) for x, a in zip(direction, axis)), arb(0))
        denominator = dot(normal[visit])
        if not denominator > 0:
            raise ValueError("Cell cannot be certified inside the tangent horizon")
        x, y = dot(east[visit])/denominator, dot(north[visit])/denominator
        for mode in np.flatnonzero(visits == visit):
            phase = arb(float(frequencies[mode, 0]))*x+arb(float(frequencies[mode, 1]))*y
            sine, cosine = phase.sin_cos()
            coefficient = complex(coefficients[mode])
            total += arb(coefficient.real)*cosine-arb(coefficient.imag)*sine
    if not total.is_finite():
        raise ValueError("Nonfinite response enclosure")
    return total


def build_atlas(center, half_widths, kernel, coefficients, *, tolerance=1e-6,
                max_cells=64, light_time_max=1., split_weights=None):
    """Refine the largest error, retaining unresolved cells when the budget ends.

    Bounds use exact dyadic normalized coordinates. Failed enclosures are
    unbounded unresolved leaves, never discarded states or false exclusions.
    """
    center, widths = np.asarray(center, float), np.asarray(half_widths, float)
    coefficients = np.asarray(coefficients, complex)
    if (center.shape != (6,) or widths.shape != (6,) or np.any(widths <= 0)
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(widths))
            or coefficients.shape != (len(kernel.visits),) or not np.all(np.isfinite(coefficients))
            or not np.isfinite(tolerance) or tolerance <= 0
            or not np.isfinite(light_time_max) or light_time_max <= 0
            or isinstance(max_cells, bool) or not isinstance(max_cells, int) or not 1 <= max_cells <= 100000):
        raise ValueError("Finite 6D box, compatible coefficients, positive tolerance and valid budget required")
    weights = np.ones(6) if split_weights is None else np.asarray(split_weights, float)
    if weights.shape != (6,) or np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("Six positive finite split weights required")
    started, evaluations = perf_counter(), 0

    def enclose(bounds, path):
        nonlocal evaluations
        evaluations += 1
        state = [arb(float(c))+arb(float(w))*arb(float(lo)).union(arb(float(hi)))
                 for c, w, (lo, hi) in zip(center, widths, bounds)]
        leaf = {"path": path, "normalized_bounds": bounds.tolist()}
        try:
            score = score_interval(state, kernel.times, kernel.observers,
                                   kernel.normal, kernel.east, kernel.north,
                                   kernel.visits, kernel.frequencies, coefficients,
                                   light_time_max=light_time_max)
            value = float(score.mid())
            error = upper_float(abs(score-arb(value)))
            leaf.update(value=value, absolute_error_bound=error,
                        lower=lower_float(score), upper=upper_float(score),
                        status="resolved" if error <= tolerance else "unresolved")
        except ValueError as failure:
            leaf.update(value=None, absolute_error_bound=None, lower=None, upper=None,
                        status="unresolved", reason=str(failure))
        return leaf

    with ctx.workprec(128):
        leaves = [enclose(np.tile([-1., 1.], (6, 1)), "")]
        while len(leaves) < max_cells:
            unresolved = [(float("inf") if leaf["absolute_error_bound"] is None
                           else leaf["absolute_error_bound"], index)
                          for index, leaf in enumerate(leaves) if leaf["status"] != "resolved"]
            if not unresolved:
                break
            _, index = max(unresolved)
            parent = leaves.pop(index)
            bounds = np.asarray(parent["normalized_bounds"])
            axis = int(np.argmax((bounds[:, 1]-bounds[:, 0])*weights))
            middle = (bounds[axis, 0]+bounds[axis, 1])/2
            if middle in bounds[axis]:
                # A machine-coordinate resolution failure also remains visible.
                parent["reason"] = "Normalized subdivision reached binary64 resolution"
                leaves.append(parent)
                break
            left, right = bounds.copy(), bounds.copy()
            left[axis, 1], right[axis, 0] = middle, middle
            leaves.extend([enclose(left, parent["path"]+f"{axis}L"),
                           enclose(right, parent["path"]+f"{axis}R")])
    complete = all(leaf["status"] == "resolved" for leaf in leaves)
    return {"method": "validated piecewise-constant continuous Fourier-score atlas",
            "center": center.tolist(), "half_widths": widths.tolist(),
            "requested_absolute_error": tolerance, "max_cells": max_cells,
            "leaves": leaves, "all_cells_resolved": complete,
            "status": "certified_under_declared_model" if complete else "budget_or_enclosure_failure",
            "cell_evaluations": evaluations, "construction_seconds": perf_counter()-started,
            "precision_bits": 128, "mu_au3_day2": MU, "c_au_day": C_AU_DAY,
            "full_objective_complete": False,
            "scope": "real part of supplied Fourier fields; fixed gnomonic bases/observer; no arbitrary instrument likelihood"}


def verify_partition(report):
    """Independent exact-volume and disjoint-interior check of the leaf cover."""
    boxes = [[[(Fraction(x)) for x in interval] for interval in leaf["normalized_bounds"]]
             for leaf in report["leaves"]]
    volume = Fraction(0)
    for i, box in enumerate(boxes):
        v = Fraction(1)
        if len(box) != 6:
            raise ValueError("Invalid partition dimension")
        for lo, hi in box:
            if not -1 <= lo < hi <= 1:
                raise ValueError("Leaf outside root or degenerate")
            v *= hi-lo
        volume += v
        for other in boxes[:i]:
            if all(max(a[0], b[0]) < min(a[1], b[1]) for a, b in zip(box, other)):
                raise ValueError("Leaf interiors overlap")
    if volume != 64:
        raise ValueError("Partition does not cover the six-dimensional root")
    return True
