"""Locally sufficient Kepler grids: interval geometry, error and integer costs.

Only exact-arithmetic Fourier-field interpolation is certified. This module
plans grids before allocation; numerical flow/application error is separate.
"""
import math
from time import perf_counter

import numpy as np
from flint import acb, arb, arb_mat

from research.detection_limit.interval_orbits import upper_float
from .cascade_domain import _state_box, contraction_weights
from .reception_flow import _union, step
from .variational_flow import flow_jacobian, peano_increment, rhs_jacobian

AXES = (0, 1, 2, 3, 4, 5)
PERMUTATION = (3, 4, 5, 2, 1, 0)


def _upper(values):
    return [upper_float(x) for x in values]


def _matrix(values):
    return [[upper_float(x) for x in row] for row in values]


def _identity():
    return arb_mat([[int(i == j) for j in AXES] for i in AXES])


def _partition(values, pieces, part):
    result = list(values)
    source = values[0].real if isinstance(values[0], acb) else values[0]
    real = source.mid()+source.rad()*arb(-1+(2*part+1)/pieces, 1/pieces)
    result[0] = acb(real, values[0].imag) if isinstance(values[0], acb) else real
    return result


def _derivative(observer, center, widths, radii, imaginary, start, stop,
                work, max_step, max_range_pieces):
    values = _state_box(center, widths, radii, imaginary)
    real = values[0].real if imaginary is not None else values[0]
    if not real > 0:
        raise ValueError("Complex neighborhood crosses inverse range zero")
    pieces = 1
    while True:
        result = None
        try:
            for part in range(pieces):
                work["flow_calls"] += 1
                _, local, cost = flow_jacobian(_partition(values, pieces, part),
                                               start, stop, observer, widths,
                                               max_step=max_step)
                work["completed_flow_steps"] += cost["state_steps"]
                if result is None:
                    result = local
                else:
                    for i in AXES:
                        for j in AXES:
                            result[i, j] = _union(result[i, j], local[i, j])
            work["largest_range_partition"] = max(work["largest_range_partition"], pieces)
            return result
        except ValueError:
            work["refused_flow_calls"] += 1
            if 2*pieces > max_range_pieces:
                raise
            pieces *= 2


def nonnegative_inverse_bound(matrix):
    """Verify rho(E)<1, then enclose the nonnegative Neumann sum (I-E)^-1."""
    if len(matrix) != 6 or any(len(row) != 6 for row in matrix):
        raise ValueError("Six by six majorant required")
    if any(not arb(x).is_finite() or not arb(x) >= 0 for row in matrix for x in row):
        raise ValueError("Finite nonnegative majorant required")
    weights, norm = contraction_weights(matrix)
    if not norm < 1:
        raise ValueError("Derivative majorant is not a verified contraction")
    inverse = (_identity()-arb_mat(matrix)).inv()
    bound = [[inverse[i, j].upper() for j in AXES] for i in AXES]
    if any(not x.is_finite() or not x >= 0 for row in bound for x in row):
        raise ValueError("Neumann inverse enclosure failed")
    return bound, {"weights": weights.tolist(), "contraction_bound": upper_float(norm)}


def axis_neighborhood(proof, observer, axis, ellipse, *, max_step=.03125,
                      max_iterations=12, max_range_pieces=16):
    """Certify all hybrid inverses with ONE hybrid coordinate complex.

    Other coordinates range over their entire real boxes. Unlike an isotropic
    polyellipse, this allows large analytic radii on weakly coupled axes.
    """
    if (axis not in AXES or not np.isfinite(ellipse) or ellipse <= 1
            or not np.isfinite(max_step) or max_step <= 0
            or type(max_iterations) is not int or max_iterations < 1
            or type(max_range_pieces) is not int or max_range_pieces < 1):
        raise ValueError("Valid axis and ellipse > 1 required")
    if not proof["all_coordinate_hybrids_certified"]:
        raise ValueError("A real inverse certificate is required")
    begin = perf_counter()
    c, w = proof["parent_center"], proof["parent_widths"]
    b = [arb(cw)/ww for cw, ww in zip(proof["child_widths"], w)]
    r = arb(float(ellipse))
    real_base, imag_base = list(b), [arb(0) for _ in AXES]
    real_base[axis] *= (r+1/r)/2
    imag_base[axis] = b[axis]*(r-1/r)/2
    radii = _upper([real_base[i]+1.1*(arb(proof["normalized_enclosing_radii"][i])-b[i])+arb("1e-8")
                    for i in AXES])
    imaginary = _upper([1.1*x+arb("1e-8") for x in imag_base])
    work = {"flow_calls": 0, "refused_flow_calls": 0, "completed_flow_steps": 0,
            "largest_range_partition": 1}
    for iteration in range(max_iterations):
        real_d = _derivative(observer, c, w, radii, None, proof["start"], proof["stop"],
                             work, max_step, max_range_pieces)
        delta = [arb(proof["normalized_center_error_bounds"][i])+
                 sum(abs(real_d[i, j]).upper()*radii[j] for j in AXES) for i in AXES]
        cd = _derivative(observer, c, w, radii, imaginary, proof["start"], proof["stop"],
                         work, max_step, max_range_pieces)
        er = [[abs(cd[i, j].real).upper() for j in AXES] for i in AXES]
        ei = [[abs(cd[i, j].imag).upper() for j in AXES] for i in AXES]
        required_real = [real_base[i]+delta[i]+sum(ei[i][j]*imaginary[j] for j in AXES) for i in AXES]
        required_imag = [imag_base[i]+sum(er[i][j]*imaginary[j] for j in AXES) for i in AXES]
        # |Dd| <= |Re Dd|+|Im Dd|; this also majorizes the real 12D block.
        E = [[er[i][j]+ei[i][j] for j in AXES] for i in AXES]
        S, norm = nonnegative_inverse_bound(E)
        if (all(arb(radii[i]) > required_real[i] for i in AXES)
                and all(arb(imaginary[i]) > required_imag[i] for i in AXES)):
            break
        radii = _upper([real_base[i]+1.1*(required_real[i]-real_base[i])+arb("1e-8") for i in AXES])
        imaginary = _upper([imag_base[i]+1.1*(required_imag[i]-imag_base[i])+arb("1e-8") for i in AXES])
    else:
        raise ValueError("Single-axis analytic neighborhood needs subdivision")
    return {"axis": axis, "ellipse": float(ellipse), "hybrid_halfwidth": upper_float(b[axis]),
            "enclosing_real_radii": radii, "enclosing_imaginary_radii": imaginary,
            "increment_majorant": _matrix(E), "inverse_majorant": _matrix(S),
            "iterations": iteration+1, "work": work, **norm,
            "seconds": perf_counter()-begin, "exact_inverse_neighborhood_certified": True}


def _time_envelope(values, start, stop, observer, widths, work, max_step):
    """Majorize |W^-1 D R(start,t) W| for EVERY t between start and stop.

    A nonnegative exponential majorant on each Picard tube covers partial as
    well as full steps. Thus cost depends on span/step, not exposure count.
    """
    start, stop = arb(start), arb(stop)
    if start == stop:
        return _identity()
    count = max(1, math.ceil(upper_float(abs(stop-start))/max_step))
    count = 1 << (count-1).bit_length()
    h = (stop-start)/count
    cumulative = _identity()
    for j in range(count):
        work["attempted_steps"] += 1
        values, tube, _ = step(values, start+j*h, h, observer)
        generator = rhs_jacobian(start+j*h+arb(0).union(h), tube, observer, widths)
        positive = arb_mat([[abs(generator[i, k]).upper() for k in AXES] for i in AXES])
        increment, _ = peano_increment(positive, abs(h))
        # Upper endpoints remove interval signs introduced by the series tail.
        local = arb_mat([[((1 if i == k else 0)+increment[i, k]).upper()
                          for k in AXES] for i in AXES])
        cumulative = local*cumulative
    return cumulative


def phase_bound(proof, neighborhood, observer, time_range, frequency_bounds,
                *, max_step=.03125, max_range_pieces=16):
    """Single-axis phase derivative bound for an entire child time interval."""
    begin = perf_counter()
    lo, hi = map(float, time_range)
    K = np.asarray(frequency_bounds, float)
    if (not np.isfinite([lo, hi]).all() or lo > hi or K.shape != (2,)
            or not np.isfinite(K).all() or np.any(K < 0) or max_step <= 0):
        raise ValueError("Finite ordered time interval and nonnegative frequency bounds required")
    w = proof["parent_widths"]
    values = _state_box(proof["parent_center"], w,
                        neighborhood["enclosing_real_radii"],
                        neighborhood["enclosing_imaginary_radii"])
    S = neighborhood["inverse_majorant"]
    axis = neighborhood["axis"]
    work = {"attempted_steps": 0, "refused_partitions": 0}
    pieces = 1
    while True:
        limits = []
        try:
            for part in range(pieces):
                for end in sorted(set((min(lo, proof["start"]), max(hi, proof["start"])))):
                    J = _time_envelope(_partition(values, pieces, part), proof["start"], end,
                                       observer, w, work, max_step)
                    limits.append(sum(arb(float(k))*w[a]*sum(J[a, j]*S[j][axis] for j in AXES)
                                      for k, a in zip(K, (3, 4))).upper())
            break
        except ValueError:
            work["refused_partitions"] += 1
            if pieces*2 > max_range_pieces:
                raise
            pieces *= 2
    return {"phase_derivative_bound": upper_float(max(limits)),
            "time_range": [lo, hi], "frequency_bounds": K.tolist(),
            "all_times_in_interval_covered": True, "range_pieces": pieces,
            "work": work, "seconds": perf_counter()-begin}


def _lebesgue(order):
    return arb(1) if order <= 2 else arb(2*order-2).sqrt()


def axis_error(order, cells, ellipse, phase, halfwidth):
    """Uniform 1D error per unit Fourier coefficient l1, exact arithmetic.

    Subtract the value at the real cell center (interpolation preserves it).
    The derivative estimate bounds the remaining analytic function by
    min(1, L*b*a/n) exp(L*b*s/n). This exploits weak hidden-coordinate effects.
    """
    if phase == 0:
        return arb(0)
    r = arb(float(ellipse))
    a, s = (r+1/r)/2, (r-1/r)/2
    scale = arb(float(phase))*float(halfwidth)/cells
    small = min(arb(1), (scale*a).upper())
    return (1+_lebesgue(order))*2*small*(scale*s).exp()*r**(-order)/(1-1/r)


def _ceil_bound(value):
    # arb -> outward double -> integer. Integers above 2**53 are still rounded
    # upward by upper_float; the final inequalities are independently checked.
    number = upper_float(value)
    if not math.isfinite(number):
        raise ValueError("Grid count exceeds finite planning bounds")
    return max(1, math.ceil(number))


def choose_axis(neighborhood, phase, tolerance, *, multiple=1, max_order=64,
                local_ellipses=(1.2, 1.5, 2., 3., 4., 6., 8., 12., 16., 32., 64.)):
    """Minimize n*p among enumerated ellipse/order choices; verify with Arb."""
    if (not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(multiple, int)
            or multiple < 1 or not isinstance(max_order, int) or not 1 <= max_order <= 64
            or not np.isfinite(phase) or phase < 0):
        raise ValueError("Positive tolerance, nested-cell multiple and valid order/phase required")
    r0, b = arb(neighborhood["ellipse"]), neighborhood["hybrid_halfwidth"]
    a0, s0 = (r0+1/r0)/2, (r0-1/r0)/2
    best, tested = None, 0
    for ellipse in sorted(set(local_ellipses+(float(r0),))):
        if not np.isfinite(ellipse) or ellipse <= 1:
            raise ValueError("All candidate ellipses must exceed one")
        r = arb(float(ellipse))
        a, s = (r+1/r)/2, (r-1/r)/2
        floor = (1 if ellipse <= float(r0) else
                 _ceil_bound(max(((a-1)/(a0-1)).upper(), (s/s0).upper())))
        floor = ((floor+multiple-1)//multiple)*multiple
        for p in range(1, max_order+1):
            tested += 1
            if best is not None and floor*p >= best["axis_nodes"]:
                continue
            def passes(n):
                return axis_error(p, n, ellipse, phase, b) <= arb(float(tolerance))
            low, high = 0, floor//multiple
            if not passes(high*multiple):
                low = high
                cap = None if best is None else (best["axis_nodes"]-1)//(multiple*p)
                while not passes(high*multiple):
                    if cap is not None and high >= cap:
                        break
                    high = 2*high if cap is None else min(2*high, cap)
                    if high > 10**100:
                        raise ValueError("Grid count planning limit exceeded")
                if not passes(high*multiple):
                    continue
            # Do not search below the geometric minimum.
            low = max(low, floor//multiple-1)
            while high-low > 1:
                mid = (low+high)//2
                if passes(mid*multiple):
                    high = mid
                else:
                    low = mid
            n = high*multiple
            assert ellipse <= float(r0) or (arb(n)*(a0-1) >= a-1 and arb(n)*s0 >= s)
            bound = axis_error(p, n, ellipse, phase, b).upper()
            candidate = {"cells": n, "order": p, "ellipse": ellipse,
                         "axis_nodes": n*p, "error_bound": upper_float(bound),
                         "lebesgue_bound": upper_float(_lebesgue(p)),
                         "phase_derivative_bound": phase, "hybrid_halfwidth": b,
                         "neighborhood_ellipse": float(r0), "cell_multiple": multiple}
            if best is None or candidate["axis_nodes"] < best["axis_nodes"]:
                best = candidate
    if best is None:
        raise ValueError("No sufficient axis grid within the candidate budget")
    best["candidates_considered"] = tested
    return best


def grid_plan(axis_choices, tolerance, *, permutation=PERMUTATION):
    """Verify a complete six-stage error budget and return exact integer size."""
    if len(axis_choices) != 6 or sorted(permutation) != list(AXES):
        raise ValueError("Six axis choices and valid permutation required")
    error, amplification = arb(0), arb(1)
    for i in permutation:
        choice = axis_choices[i]
        local = axis_error(choice["order"], choice["cells"], choice["ellipse"],
                           choice["phase_derivative_bound"], choice["hybrid_halfwidth"])
        L = _lebesgue(choice["order"])
        error = L*error+local
        amplification *= L
    if not error <= arb(float(tolerance)):
        raise ValueError("Accumulated cascade error exceeds tolerance")
    shape = [x["axis_nodes"] for x in axis_choices]
    return {"axes": axis_choices, "cells": [x["cells"] for x in axis_choices],
            "orders": [x["order"] for x in axis_choices], "shape": shape,
            "nodes": math.prod(shape), "error_bound": upper_float(error),
            "input_error_amplification": upper_float(amplification),
            "tolerance": tolerance, "exact_arithmetic_interpolation_certified": True,
            "grid_allocated": False, "floating_point_evaluation_certified": False}


def local_grid(proof, observer, time_range, frequency_bounds, tolerance=1e-8, *,
               neighborhood_ellipses=(1.2, 2., 4., 8.), multiples=(1,)*6,
               max_order=64, max_step=.03125):
    """Construct sufficient anisotropic cells, retaining all refused trials."""
    begin = perf_counter()
    if (not np.isfinite(tolerance) or tolerance <= 0 or len(multiples) != 6
            or any(type(m) is not int or m < 1 for m in multiples)
            or type(max_order) is not int or not 1 <= max_order <= 64
            or not neighborhood_ellipses):
        raise ValueError("Valid tolerance, six positive nested counts and order budget required")
    # A priori stability cap makes independently optimized axes jointly safe.
    cap = _lebesgue(max_order)
    axis_tolerance = float((arb(float(tolerance))/(6*cap**5)).lower())
    if axis_tolerance <= 0:
        raise ValueError("Error budget underflows; use a larger planning precision")
    choices, certificates, trials, options = [], [], [], []
    for axis in AXES:
        best, winner, axis_options = None, None, []
        for ellipse in neighborhood_ellipses:
            started = perf_counter()
            try:
                neighborhood = axis_neighborhood(proof, observer, axis, ellipse, max_step=max_step)
                sensitivity = phase_bound(proof, neighborhood, observer, time_range,
                                          frequency_bounds, max_step=max_step)
                axis_options.append({"neighborhood": neighborhood, "sensitivity": sensitivity})
                choice = choose_axis(neighborhood, sensitivity["phase_derivative_bound"],
                                     axis_tolerance, multiple=int(multiples[axis]), max_order=max_order)
                trials.append({"axis": axis, "ellipse": ellipse, "status": "certified",
                               "axis_nodes": choice["axis_nodes"], "seconds": perf_counter()-started,
                               "neighborhood_work": neighborhood["work"],
                               "phase_work": sensitivity["work"]})
                if best is None or choice["axis_nodes"] < best["axis_nodes"]:
                    best, winner = choice, {"neighborhood": neighborhood, "sensitivity": sensitivity}
            except ValueError as error:
                trials.append({"axis": axis, "ellipse": ellipse, "status": "refused",
                               "reason": str(error), "seconds": perf_counter()-started})
        if best is None:
            raise ValueError(f"No certified neighborhood for axis {axis}; subdivide the patch")
        choices.append(best)
        certificates.append(winner)
        options.append(axis_options)
    # Relax the conservative independent budgets using the actual selected
    # orders. Never increase an order: all previously budgeted amplification
    # factors then remain upper bounds. Verify the full recurrence afterwards.
    for _ in range(2):
        for axis in reversed(PERMUTATION):
            rest, later = arb(0), arb(1)
            for i in PERMUTATION:
                q = choices[i]
                local = arb(0) if i == axis else axis_error(
                    q["order"], q["cells"], q["ellipse"],
                    q["phase_derivative_bound"], q["hybrid_halfwidth"])
                rest = _lebesgue(q["order"])*rest+local
            for i in PERMUTATION[PERMUTATION.index(axis)+1:]:
                later *= _lebesgue(choices[i]["order"])
            allowance = (arb(tolerance)-rest)/later*arb("0.95")
            if not allowance > 0:
                continue
            target = float(np.nextafter(float(allowance.lower()), 0.))
            if target <= 0:
                continue
            for option in options[axis]:
                candidate = choose_axis(option["neighborhood"],
                                        option["sensitivity"]["phase_derivative_bound"], target,
                                        multiple=int(multiples[axis]),
                                        max_order=choices[axis]["order"])
                if candidate["axis_nodes"] < choices[axis]["axis_nodes"]:
                    choices[axis], certificates[axis] = candidate, option
    report = grid_plan(choices, tolerance)
    report.update({"certificates": certificates, "trials_including_refusals": trials,
                   "construction_seconds": perf_counter()-begin,
                   "time_range": list(time_range), "frequency_bounds": list(frequency_bounds),
                   "full_objective_complete": False})
    return report
