"""Choose the cheapest phase expansion meeting a uniform operator tolerance."""

from itertools import product
from math import factorial

from flint import arb, ctx

from research.detection_limit.interval_orbits import upper_float
from .local_transform import QUADRATIC


def choose_order(atlas, tolerance):
    """Guarantee analytic |error| <= tolerance * sum(abs(input coefficients)).

    Covers all six normalized coordinates in this atlas. Geometry errors and
    floating-point/NUFFT arithmetic are separate. If no order <=4 suffices,
    callers can use exact polynomial phases with per-exposure summation.
    """
    if not 0 < tolerance < 1:
        raise ValueError("Phase operator tolerance must lie between zero and one")
    outer_powers = getattr(atlas, "outer_powers", list(product(range(3), repeat=3)))
    with ctx.workprec(128):
        norms = [[arb(0), arb(0)] for _ in atlas.geometry.times]
        for outer in outer_powers:
            for inner in QUADRATIC:
                for visit in range(len(atlas.geometry.times)):
                    for axis in range(2):
                        norms[visit][axis] += abs(arb(float(atlas.coefficients[outer+inner+(visit, axis)])))
        maximum = 0.
        for visit, frequency in zip(atlas.geometry.visits, atlas.geometry.frequencies):
            phase = sum((abs(arb(float(k)))*bound for k, bound in zip(frequency, norms[visit])), arb(0))
            maximum = max(maximum, upper_float(phase))
        bounds = [upper_float(arb(maximum)**(order+1)/factorial(order+1)) for order in range(5)]
    order = next((i for i, bound in enumerate(bounds) if bound <= tolerance), None)
    return {"selected_order": order, "maximum_quadratic_phase": maximum,
            "operator_bounds_by_order": bounds, "requested_operator_tolerance": tolerance,
            "uniform_6d_analytic_bound": True, "floating_point_arithmetic_certified": False}
