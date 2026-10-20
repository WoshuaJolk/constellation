"""Higher-order outer coordinates: quartic x quartic x quartic x quadratic^3.

All interpolation nodes are exact rationals. Fifth variational derivatives
bound the outer interpolation, third derivatives the inner interpolation.
This is a second broad-domain construction, not a sampled accuracy heuristic.
"""

from fractions import Fraction
from itertools import product
from math import factorial
from time import perf_counter

import numpy as np
from flint import arb, ctx

from research.detection_limit import interval_orbits as flow
from .chart_atlas import ChartAtlas, scalar_position, third_coefficients
from .fifth_geometry import Jet, retarded_jets
from .geometry_certificate import array_digest, geometry_digest
from .local_transform import LocalTransform, POWERS, QUADRATIC
from .universal import apparent


def lagrange_coefficients(nodes):
    columns = []
    for i, node in enumerate(nodes):
        polynomial, denominator = [Fraction(1)], Fraction(1)
        for j, other in enumerate(nodes):
            if i == j:
                continue
            following = [Fraction(0)]*(len(polynomial)+1)
            for degree, coefficient in enumerate(polynomial):
                following[degree] -= other*coefficient
                following[degree+1] += coefficient
            polynomial = following
            denominator *= node-other
        columns.append([coefficient/denominator for coefficient in polynomial])
    return list(zip(*columns))


QUARTIC = lagrange_coefficients([Fraction(j, 2) for j in range(-2, 3)])
QUADRATIC_INTERPOLATION = lagrange_coefficients([Fraction(j) for j in range(-1, 2)])
OUTER_POWERS = list(product(range(5), repeat=3))


def transform(values, *, exact=False):
    for axis in range(6):
        coefficients = QUARTIC if axis < 3 else QUADRATIC_INTERPOLATION
        matrix = np.array([[arb(q.numerator)/q.denominator if exact else float(q)
                            for q in row] for row in coefficients], dtype=object if exact else float)
        values = np.moveaxis(np.tensordot(matrix, values, axes=(1, axis)), 0, axis)
    return values


def interpolation_constants():
    """Interval proof over an exact partition of [-1,1], not point sampling."""
    with ctx.workprec(128):
        lebesgue, nodal_product = 0., 0.
        for j in range(256):
            x = (arb(-1)+arb(j)/128).union(arb(-1)+arb(j+1)/128)
            norm = arb(0)
            for column in zip(*QUARTIC):
                value = arb(0)
                for q in reversed(column):
                    value = value*x+arb(q.numerator)/q.denominator
                norm += abs(value)
            value = arb(1)
            for j2 in range(-2, 3):
                value *= x-arb(j2)/2
            lebesgue = max(lebesgue, flow.upper_float(norm))
            nodal_product = max(nodal_product, flow.upper_float(abs(value)))
    if lebesgue > 3 or nodal_product > .125:
        raise ValueError("Quartic interpolation constants not established")
    return {"partition_intervals": 256, "lebesgue_upper": lebesgue,
            "nodal_product_upper": nodal_product,
            "used_lebesgue_bound": 3, "used_nodal_product_bound": .125}


def fifth_coefficients(chart, center, widths, direction, time, observer, normal, east, north):
    parameters = [Jet([arb(float(c), float(w)), float(w) if j == direction else 0, 0, 0, 0, 0])
                  for j, (c, w) in enumerate(zip(center, widths))]
    rho = parameters[0]
    q = retarded_jets(chart.scaled_initial(parameters), time, [rho*float(o) for o in observer],
                     mu=rho*rho*rho*arb(flow.MU), light_speed=rho*arb(flow.C_AU_DAY),
                     path_offset=chart.path_offset(parameters) if chart.retarded_epoch else None)
    dot = lambda axis: sum((r*float(a) for r, a in zip(q, axis)), Jet(0))
    denominator = dot(normal)
    if not denominator.c[0] > 0:
        raise ValueError("Fifth-order enclosure crosses tangent horizon")
    return [(dot(axis)/denominator).c[5] for axis in (east, north)]


class QuarticChartAtlas(ChartAtlas):
    outer_powers = OUTER_POWERS

    def __init__(self, chart, center, widths, geometry):
        started = perf_counter()
        self.chart, self.geometry = chart, geometry
        self.center, self.widths = np.asarray(center, float), np.asarray(widths, float)
        if (self.center.shape != (6,) or self.widths.shape != (6,)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.widths))
                or np.any(self.widths <= 0) or self.center[0] <= self.widths[0]):
            raise ValueError("Finite positive-width, positive-distance chart required")
        self.basis = np.diag(self.widths)
        self.indices = list(product(range(5), range(5), range(5), range(3), range(3), range(3)))
        nodes = np.array(self.indices, float)
        nodes[:, :3] = nodes[:, :3]/2-1
        nodes[:, 3:] -= 1
        emitted, _ = apparent(self.states(nodes), geometry.times, geometry.observers)
        delta = emitted[..., :3]-geometry.observers
        denominator = np.einsum("nti,ti->nt", delta, geometry.normal)
        if np.any(denominator <= 0):
            raise ValueError("Quartic stencil crosses tangent horizon")
        positions = np.stack([np.einsum("nti,ti->nt", delta, axis)/denominator
                              for axis in (geometry.east, geometry.north)], axis=-1)
        self.coefficients = transform(positions.reshape((5, 5, 5, 3, 3, 3, len(geometry.times), 2)))
        self.digest = array_digest(self.center, self.widths, self.coefficients,
                                   chart.observer, chart.velocity, chart.axes, [float(chart.retarded_epoch)])
        self.report = {"construction_seconds": perf_counter()-started,
                       "geometry_state_time_pairs": 3375*len(geometry.times),
                       "continuous_dimensions": 6, "coefficient_bytes": self.coefficients.nbytes,
                       "coordinates": "observer chart; quartic outer coordinates",
                       "physical_geometry_error_certified": False}

    def slice(self, outer, *, correction_order=1, eps=1e-12):
        outer = np.asarray(outer, float)
        if outer.shape != (3,) or not np.all(np.isfinite(outer)) or np.any(abs(outer) > 1):
            raise ValueError("Outer points in [-1,1]^3 required")
        values = self.coefficients
        for value in outer:
            values = np.tensordot(value**np.arange(5), values, axes=(0, 0))
        selected = np.array([values[powers] for powers in POWERS])
        result = LocalTransform.from_position_coefficients(self.center+outer @ self.basis[:3],
            self.basis[3:], selected, self.geometry, correction_order=correction_order, eps=eps)
        result.atlas_sha256 = self.digest
        with ctx.workprec(128):
            rounding = [[arb(0), arb(0)] for _ in self.geometry.times]
            for index, powers in enumerate(POWERS):
                for visit in range(len(self.geometry.times)):
                    for coordinate in range(2):
                        exact = arb(0)
                        for exponent in OUTER_POWERS:
                            term = arb(float(self.coefficients[exponent+powers+(visit, coordinate)]))
                            for value, degree in zip(outer, exponent):
                                term *= arb(float(value))**degree
                            exact += term
                        rounding[visit][coordinate] += abs(exact-arb(float(selected[index, visit, coordinate])))
            result.slice_roundoff_radians = np.array([[flow.upper_float(x) for x in row] for row in rounding])
        return result

    def certify(self, *, max_position_error=None):
        if max_position_error is not None and (not np.isfinite(max_position_error) or max_position_error <= 0):
            raise ValueError("Positive finite position tolerance required")
        started, geometry = perf_counter(), self.geometry
        constants = interpolation_constants()
        with ctx.workprec(128):
            cached = {}
            for visit in np.argsort(-abs(geometry.times)):
                axes = (geometry.observers[visit], geometry.normal[visit],
                        geometry.east[visit], geometry.north[visit])
                derivatives = [(fifth_coefficients if j < 3 else third_coefficients)(
                    self.chart, self.center, self.widths, j, float(geometry.times[visit]), *axes)
                    for j in range(6)]
                components, norm = [], arb(1)
                for j in range(6):
                    product_bound = arb(1)/8 if j < 3 else arb(2)/(3*arb(3).sqrt())
                    components.append([norm*product_bound*abs(value) for value in derivatives[j]])
                    norm *= 3 if j < 3 else arb(5)/4
                cached[int(visit)] = components
                if max_position_error is not None and any(flow.upper_float(
                        sum((row[a] for row in components), arb(0))) > max_position_error for a in range(2)):
                    return {"status": "needs_subdivision", "physical_geometry_error_certified": False,
                            "screen_visit": int(visit), "construction_seconds": perf_counter()-started,
                            "directional_remainder_radians": [[[flow.upper_float(x) for x in row] for row in components]],
                            "nodal_orbit_time_enclosures": 0, "variational_orbit_time_enclosures": 6*len(cached)}
            errors, remainders, differences_all, directional = [], [], [], []
            for visit, time in enumerate(geometry.times):
                axes = (geometry.observers[visit], geometry.normal[visit], geometry.east[visit], geometry.north[visit])
                values = []
                for index in self.indices:
                    parameters = [arb(float(c))+arb(float(w))*(arb(k)/2-1 if j < 3 else k-1)
                                  for j, (c, w, k) in enumerate(zip(self.center, self.widths, index))]
                    values.append(scalar_position(self.chart, parameters, float(time), *axes))
                exact = transform(np.array(values, dtype=object).reshape((5, 5, 5, 3, 3, 3, 2)), exact=True)
                components = cached[visit]
                remainder = [sum((row[axis] for row in components), arb(0)) for axis in range(2)]
                difference = [arb(0), arb(0)]
                for powers in self.indices:
                    for axis in range(2):
                        fitted = arb(float(self.coefficients[powers+(visit, axis)])) if powers[3:] in POWERS else arb(0)
                        difference[axis] += abs(exact[powers+(axis,)]-fitted)
                errors.append([flow.upper_float(a+b) for a, b in zip(remainder, difference)])
                remainders.append([flow.upper_float(x) for x in remainder])
                differences_all.append([flow.upper_float(x) for x in difference])
                directional.append([[flow.upper_float(x) for x in row] for row in components])
        return {"method": "quartic outer/quad inner tensor with fifth/third interval variational remainder",
                "status": "certified", "geometry_sha256": geometry_digest(geometry), "atlas_sha256": self.digest,
                "chart": self.chart.to_dict(), "center": self.center.tolist(), "widths": self.widths.tolist(),
                "position_error_radians": errors, "interpolation_remainder_radians": remainders,
                "coefficient_difference_radians": differences_all,
                "directional_remainder_radians": directional,
                "interpolation_constants_proof": constants,
                "construction_seconds": perf_counter()-started, "nodal_orbit_time_enclosures": 3375*len(geometry.times),
                "variational_orbit_time_enclosures": 6*len(geometry.times),
                "physical_geometry_error_certified": True, "full_6d_cell_covered": True,
                "precision_bits": 128, "mu_au3_day2": flow.MU, "c_au_day": flow.C_AU_DAY,
                "slice_and_nufft_arithmetic_certified": False, "full_objective_complete": False}

    def phase_tail_bound(self, data, correction_order=1):
        data = np.asarray(data, complex)
        if (data.shape != (len(self.geometry.visits),) or not np.all(np.isfinite(data))
                or isinstance(correction_order, bool) or not isinstance(correction_order, int)
                or not 0 <= correction_order <= 4):
            raise ValueError("Compatible finite coefficients and order 0..4 required")
        with ctx.workprec(128):
            norms = [[arb(0), arb(0)] for _ in self.geometry.times]
            for outer in OUTER_POWERS:
                for inner in QUADRATIC:
                    for visit in range(len(self.geometry.times)):
                        for axis in range(2):
                            norms[visit][axis] += abs(arb(float(self.coefficients[outer+inner+(visit, axis)])))
            result = arb(0)
            for index, datum in enumerate(data):
                phase = sum((abs(arb(float(k)))*bound for k, bound in
                             zip(self.geometry.frequencies[index], norms[self.geometry.visits[index]])), arb(0))
                amplitude = (arb(float(datum.real))**2+arb(float(datum.imag))**2).sqrt()
                result += amplitude*phase**(correction_order+1)/factorial(correction_order+1)
            return flow.upper_float(result)
