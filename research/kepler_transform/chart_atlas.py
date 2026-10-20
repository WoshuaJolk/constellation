"""Six-dimensional observer-chart atlas with continuous geometry certificates."""

from itertools import product
import math
from time import perf_counter

import numpy as np
from flint import arb, ctx

from research.detection_limit import interval_orbits as flow
from .atlas import GeometryAtlas, tensor_coefficients
from .geometry_certificate import array_digest, geometry_digest
from .jet_geometry import Jet, retarded_jets
from .local_transform import POWERS
from .universal import apparent


def ode_scalar_position(chart, parameters, time, observer, normal, east, north):
    """Independent validated scaled ODE at one chart point or interval."""
    rho = parameters[0]
    mu, speed = arb(flow.MU)*rho**3, arb(flow.C_AU_DAY)*rho
    if not rho > 0:
        raise ValueError("Inverse distance must be bounded away from zero")
    state = chart.scaled_initial(parameters)
    time = arb(float(time))
    steps = max(1, int(math.ceil(flow.upper_float(abs(time))/2)))
    for _ in range(steps):
        state = flow.step(state, time/steps, mu=mu)
    observer = [rho*arb(float(o)) for o in observer]
    offset = chart.path_offset(parameters) if chart.retarded_epoch else arb(0)
    delay = arb(-1).union(arb(1 if chart.retarded_epoch else 0))
    enclosing = flow._tube(state, delay, mu)
    if not flow._norm(enclosing[3:]) < speed:
        raise ValueError("Scaled flow cannot establish unique light-time root")
    for iteration in range(12):
        emitted = flow.step(state, delay, mu=mu)
        updated = (offset-flow._norm([r-o for r, o in zip(emitted[:3], observer)]))/speed
        if iteration == 0 and not delay.contains(updated):
            raise ValueError("Light travel time exceeds the one-day certificate bracket")
        delay = delay.intersection(updated)
    emitted = flow.step(state, delay, mu=mu)
    delta = [r-o for r, o in zip(emitted[:3], observer)]
    dot = lambda axis: sum((r*arb(float(a)) for r, a in zip(delta, axis)), arb(0))
    denominator = dot(normal)
    if not denominator > 0:
        raise ValueError("Chart enclosure crosses tangent horizon")
    return [dot(east)/denominator, dot(north)/denominator]


def scalar_position(chart, parameters, time, observer, normal, east, north):
    """Fast validated universal root, with validated ODE fallback."""
    from .validated_universal import retarded_position
    rho = parameters[0]
    if not rho > 0:
        raise ValueError("Positive inverse distance required")
    try:
        delta = retarded_position(chart.scaled_initial(parameters), arb(float(time)),
            [rho*arb(float(o)) for o in observer], arb(flow.MU)*rho**3,
            arb(flow.C_AU_DAY)*rho, chart.path_offset(parameters) if chart.retarded_epoch else 0)
    except ValueError:
        return ode_scalar_position(chart, parameters, time, observer, normal, east, north)
    dot = lambda axis: sum((r*arb(float(a)) for r, a in zip(delta, axis)), arb(0))
    denominator = dot(normal)
    if not denominator > 0:
        raise ValueError("Validated universal position crosses tangent horizon")
    return [dot(east)/denominator, dot(north)/denominator]


def third_coefficients(chart, center, widths, direction, time, observer, normal, east, north, *, translated=False):
    """Third derivative / factorial in a normalized chart coordinate.

    Constant jets cover the ENTIRE parameter cell. The normalized equations
    include parameter derivatives of both mu*rho^3 and c*rho.
    """
    parameters = [Jet([arb(float(c), float(w)), float(w) if j == direction else 0, 0, 0])
                  for j, (c, w) in enumerate(zip(center, widths))]
    rho = parameters[0]
    initial = chart.relative_initial(parameters) if translated else chart.scaled_initial(parameters)
    r, h = rho.c[:2]
    # rho is affine in the differentiation variable. Its exact cubic
    # coefficients preserve positivity of mu even for wide distance cells.
    # Repeated ball multiplication can lose that known positivity.
    r2 = (r.lower()**2).union(r.upper()**2)
    r3 = (r.lower()**3).union(r.upper()**3)
    gravity = Jet([r3, 3*r2*h, 3*r*h*h, h*h*h])*arb(flow.MU)
    observer_coordinates = ([rho*(arb(float(o))-arb(float(o0))) for o, o0 in zip(observer, chart.observer)]
                            if translated else [rho*float(o) for o in observer])
    q = retarded_jets(initial, time, observer_coordinates,
                      mu=gravity, light_speed=rho*arb(flow.C_AU_DAY),
                      path_offset=chart.path_offset(parameters) if chart.retarded_epoch else None,
                      origin=[rho*float(o) for o in chart.observer] if translated else None)
    dot = lambda axis: sum((r*float(a) for r, a in zip(q, axis)), Jet(0))
    denominator = dot(normal)
    if not denominator.c[0] > 0:
        raise ValueError("Chart variational enclosure crosses tangent horizon")
    return [(dot(axis)/denominator).c[3] for axis in (east, north)]


class ChartAtlas(GeometryAtlas):
    """Tensor-quadratic geometry; inherited slice/adjoint/inverse machinery.

    center/basis below describe chart parameters, not Cartesian states. Use
    states() to obtain Cartesian initial conditions. Slice.center/basis are
    labels only; the LocalTransform is compiled from position coefficients.
    """
    def __init__(self, chart, center, widths, geometry, *, translated=False):
        started = perf_counter()
        self.chart = chart
        self.translated = translated
        self.center, self.widths = np.asarray(center, float), np.asarray(widths, float)
        if (self.center.shape != (6,) or self.widths.shape != (6,)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.widths))
                or np.any(self.widths <= 0) or self.center[0] <= self.widths[0]):
            raise ValueError("Finite positive chart widths and strictly positive inverse distance required")
        self.basis = np.diag(self.widths)
        self.geometry = geometry
        self.indices = list(product(range(3), repeat=6))
        nodes = np.asarray(self.indices)-1.
        states = self.states(nodes)
        emitted, _ = apparent(states, geometry.times, geometry.observers)
        delta = emitted[..., :3]-geometry.observers
        denominator = np.einsum("nti,ti->nt", delta, geometry.normal)
        if np.any(denominator <= 0):
            raise ValueError("Chart stencil crosses tangent horizon")
        positions = np.stack([np.einsum("nti,ti->nt", delta, axis)/denominator
                              for axis in (geometry.east, geometry.north)], axis=-1)
        self.coefficients = tensor_coefficients(positions.reshape((3,)*6+(len(geometry.times), 2)), 6)
        self.digest = array_digest(self.center, self.widths, self.coefficients,
                                   chart.observer, chart.velocity, chart.axes, [float(chart.retarded_epoch)])
        self.report = {"construction_seconds": perf_counter()-started,
                       "geometry_state_time_pairs": 729*len(geometry.times),
                       "continuous_dimensions": 6, "coefficient_bytes": self.coefficients.nbytes,
                       "coordinates": "observer chart", "physical_geometry_error_certified": False}

    def states(self, normalized):
        normalized = np.asarray(normalized, float)
        if (normalized.ndim != 2 or normalized.shape[1] != 6
                or not np.all(np.isfinite(normalized)) or np.any(abs(normalized) > 1)):
            raise ValueError("Normalized chart points must lie in [-1,1]^6")
        return self.chart.states(self.center+normalized*self.widths)

    def certify(self, *, max_position_error=None):
        """Return a certificate, or a subdivision request from a cheap screen.

        A screen exceeding the requested tolerance is not a lower bound on
        the actual error; it means this sufficient bound cannot accept the cell.
        Every accepted cell goes through all variational and nodal checks.
        """
        if max_position_error is not None and (not np.isfinite(max_position_error) or max_position_error <= 0):
            raise ValueError("Positive finite position tolerance required")
        started = perf_counter()
        geometry = self.geometry
        with ctx.workprec(128):
            errors, remainders, differences_all, directional = [], [], [], []
            cached = {}
            # Longest offsets usually expose invalid wide cells earliest.
            for visit in np.argsort(-abs(geometry.times)):
                axes = (geometry.observers[visit], geometry.normal[visit],
                        geometry.east[visit], geometry.north[visit])
                derivatives = [third_coefficients(self.chart, self.center, self.widths,
                    j, float(geometry.times[visit]), *axes, translated=self.translated) for j in range(6)]
                components = [[arb(2)/(3*arb(3).sqrt())*abs(derivatives[j][axis])*(arb(5)/4)**j
                               for axis in range(2)] for j in range(6)]
                cached[int(visit)] = components
                if (max_position_error is not None and
                        any(flow.upper_float(sum((row[a] for row in components), arb(0)))
                            > max_position_error for a in range(2))):
                    return {"status": "needs_subdivision", "physical_geometry_error_certified": False,
                            "screen_visit": int(visit),
                            "directional_remainder_radians": [[[flow.upper_float(x) for x in row]
                                                               for row in components]],
                            "construction_seconds": perf_counter()-started,
                            "nodal_orbit_time_enclosures": 0,
                            "variational_orbit_time_enclosures": 6*len(cached)}
            for visit, time in enumerate(geometry.times):
                axes = (geometry.observers[visit], geometry.normal[visit],
                        geometry.east[visit], geometry.north[visit])
                # Node arithmetic in Arb begins from exact stored binary64
                # center/width. Float node generation errors are included by
                # comparison with the resulting exact interval polynomial.
                values = [scalar_position(self.chart,
                    [arb(float(c))+arb(float(w))*(k-1) for c, w, k in
                     zip(self.center, self.widths, index)], float(time), *axes) for index in self.indices]
                exact = tensor_coefficients(np.array(values, dtype=object).reshape((3,)*6+(2,)), 6)
                components = cached[visit]
                remainder = [sum((row[axis] for row in components), arb(0)) for axis in range(2)]
                difference = [arb(0), arb(0)]
                for powers in self.indices:
                    for axis in range(2):
                        fitted = (arb(float(self.coefficients[powers+(visit, axis)]))
                                  if powers[3:] in POWERS else arb(0))
                        difference[axis] += abs(exact[powers+(axis,)]-fitted)
                errors.append([flow.upper_float(a+b) for a, b in zip(remainder, difference)])
                remainders.append([flow.upper_float(x) for x in remainder])
                differences_all.append([flow.upper_float(x) for x in difference])
                directional.append([[flow.upper_float(x) for x in row] for row in components])
        return {"method": "scaled observer-chart variational ODE and 6D tensor interpolation",
                "status": "certified",
                "geometry_sha256": geometry_digest(geometry), "atlas_sha256": self.digest,
                "chart": self.chart.to_dict(), "center": self.center.tolist(), "widths": self.widths.tolist(),
                "translated_certificate": self.translated,
                "position_error_radians": errors, "interpolation_remainder_radians": remainders,
                "coefficient_difference_radians": differences_all,
                "directional_remainder_radians": directional,
                "construction_seconds": perf_counter()-started,
                "nodal_orbit_time_enclosures": 729*len(geometry.times),
                "variational_orbit_time_enclosures": 6*len(geometry.times),
                "physical_geometry_error_certified": True, "full_6d_cell_covered": True,
                "precision_bits": 128, "mu_au3_day2": flow.MU, "c_au_day": flow.C_AU_DAY,
                "slice_and_nufft_arithmetic_certified": False, "full_objective_complete": False}
