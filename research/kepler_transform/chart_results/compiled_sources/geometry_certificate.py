"""Continuous inner-cube error bounds for the local polynomial construction.

Tensor quadratic interpolation at {-1,0,1} has one-dimensional Lebesgue
constant 5/4 and nodal-product maximum 2/(3 sqrt(3)). Third derivatives / 3!
are enclosed by the validated variational ODE. A telescoping interpolation
identity bounds the tensor error. Exact interval nodal coefficients also
charge omission of mixed terms and rounding of the fitted coefficients.
"""

from itertools import product
import hashlib
from time import perf_counter

import numpy as np
from flint import arb, ctx

from research.detection_limit import interval_orbits as flow
from .jet_geometry import projected_third_coefficients
from .local_transform import POWERS


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array, dtype="<f8")
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def geometry_digest(geometry):
    return array_digest(geometry.times, geometry.observers, geometry.normal, geometry.east, geometry.north)


def certify_geometry(plan):
    started = perf_counter()
    geometry = plan.geometry
    degree_indices = list(product(range(3), repeat=3))
    # Coefficients of L_-1=(z²-z)/2, L_0=1-z², L_1=(z²+z)/2.
    lagrange = [[0., -.5, .5], [1., 0., -1.], [0., .5, .5]]
    with ctx.workprec(128):
        center = [arb(float(x)) for x in plan.center]
        basis = [[arb(float(x)) for x in row] for row in plan.basis]
        box = [c+sum((abs(row[axis]) for row in basis), arb(0))*arb(-1).union(arb(1))
               for axis, c in enumerate(center)]
        derivative_bounds = []
        nodal_values = []
        for visit, time in enumerate(geometry.times):
            derivative = []
            for direction in plan.basis:
                derivative.append(projected_third_coefficients(
                    box, direction, float(time), geometry.observers[visit], geometry.normal[visit],
                    geometry.east[visit], geometry.north[visit]))
            derivative_bounds.append(derivative)
            values = []
            for node in degree_indices:
                state = [c+sum((basis[axis][component]*(node[axis]-1) for axis in range(3)), arb(0))
                         for component, c in enumerate(center)]
                position, _ = flow.apparent(state, float(time), geometry.observers[visit])
                delta = [r-arb(float(o)) for r, o in zip(position, geometry.observers[visit])]
                dot = lambda axis: sum((r*arb(float(a)) for r, a in zip(delta, axis)), arb(0))
                denominator = dot(geometry.normal[visit])
                if not denominator > 0:
                    raise ValueError("Nodal interval crosses the tangent horizon")
                values.append([dot(geometry.east[visit])/denominator, dot(geometry.north[visit])/denominator])
            nodal_values.append(values)
        errors, discarded, remainders = [], [], []
        product_max = arb(2)/(3*arb(3).sqrt())
        for visit in range(len(geometry.times)):
            axis_errors, axis_discarded, axis_remainders = [], [], []
            for coordinate in range(2):
                remainder = product_max*sum((abs(derivative_bounds[visit][axis][coordinate])
                                              *(arb(5)/4)**axis for axis in range(3)), arb(0))
                difference = arb(0)
                for powers in degree_indices:
                    coefficient = arb(0)
                    for node, values in zip(degree_indices, nodal_values[visit]):
                        factor = arb(1)
                        for axis in range(3):
                            factor *= arb(lagrange[node[axis]][powers[axis]])
                        coefficient += factor*values[coordinate]
                    fitted = (arb(float(plan.position_coefficients[POWERS.index(powers), visit, coordinate]))
                              if powers in POWERS else arb(0))
                    difference += abs(coefficient-fitted)
                axis_errors.append(flow.upper_float(remainder+difference))
                axis_discarded.append(flow.upper_float(difference))
                axis_remainders.append(flow.upper_float(remainder))
            errors.append(axis_errors)
            discarded.append(axis_discarded)
            remainders.append(axis_remainders)
    return {"method": "interval variational jets and tensor interpolation remainder",
            "geometry_sha256": geometry_digest(geometry),
            "polynomial_sha256": array_digest(plan.center, plan.basis, plan.position_coefficients),
            "center": plan.center.tolist(), "basis": plan.basis.tolist(),
            "domain": "entire normalized inner cube [-1,1]^3, fixed outer state",
            "position_error_radians": errors,
            "interpolation_remainder_radians": remainders,
            "coefficient_difference_radians": discarded,
            "construction_seconds": perf_counter()-started,
            "nodal_orbit_time_enclosures": 27*len(geometry.times),
            "variational_orbit_time_enclosures": 3*len(geometry.times),
            "precision_bits": 128, "mu_au3_day2": flow.MU, "c_au_day": flow.C_AU_DAY,
            "includes_light_time_derivatives": True,
            "physical_geometry_error_certified": True,
            "full_6d_coverage_certified": False,
            "nufft_or_coefficient_arithmetic_certified": False}


def score_error_bound(plan, certificate, coefficients):
    """Outward geometry-to-response bound, excluding NUFFT and phase arithmetic."""
    coefficients = np.asarray(coefficients, complex)
    if coefficients.shape != plan.constant.shape or not np.all(np.isfinite(coefficients)):
        raise ValueError("Finite compatible Fourier coefficients required")
    if certificate["geometry_sha256"] != geometry_digest(plan.geometry):
        raise ValueError("Geometry certificate belongs to a different exposure model")
    if "polynomial_sha256" in certificate:
        if certificate["polynomial_sha256"] != array_digest(plan.center, plan.basis, plan.position_coefficients):
            raise ValueError("Certificate belongs to a different local polynomial")
    elif certificate.get("atlas_sha256") != getattr(plan, "atlas_sha256", None):
        raise ValueError("Certificate belongs to a different six-dimensional atlas")
    with ctx.workprec(128):
        result = arb(0)
        for index, datum in enumerate(coefficients):
            amplitude = (arb(float(datum.real))**2+arb(float(datum.imag))**2).sqrt()
            error = np.asarray(certificate["position_error_radians"])[plan.geometry.visits[index]]
            contraction = getattr(plan, "slice_roundoff_radians", np.zeros((len(plan.geometry.times), 2)))[plan.geometry.visits[index]]
            phase_error = sum((abs(arb(float(k)))*arb(float(e)) for k, e in
                               zip(plan.geometry.frequencies[index], error)), arb(0))
            phase_error += sum((abs(arb(float(k)))*arb(float(e)) for k, e in
                               zip(plan.geometry.frequencies[index], contraction)), arb(0))
            result += amplitude*phase_error
        return flow.upper_float(result)
