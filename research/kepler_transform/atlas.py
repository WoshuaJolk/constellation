"""Continuous six-dimensional geometry atlas feeding local NUFFT slices.

This changes the 3D-slice experiment into a representation over all six initial
state coordinates. It still does not guarantee a profitable globally complete
all-orbit transform: outer-coordinate sampling/synthesis and covering a broad
domain have costs. Certificates quantify the represented cell, not all space.
"""

from itertools import product
from math import factorial
from time import perf_counter

import numpy as np
from flint import arb, ctx

from research.detection_limit import interval_orbits as flow
from .jet_geometry import projected_third_coefficients
from .local_transform import LocalTransform, POWERS, QUADRATIC
from .universal import apparent
from .geometry_certificate import array_digest, geometry_digest


INTERPOLATION = np.array([[0, 1, 0], [-.5, 0, .5], [.5, -1, .5]])


def tensor_coefficients(values, dimensions):
    for axis in range(dimensions):
        values = np.moveaxis(np.tensordot(INTERPOLATION, values, axes=(1, axis)), 0, axis)
    return values


class GeometryAtlas:
    def __init__(self, center, basis, geometry):
        started = perf_counter()
        self.center, self.basis = np.asarray(center, float), np.asarray(basis, float)
        if (self.center.shape != (6,) or self.basis.shape != (6, 6)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.basis))
                or np.linalg.matrix_rank(self.basis) != 6):
            raise ValueError("A finite full-rank six-dimensional initial-state cell is required")
        self.geometry = geometry
        self.indices = list(product(range(3), repeat=6))
        nodes = np.asarray(self.indices)-1.
        states = self.center+nodes @ self.basis
        emitted, _ = apparent(states, geometry.times, geometry.observers)
        delta = emitted[..., :3]-geometry.observers
        denominator = np.einsum("nti,ti->nt", delta, geometry.normal)
        if np.any(denominator <= 0):
            raise ValueError("Atlas stencil crosses tangent horizon")
        positions = np.stack([np.einsum("nti,ti->nt", delta, axis)/denominator
                              for axis in (geometry.east, geometry.north)], axis=-1)
        self.coefficients = tensor_coefficients(positions.reshape((3,)*6+(len(geometry.times), 2)), 6)
        self.digest = array_digest(self.center, self.basis, self.coefficients)
        self.report = {"construction_seconds": perf_counter()-started,
                       "geometry_state_time_pairs": 729*len(geometry.times),
                       "continuous_dimensions": 6, "tensor_coefficients_per_coordinate": 729,
                       "coefficient_bytes": self.coefficients.nbytes,
                       "physical_geometry_error_certified": False}

    def slice(self, outer, *, correction_order=1, eps=1e-12):
        outer = np.asarray(outer, float)
        if outer.shape != (3,) or not np.all(np.isfinite(outer)) or np.max(abs(outer)) > 1:
            raise ValueError("Outer coordinates must be in [-1,1]^3")
        values = self.coefficients
        for value in outer:
            values = np.tensordot(np.array([1., value, value**2]), values, axes=(0, 0))
        selected = np.array([values[powers] for powers in POWERS])
        result = LocalTransform.from_position_coefficients(self.center+outer @ self.basis[:3],
            self.basis[3:], selected, self.geometry, correction_order=correction_order, eps=eps)
        result.atlas_sha256 = self.digest
        # Charge the actual floating-point contraction to the certificate,
        # instead of assuming arbitrary outer-coordinate queries are exact.
        with ctx.workprec(128):
            rounding = [[arb(0), arb(0)] for _ in self.geometry.times]
            for index, powers in enumerate(POWERS):
                for visit in range(len(self.geometry.times)):
                    for coordinate in range(2):
                        exact = arb(0)
                        for exponent in product(range(3), repeat=3):
                            term = arb(float(self.coefficients[exponent+powers+(visit, coordinate)]))
                            for value, degree in zip(outer, exponent):
                                term *= arb(float(value))**degree
                            exact += term
                        rounding[visit][coordinate] += abs(exact-arb(float(selected[index, visit, coordinate])))
            result.slice_roundoff_radians = np.array([[flow.upper_float(x) for x in row] for row in rounding])
        return result

    def certify(self):
        """Enclose the entire 6D cell, including all outer points between slices.

        Error covers the exact polynomial with stored binary64 coefficients,
        after omission of inner terms of total degree >2. Floating-point slice
        contraction and subsequent NUFFT arithmetic are separate uncertainties.
        """
        started = perf_counter()
        geometry = self.geometry
        with ctx.workprec(128):
            center = [arb(float(x)) for x in self.center]
            basis = [[arb(float(x)) for x in row] for row in self.basis]
            box = [c+sum((abs(row[axis]) for row in basis), arb(0))*arb(-1).union(arb(1))
                   for axis, c in enumerate(center)]
            errors, remainder_errors, coefficient_errors = [], [], []
            for visit, time in enumerate(geometry.times):
                derivatives = [projected_third_coefficients(box, direction, float(time),
                    geometry.observers[visit], geometry.normal[visit], geometry.east[visit],
                    geometry.north[visit]) for direction in self.basis]
                values = []
                for node in self.indices:
                    state = [c+sum((basis[axis][component]*(node[axis]-1) for axis in range(6)), arb(0))
                             for component, c in enumerate(center)]
                    position, _ = flow.apparent(state, float(time), geometry.observers[visit])
                    delta = [r-arb(float(o)) for r, o in zip(position, geometry.observers[visit])]
                    dot = lambda axis: sum((r*arb(float(a)) for r, a in zip(delta, axis)), arb(0))
                    denominator = dot(geometry.normal[visit])
                    if not denominator > 0:
                        raise ValueError("Certified atlas node crosses tangent horizon")
                    values.append([dot(geometry.east[visit])/denominator, dot(geometry.north[visit])/denominator])
                exact = tensor_coefficients(np.array(values, dtype=object).reshape((3,)*6+(2,)), 6)
                remainder = [arb(2)/(3*arb(3).sqrt())*sum((abs(derivatives[axis][coordinate])
                    *(arb(5)/4)**axis for axis in range(6)), arb(0)) for coordinate in range(2)]
                differences = [arb(0), arb(0)]
                for powers in self.indices:
                    for coordinate in range(2):
                        fitted = (arb(float(self.coefficients[powers+(visit, coordinate)]))
                                  if powers[3:] in POWERS else arb(0))
                        differences[coordinate] += abs(exact[powers+(coordinate,)]-fitted)
                errors.append([flow.upper_float(a+b) for a, b in zip(remainder, differences)])
                remainder_errors.append([flow.upper_float(x) for x in remainder])
                coefficient_errors.append([flow.upper_float(x) for x in differences])
        return {"method": "6D tensor interpolation with interval variational remainder",
                "geometry_sha256": geometry_digest(geometry), "atlas_sha256": self.digest,
                "center": self.center.tolist(), "basis": self.basis.tolist(),
                "position_error_radians": errors, "interpolation_remainder_radians": remainder_errors,
                "coefficient_difference_radians": coefficient_errors,
                "construction_seconds": perf_counter()-started,
                "nodal_orbit_time_enclosures": 729*len(geometry.times),
                "variational_orbit_time_enclosures": 6*len(geometry.times),
                "physical_geometry_error_certified": True, "full_6d_cell_covered": True,
                "domain": "all normalized coordinates in [-1,1]^6; one bounded initial-state cell",
                "precision_bits": 128, "mu_au3_day2": flow.MU, "c_au_day": flow.C_AU_DAY,
                "slice_and_nufft_arithmetic_certified": False,
                "full_objective_complete": False}

    def phase_tail_bound(self, data, correction_order=1):
        """Uniform phase-series error across all SIX normalized coordinates.

        The quadratic coefficients vary polynomially with the outer variables;
        their coefficient L1 norm bounds them everywhere on the outer cube.
        NUFFT and floating-point phase arithmetic remain separate.
        """
        data = np.asarray(data, complex)
        if (data.shape != (len(self.geometry.visits),) or not np.all(np.isfinite(data))
                or isinstance(correction_order, bool) or not isinstance(correction_order, int)
                or not 0 <= correction_order <= 4):
            raise ValueError("Compatible finite coefficients and order 0..4 required")
        with ctx.workprec(128):
            norms = [[arb(0), arb(0)] for _ in self.geometry.times]
            for outer in product(range(3), repeat=3):
                for inner in QUADRATIC:
                    for visit in range(len(self.geometry.times)):
                        for coordinate in range(2):
                            norms[visit][coordinate] += abs(arb(float(self.coefficients[outer+inner+(visit, coordinate)])))
            result = arb(0)
            for index, coefficient in enumerate(data):
                phase = sum((abs(arb(float(k)))*bound for k, bound in
                             zip(self.geometry.frequencies[index], norms[self.geometry.visits[index]])), arb(0))
                amplitude = (arb(float(coefficient.real))**2+arb(float(coefficient.imag))**2).sqrt()
                result += amplitude*phase**(correction_order+1)/factorial(correction_order+1)
            return flow.upper_float(result)
