"""Interval derivatives of the exact reception-coordinate Kepler flow.

First-order automatic differentiation and a Peano--Baker enclosure are used;
finite differences are not used to certify derivatives. The state tube comes
from reception_flow's interval Picard inclusion. Model constants are the exact
binary values supplied to that implementation.
"""

import math

import numpy as np
from flint import acb, acb_mat, arb, arb_mat

from magictools.orbits import C_AU_DAY, GM
from research.detection_limit.interval_orbits import upper_float
from .reception_flow import _square, rhs, step


class Jet:
    """One value and six interval directional derivatives."""

    def __init__(self, value, derivative=None):
        self.v = value
        self.d = [value*0 for _ in range(6)] if derivative is None else derivative

    def coerce(self, other):
        return other if isinstance(other, Jet) else Jet(self.v*0+other)

    def __add__(self, other):
        other = self.coerce(other)
        return Jet(self.v+other.v, [a+b for a, b in zip(self.d, other.d)])

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, [-a for a in self.d])

    def __sub__(self, other):
        return self+-self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other)+-self

    def __mul__(self, other):
        other = self.coerce(other)
        return Jet(self.v*other.v,
                   [a*other.v+self.v*b for a, b in zip(self.d, other.d)])

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.coerce(other)
        value = self.v/other.v
        return Jet(value, [(a-value*b)/other.v for a, b in zip(self.d, other.d)])

    def __rtruediv__(self, other):
        return self.coerce(other)/self

    def __pow__(self, power):
        value = self.v**power
        factor = power*self.v**(power-1)
        return Jet(value, [factor*a for a in self.d])

    def square(self):
        return Jet(_square(self.v), [2*self.v*a for a in self.d])


def rhs_jacobian(time, values, observer, scales):
    """Return W^-1 Df W on an entire (real or complex) state/time box."""
    scales = np.asarray(scales, float)
    if scales.shape != (6,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Six positive finite derivative scales required")
    # Enforce exactly the same physical/holomorphic branch checks as the ODE.
    rhs(time, values, observer)
    q = [Jet(v, [v*0+(float(scales[j]) if i == j else 0) for j in range(6)])
         for i, v in enumerate(values)]
    rho, eta, wy, x, y, wx = q
    n, e, b = observer.axes
    O, V, A = observer.jet(time)
    D = [float(nj)+x*float(ej)+y*float(bj) for nj, ej, bj in zip(n, e, b)]
    length = sum(d.square() for d in D)**arb(".5")
    line = [d/length for d in D]
    w = [(eta*d+wx*float(ej)+wy*float(bj))/rho
         for d, ej, bj in zip(D, e, b)]
    radial = sum(a*c for a, c in zip(line, w))
    alpha = 1-radial/C_AU_DAY
    velocity = [(u+v)/alpha for v, u in zip(V, w)]
    position = [d/rho+o for o, d in zip(O, D)]
    inverse_cube = sum(r.square() for r in position)**arb("-1.5")
    acceleration = [-arb(GM["sun"])*r*inverse_cube for r in position]
    transverse = sum(u.square() for u in w)-radial.square()
    denominator = sum(a*v for a, v in zip(line, velocity))+C_AU_DAY
    alpha_prime = -(transverse/(length/rho)
                    +sum(a*(g*alpha.square()-oa)
                         for a, g, oa in zip(line, acceleration, A)))/denominator
    w_prime = [g*alpha.square()+v*alpha_prime-oa
               for g, v, oa in zip(acceleration, velocity, A)]
    dual_n, dual_e, dual_b = observer.dual_axes
    axial = sum(a*j for a, j in zip(w_prime, dual_n))*rho
    east = sum(a*j for a, j in zip(w_prime, dual_e))*rho
    north = sum(a*j for a, j in zip(w_prime, dual_b))*rho
    result = [-rho*eta, axial-eta.square(), north-y*axial-2*eta*wy,
              wx, wy, east-x*axial-2*eta*wx]
    matrix_type = acb_mat if isinstance(values[0], acb) else arb_mat
    return matrix_type([[a/float(scales[i]) for a in row.d]
                        for i, row in enumerate(result)])


def peano_increment(matrix, offset, *, terms=24):
    """Enclose J(h)-I for J'=A(t)J, J(0)=I and A(t) in matrix.

    A(t) need not commute at distinct times. Interval powers enclose every
    ordered product. The omitted ordered-simplex integrals are bounded with
    the induced infinity norm and an exponential-series tail.
    """
    if not isinstance(terms, int) or terms < 1:
        raise ValueError("A positive integer Peano order is required")
    offset = arb(offset)
    dimension = matrix.nrows()
    if matrix.ncols() != dimension or not offset.is_exact() or not offset.is_finite():
        raise ValueError("Square matrix and finite exact offset required")
    kind = acb_mat if isinstance(matrix, acb_mat) else arb_mat
    identity = kind([[int(i == j) for j in range(dimension)] for i in range(dimension)])
    power, result = identity, identity*0
    scaled = matrix*offset
    norm = max(sum(abs(scaled[i, j]).upper() for j in range(dimension))
               for i in range(dimension)).upper()
    for j in range(1, terms+1):
        power = power*scaled/j
        result += power
    # exp(x) * x^(n+1)/(n+1)! bounds the entire omitted tail, x>=0.
    tail = (norm.exp()*norm**(terms+1)/math.factorial(terms+1)).upper()
    if not tail.is_finite():
        raise ValueError("Nonfinite variational tail; reduce the time step")
    error = acb(arb(0, tail), arb(0, tail)) if kind is acb_mat else arb(0, tail)
    result += kind([[error for _ in range(dimension)] for _ in range(dimension)])
    return result, tail


def flow_jacobian(values, start, stop, observer, scales, *, max_step=.03125, terms=24):
    """Enclose endpoint and W^-1 DR W-I for every initial state in values."""
    start, stop = arb(start), arb(stop)
    if (not start.is_exact() or not stop.is_exact() or not start.is_finite()
            or not stop.is_finite() or not np.isfinite(max_step) or max_step <= 0):
        raise ValueError("Finite exact times and positive finite maximum step required")
    count = max(1, math.ceil(upper_float(abs(stop-start))/max_step))
    count = 1 << (count-1).bit_length()
    h = (stop-start)/count
    kind = acb_mat if isinstance(values[0], acb) else arb_mat
    increment = kind([[0]*6 for _ in range(6)])
    endpoint = list(values)
    iterations, largest_tail = 0, arb(0)
    for j in range(count):
        time = start+j*h
        endpoint, tube, used = step(endpoint, time, h, observer)
        matrix = rhs_jacobian(time+arb(0).union(h), tube, observer, scales)
        local, tail = peano_increment(matrix, h, terms=terms)
        # (I+local)(I+increment)-I, without subtracting nearly equal matrices.
        increment = local+increment+local*increment
        largest_tail = max(largest_tail, tail)
        iterations += used
    return endpoint, increment, {
        "state_steps": count, "picard_iterations": iterations,
        "jacobian_evaluations": count, "peano_terms": terms,
        "largest_local_tail": upper_float(largest_tail),
        "whole_box_derivative_enclosure": True,
    }
