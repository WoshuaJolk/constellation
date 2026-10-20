"""Reception-time coordinates and exact segment-composition research.

This is an identity/obstruction experiment, not a fast image transform.
In particular, ``rebase`` does not compress a set of input states.
"""

import math

import numpy as np

from magictools.orbits import C_AU_DAY
from .geometry_certificate import array_digest
from .observer_chart import ObserverChart
from .universal import apparent, propagate


class ReceptionChart:
    """Same coordinate ordering as ObserverChart, with *observed* rates.

    q = (inverse axial range, fractional axial range rate, north angular rate,
         east tangent position, north tangent position, east angular rate).

    Rates are derivatives with respect to reception time. The Cartesian state
    returned by ``states`` is at the reference reception epoch. The object's
    position specified by q is at the corresponding retarded emission epoch.
    All velocities are in AU/day, angles in tangent-plane radians.

    This is not a drop-in ObserverChart: the older compiled/compensated
    projection routines explicitly assume emitted rates. Use Cartesian states
    and the universal propagator for this experiment.
    """

    def __init__(self, observer, velocity, normal, east, north):
        validated = ObserverChart(observer, velocity, normal, east, north, retarded_epoch=True)
        self.observer, self.velocity, self.axes = validated.observer, validated.velocity, validated.axes
        if np.linalg.norm(self.velocity) >= C_AU_DAY:
            raise ValueError("Subluminal observer velocity required")
        self.digest = array_digest(self.observer, self.velocity, self.axes, [2.])

    def scaled_emitted(self, parameters):
        """Generic algebra for rho times the Cartesian emitted state.

        This deliberately uses a different method name from the old chart's
        compensated projection protocol, which assumes a different rate.
        """
        rho, eta, wy, x, y, wx = parameters
        normal, east, north = self.axes
        direction = [float(n)+x*float(e)+y*float(v)
                     for n, e, v in zip(normal, east, north)]
        w = [eta*d+wx*float(e)+wy*float(n) for d, e, n in zip(direction, east, north)]
        square = sum(d*d for d in direction)
        length = square.sqrt() if hasattr(square, "sqrt") else math.sqrt(square)
        alpha = 1-sum(d*vel for d, vel in zip(direction, w))/(rho*length*C_AU_DAY)
        velocity = [(rho*float(o)+vel)/alpha for o, vel in zip(self.velocity, w)]
        return [rho*float(o)+d for o, d in zip(self.observer, direction)]+velocity

    def to_dict(self):
        return {"observer": self.observer.tolist(), "velocity": self.velocity.tolist(),
                "normal": self.axes[0].tolist(), "east": self.axes[1].tolist(),
                "north": self.axes[2].tolist(),
                "rate_convention": "reception-time derivatives", "chart_kind": "ReceptionChart"}

    def emitted(self, parameters):
        q = np.asarray(parameters, float)
        if (q.ndim != 2 or q.shape[1] != 6 or not len(q)
                or not np.all(np.isfinite(q)) or np.any(q[:, 0] <= 0)):
            raise ValueError("Finite nonempty (N,6) parameters with positive inverse range required")
        rho, eta, wy, x, y, wx = q.T
        normal, east, north = self.axes
        direction = normal + x[:, None]*east + y[:, None]*north
        delta = direction/rho[:, None]
        distance = np.linalg.norm(delta, axis=1)
        line = delta/distance[:, None]
        # w = d(r_emitted(t)-observer(t))/dt, not the emitted velocity.
        w = (eta[:, None]*direction + wx[:, None]*east + wy[:, None]*north)/rho[:, None]
        alpha = 1-np.einsum("ni,ni->n", line, w)/C_AU_DAY
        if np.any(alpha <= 0):
            raise ValueError("Reception/emission time map is not increasing")
        velocity = (w+self.velocity)/alpha[:, None]
        if np.any(np.linalg.norm(velocity, axis=1) >= C_AU_DAY):
            raise ValueError("Subluminal source velocity required")
        return np.column_stack((self.observer+delta, velocity)), distance/C_AU_DAY

    def states(self, parameters):
        emitted, delay = self.emitted(parameters)
        return propagate(emitted, delay[:, None])[:, 0]

    def parameters(self, states):
        states = np.asarray(states, float)
        if (states.ndim != 2 or states.shape[1] != 6 or not len(states)
                or not np.all(np.isfinite(states))):
            raise ValueError("Finite nonempty (N,6) states required")
        emitted, _ = apparent(states, np.array([0.]), self.observer[None])
        emitted = emitted[:, 0]
        delta = emitted[:, :3]-self.observer
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance <= 0) or np.any(np.linalg.norm(emitted[:, 3:], axis=1) >= C_AU_DAY):
            raise ValueError("Nonzero range and subluminal source velocity required")
        line = delta/distance[:, None]
        alpha = ((1+line @ self.velocity/C_AU_DAY)
                 / (1+np.einsum("ni,ni->n", line, emitted[:, 3:])/C_AU_DAY))
        w = emitted[:, 3:]*alpha[:, None]-self.velocity
        r, v = delta @ self.axes.T, w @ self.axes.T
        if np.any(r[:, 0] <= 0):
            raise ValueError("States must be in front of the chart")
        rho = 1/r[:, 0]
        x, y, eta = r[:, 1]*rho, r[:, 2]*rho, v[:, 0]*rho
        return np.column_stack((rho, eta, v[:, 2]*rho-eta*y, x, y,
                                v[:, 1]*rho-eta*x))

    def rebase(self, parameters, reception_offset, target):
        """Change reception epoch and observer, preserving the physical orbit."""
        if not isinstance(target, ReceptionChart) or not np.isfinite(reception_offset):
            raise ValueError("Finite reception offset and target ReceptionChart required")
        states = propagate(self.states(parameters), np.array([float(reception_offset)]))[:, 0]
        return target.parameters(states)


def conditional_tree_counts(bandwidth, depth, *, half_span=1.):
    """Count a proposed anisotropic hierarchy, NOT a verified Kepler mesh.

    Assumes bounded, analytically regular charts with derivative weights
    (h**2, h**2, h, 1, 1, h), a bounded number of charts per node, constant-cost
    local interpolation stencils, and a balanced dyadic time partition.
    Missing geometric certificates are deliberately exposed in the result.
    Leaf fields are two-dimensional; their other four coordinates are absent.
    """
    if (isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 30
            or not np.isfinite(bandwidth) or bandwidth <= 0
            or not np.isfinite(half_span) or half_span <= 0):
        raise ValueError("Positive finite bandwidth/span and integer depth in [0,30] required")
    levels = []
    for level in range(depth+1):
        h = half_span/2**level
        spatial = max(1, math.ceil(bandwidth))
        rate = max(1, math.ceil(bandwidth*h))
        hidden = max(1, math.ceil(bandwidth*h*h))
        cells = spatial**2 if level == depth else spatial**2*rate**2*hidden**2
        levels.append({"level": level, "nodes": 2**level, "half_span": h,
                       "cells_per_node": cells, "cells_at_level": 2**level*cells})
    total = sum(level["cells_at_level"] for level in levels)
    return {"bandwidth": bandwidth, "depth": depth, "levels": levels,
            "total_cells": total, "root_cells": levels[0]["cells_per_node"],
            "leaf_cells": levels[-1]["cells_at_level"],
            "ratio_to_root_plus_leaves": total/(levels[0]["cells_per_node"]
                                              + levels[-1]["cells_at_level"]),
            "interpolation_order_and_constants_included": False,
            "chart_cover_certified": False, "full_objective_complete": False}


def finite_difference_vector(order):
    """Unit Euclidean norm alternating binomial weights, with no DC mode."""
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 64:
        raise ValueError("Integer finite difference order in [1,64] required")
    weights = np.array([(-1.)**j*math.comb(order, j) for j in range(order+1)])
    return weights/np.linalg.norm(weights)


def finite_difference_bound(order, step, derivative_bound):
    """Uniform response bound from a *supplied* uniform time derivative bound.

    For unit-norm weights a_j and |f^(order)| <= D on the whole interval,
    |sum_j a_j f(t+j*step)| <= D*|step|**order/sqrt(binomial(2*order,order)).
    Float evaluation of a mathematical bound; not outward-rounded arithmetic.
    """
    finite_difference_vector(order)
    if not np.isfinite(step) or not np.isfinite(derivative_bound) or derivative_bound < 0:
        raise ValueError("Finite step and nonnegative derivative bound required")
    return derivative_bound*abs(step)**order/math.sqrt(math.comb(2*order, order))
