"""Competent serial compiled direct-orbit reference, without fast-math.

This deliberately strengthens the baseline. Each query still solves every
exposure's all-conic Kepler and light-time equations; no atlas is used.
Compilation/startup must be reported separately from steady-state throughput.
Requires the optional accelerate extra.
"""

import math

import numpy as np
from numba import njit

from magictools.orbits import GM, C_AU_DAY


ROOT_MU = math.sqrt(GM["sun"])


@njit
def _stumpff(z):
    if abs(z) < .001:
        return (.5-z/24+z*z/720-z**3/40320+z**4/3628800,
                1/6-z/120+z*z/5040-z**3/362880+z**4/39916800)
    if z > 0:
        x = math.sqrt(z)
        return 2*math.sin(x/2)**2/z, (x-math.sin(x))/x**3
    x = math.sqrt(-z)
    if x > 600:
        return math.nan, math.nan
    return 2*math.sinh(x/2)**2/(-z), (math.sinh(x)-x)/x**3


@njit
def _equation(chi, time, radius, rv, alpha):
    z = alpha*chi*chi
    c, s = _stumpff(z)
    f = rv*chi*chi*c+(1-alpha*radius)*chi**3*s+radius*chi-ROOT_MU*time
    derivative = rv*chi*(1-z*s)+(1-alpha*radius)*chi*chi*c+radius
    return f, derivative, c, s


@njit
def _position(state, time, radius, rv, alpha):
    if time == 0:
        return state[0], state[1], state[2], True
    sign = 1. if time > 0 else -1.
    edge = max(ROOT_MU*abs(time)/radius, 1e-10)*sign
    bracketed = False
    for _ in range(80):
        value = _equation(edge, time, radius, rv, alpha)[0]
        if not math.isfinite(value):
            return 0., 0., 0., False
        if value*sign >= 0:
            bracketed = True
            break
        edge *= 2
    if not bracketed:
        return 0., 0., 0., False
    low, high = min(0., edge), max(0., edge)
    chi = (low+high)/2
    converged = False
    for _ in range(120):
        value, derivative, c, s = _equation(chi, time, radius, rv, alpha)
        if not math.isfinite(value) or not derivative > 0:
            return 0., 0., 0., False
        if value < 0:
            low = chi
        else:
            high = chi
        step = value/derivative
        if abs(step) <= 3e-14*max(1., abs(chi)):
            converged = True
            break
        proposed = chi-step
        chi = proposed if low < proposed < high else (low+high)/2
    if not converged:
        return 0., 0., 0., False
    _, _, c, s = _equation(chi, time, radius, rv, alpha)
    f, g = 1-chi*chi*c/radius, time-chi**3*s/ROOT_MU
    x, y, z = f*state[0]+g*state[3], f*state[1]+g*state[4], f*state[2]+g*state[5]
    return x, y, z, math.isfinite(x+y+z) and x*x+y*y+z*z >= 1e-10


@njit
def _project(states, times, observers, normals, east, north, epoch_delays):
    output = np.empty((len(states), len(times), 2))
    failed = np.zeros((len(states), len(times)), np.bool_)
    for row in range(len(states)):
        state = states[row]
        radius = math.sqrt(state[0]**2+state[1]**2+state[2]**2)
        if radius < 1e-5:
            failed[row, :] = True
            continue
        rv = (state[0]*state[3]+state[1]*state[4]+state[2]*state[5])/ROOT_MU
        alpha = 2/radius-(state[3]**2+state[4]**2+state[5]**2)/(ROOT_MU**2)
        for visit in range(len(times)):
            reception = times[visit]+epoch_delays[row]
            emission, converged = reception, False
            for _ in range(40):
                x, y, z, valid = _position(state, emission, radius, rv, alpha)
                if not valid:
                    break
                dx, dy, dz = x-observers[visit, 0], y-observers[visit, 1], z-observers[visit, 2]
                distance = math.sqrt(dx*dx+dy*dy+dz*dz)
                if distance < 1e-8:
                    break
                updated = reception-distance/C_AU_DAY
                if abs(updated-emission) < 1e-11:
                    emission, converged = updated, True
                    break
                emission = updated
            if not converged:
                failed[row, visit] = True
                continue
            x, y, z, valid = _position(state, emission, radius, rv, alpha)
            dx, dy, dz = x-observers[visit, 0], y-observers[visit, 1], z-observers[visit, 2]
            denominator = dx*normals[visit, 0]+dy*normals[visit, 1]+dz*normals[visit, 2]
            if not valid or not denominator > 0:
                failed[row, visit] = True
                continue
            output[row, visit, 0] = (dx*east[visit, 0]+dy*east[visit, 1]+dz*east[visit, 2])/denominator
            output[row, visit, 1] = (dx*north[visit, 0]+dy*north[visit, 1]+dz*north[visit, 2])/denominator
    return output, failed


def projected(states, geometry):
    states = np.asarray(states, float)
    if (states.ndim != 2 or states.shape[1] != 6 or not len(states) or not np.all(np.isfinite(states))
            or any(np.asarray(getattr(geometry, name)).shape != (len(geometry.times), 3)
                   or not np.all(np.isfinite(getattr(geometry, name)))
                   for name in ["observers", "normal", "east", "north"])
            or not np.all(np.isfinite(geometry.times))):
        raise ValueError("Finite compatible states and exposure geometry required")
    positions, failed = _project(np.ascontiguousarray(states), np.ascontiguousarray(geometry.times),
        np.ascontiguousarray(geometry.observers), np.ascontiguousarray(geometry.normal),
        np.ascontiguousarray(geometry.east), np.ascontiguousarray(geometry.north), np.zeros(len(states)))
    if np.any(failed) or not np.all(np.isfinite(positions)):
        raise ValueError("Compiled direct geometry failed; no queries omitted")
    return positions


def projected_chart(chart, parameters, geometry):
    """Directly use the chart's emission epoch; no extra epoch-conversion orbit.

    This gives the baseline the same useful coordinate change as the atlas.
    It still integrates every requested trajectory at every exposure.
    """
    parameters = np.asarray(parameters, float)
    if (parameters.ndim != 2 or parameters.shape[1] != 6 or not len(parameters)
            or not np.all(np.isfinite(parameters)) or np.any(parameters[:, 0] <= 0)):
        raise ValueError("Finite positive-distance (N,6) chart parameters required")
    rho, eta, wy, x, y, wx = parameters.T
    direction = chart.axes[0]+x[:, None]*chart.axes[1]+y[:, None]*chart.axes[2]
    states = np.column_stack((chart.observer+direction/rho[:, None],
        chart.velocity+(eta[:, None]*direction+wx[:, None]*chart.axes[1]+wy[:, None]*chart.axes[2])/rho[:, None]))
    delay = np.linalg.norm(direction, axis=1)/rho/C_AU_DAY if chart.retarded_epoch else np.zeros(len(states))
    positions, failed = _project(np.ascontiguousarray(states), np.ascontiguousarray(geometry.times),
        np.ascontiguousarray(geometry.observers), np.ascontiguousarray(geometry.normal),
        np.ascontiguousarray(geometry.east), np.ascontiguousarray(geometry.north), np.ascontiguousarray(delay))
    if np.any(failed) or not np.all(np.isfinite(positions)):
        raise ValueError("Compiled direct chart geometry failed; no queries omitted")
    return positions
