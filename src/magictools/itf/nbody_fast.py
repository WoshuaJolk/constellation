"""Compiled adaptive DOP853 for the JPL spline force and its variational system.

Uses SciPy's published DOP853 coefficients and error/dense-output formulas.
No fast-math, fixed-step shortcut, or reduced dynamical model. Python regains
control after at most 64 accepted steps to enforce the caller's deadline.
"""
import numpy as np
from numba import njit
from scipy.integrate._ivp import dop853_coefficients as coefficients

from .fit import check_deadline

A = coefficients.A.copy()
B = coefficients.B.copy()
C = coefficients.C.copy()
D = coefficients.D.copy()
E3 = coefficients.E3.copy()
E5 = coefficients.E5.copy()


@njit(cache=True)
def derivative(t, y, grid, spline, masses, mu):
    out = np.empty(42)
    derivative_into(t, y, grid, spline, masses, mu, out, np.empty(15))
    return out


@njit(cache=True)
def derivative_into(t, y, grid, spline, masses, mu, out, scratch):
    """Write the force/STM derivative into caller-owned RK storage."""
    k = min(max(np.searchsorted(grid, t, side='right')-1, 0), len(grid)-2)
    if t < grid[0]-1e-9 or t > grid[-1]+1e-9:
        raise ValueError('Force outside prepared ephemeris interval')
    dt = t-grid[k]
    r2 = y[0]*y[0]+y[1]*y[1]+y[2]*y[2]
    r = np.sqrt(r2)
    if r < .00465:
        raise ValueError('Trajectory intersects the Sun')
    gradient = scratch[:9].reshape(3, 3)
    for i in range(3):
        out[i] = y[i+3]
        out[i+3] = -mu*y[i]/(r2*r)
        for j in range(3):
            gradient[i, j] = mu*(3*y[i]*y[j]/(r2*r2*r)-(i == j)/(r2*r))
    delta = scratch[9:12]
    planet = scratch[12:15]
    for body in range(len(masses)):
        p2 = 0.
        for i in range(3):
            planet[i] = ((spline[0, k, body, i]*dt+spline[1, k, body, i])*dt+
                         spline[2, k, body, i])*dt+spline[3, k, body, i]
            delta[i] = planet[i]-y[i]
            p2 += planet[i]*planet[i]
        d2 = np.dot(delta, delta)
        distance = np.sqrt(d2)
        if distance < 1e-8:
            raise ValueError('Unresolved planetary encounter')
        d3, d5, p3 = d2*distance, d2*d2*distance, p2*np.sqrt(p2)
        for i in range(3):
            out[i+3] += masses[body]*(delta[i]/d3-planet[i]/p3)
            for j in range(3):
                gradient[i, j] += masses[body]*(3*delta[i]*delta[j]/d5-(i == j)/d3)
    for i in range(3):
        for j in range(6):
            out[6+i*6+j] = y[24+i*6+j]
            out[24+i*6+j] = 0.
            for l in range(3):
                out[24+i*6+j] += gradient[i, l]*y[6+l*6+j]


@njit(cache=True)
def initial_step(y, f, epoch, target, rtol, atol, max_step, grid, spline, masses, mu):
    scale = atol+np.abs(y)*rtol
    d0 = np.sqrt(np.mean((y/scale)**2))
    d1 = np.sqrt(np.mean((f/scale)**2))
    h0 = min(1e-6 if d0 < 1e-5 or d1 < 1e-5 else .01*d0/d1, abs(target))
    direction = 1. if target > 0 else -1.
    f1 = derivative(epoch+direction*h0, y+direction*h0*f, grid, spline, masses, mu)
    d2 = np.sqrt(np.mean(((f1-f)/scale)**2))/h0
    h1 = max(1e-6, h0*1e-3) if d1 <= 1e-15 and d2 <= 1e-15 else (.01/max(d1, d2))**.125
    return min(100*h0, h1, abs(target), max_step)


@njit(cache=True)
def integrate_chunk(t, y, f, h_abs, target, epoch, rtol, atol, max_step,
                    grid, spline, masses, mu):
    times = np.empty(65)
    old = np.empty((64, 42))
    dense = np.empty((64, 7, 42))
    stages = np.empty((16, 42))
    trial = np.empty(42)
    scratch = np.empty(15)
    times[0] = t
    direction = 1. if target > 0 else -1.
    count = 0
    attempts = 0
    while direction*(target-t) > 0 and count < 64 and attempts < 256:
        min_step = 10*abs(np.nextafter(t, direction*np.inf)-t)
        h_abs = min(max_step, max(min_step, h_abs))
        rejected = False
        while True:
            attempts += 1
            if h_abs < min_step or attempts > 512:
                raise ValueError('N-body integration step cannot converge')
            end = t+direction*h_abs
            if direction*(end-target) > 0:
                end = target
            h = end-t
            h_abs = abs(h)
            stages[0] = f
            for stage in range(1, 12):
                trial[:] = 0.
                for l in range(stage):
                    coefficient = A[stage, l]
                    if coefficient != 0.:
                        for j in range(42):
                            trial[j] += coefficient*stages[l, j]
                for j in range(42):
                    trial[j] = y[j]+h*trial[j]
                derivative_into(epoch+t+C[stage]*h, trial, grid, spline, masses, mu, stages[stage], scratch)
            new = y.copy()
            for j in range(42):
                value = 0.
                for l in range(12):
                    value += B[l]*stages[l, j]
                new[j] += h*value
            derivative_into(epoch+end, new, grid, spline, masses, mu, stages[12], scratch)
            f_new = stages[12].copy()
            err5, err3 = 0., 0.
            for j in range(42):
                scale = atol+max(abs(y[j]), abs(new[j]))*rtol
                a, b = 0., 0.
                for l in range(13):
                    a += E5[l]*stages[l, j]
                    b += E3[l]*stages[l, j]
                err5 += (a/scale)**2
                err3 += (b/scale)**2
            error = 0. if err5 == 0 and err3 == 0 else abs(h)*err5/np.sqrt((err5+.01*err3)*42)
            if not np.isfinite(error):
                raise ValueError('Nonfinite N-body integration error')
            if error < 1:
                factor = 10. if error == 0 else min(10., .9*error**(-.125))
                h_abs *= min(1., factor) if rejected else factor
                break
            h_abs *= max(.2, .9*error**(-.125))
            rejected = True
        for stage in range(13, 16):
            trial[:] = 0.
            for l in range(stage):
                coefficient = A[stage, l]
                if coefficient != 0.:
                    for j in range(42):
                        trial[j] += coefficient*stages[l, j]
            for j in range(42):
                trial[j] = y[j]+h*trial[j]
            derivative_into(epoch+t+C[stage]*h, trial, grid, spline, masses, mu, stages[stage], scratch)
        old[count] = y
        dense[count, 0] = new-y
        dense[count, 1] = h*f-(new-y)
        dense[count, 2] = 2*(new-y)-h*(f_new+f)
        for l in range(4):
            trial[:] = 0.
            for stage in range(16):
                coefficient = D[l, stage]
                if coefficient != 0.:
                    for j in range(42):
                        trial[j] += coefficient*stages[stage, j]
            for j in range(42):
                dense[count, l+3, j] = h*trial[j]
        y, f, t = new, f_new, end
        count += 1
        times[count] = t
    return times[:count+1], old[:count], dense[:count], y, f, h_abs


@njit(cache=True)
def evaluate(times, boundaries, old, dense, direction, components=42):
    answer = np.empty((components, len(times)))
    for k in range(len(times)):
        index = min(max(np.searchsorted(boundaries, direction*times[k], side='left')-1, 0), len(old)-1)
        x = (direction*times[k]-boundaries[index])/(boundaries[index+1]-boundaries[index])
        for j in range(components):
            value = 0.
            for l in range(7):
                value += dense[index, 6-l, j]
                value *= x if l % 2 == 0 else 1-x
            answer[j, k] = value+old[index, j]
    return answer


class CompiledSolution:
    def __init__(self, initial, epoch, target, ephemeris, *, rtol, max_step, deadline=np.inf):
        check_deadline(deadline)
        grid, spline = ephemeris.spline.x, ephemeris.spline.c
        masses, mu = ephemeris.masses, ephemeris.sun_mu
        atol = 1e-12
        f = derivative(epoch, initial, grid, spline, masses, mu)
        h = initial_step(initial, f, epoch, target, rtol, atol, max_step, grid, spline, masses, mu)
        t, y = 0., initial.copy()
        self.direction = 1 if target > 0 else -1
        boundaries, old, dense = [np.array([0.])], [], []
        while self.direction*(target-t) > 0:
            check_deadline(deadline)
            times, prior, coefficients_, y, f, h = integrate_chunk(
                t, y, f, h, target, epoch, rtol, atol, max_step, grid, spline, masses, mu)
            if len(times) < 2:
                raise ValueError('N-body integration made no progress')
            t = times[-1]
            boundaries.append(times[1:])
            old.append(prior)
            dense.append(coefficients_)
        check_deadline(deadline)
        self.boundaries = self.direction*np.concatenate(boundaries)
        self.old = np.concatenate(old)
        self.dense = np.concatenate(dense)

    def __call__(self, times):
        return evaluate(np.atleast_1d(times), self.boundaries, self.old, self.dense, self.direction)

    def states(self, times):
        return evaluate(np.atleast_1d(times), self.boundaries, self.old, self.dense, self.direction, 6)
