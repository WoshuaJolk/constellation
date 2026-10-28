"""Experimental range search with a linear velocity solve, for initialization only.

Hold heliocentric radius fixed in the acceleration coefficient to obtain f/g
functions. This approximation never accepts an orbit: exact light-time/two-body
correction and every observation's residual limits remain mandatory afterward.
"""
import math
import numpy as np
from .kernels import MU, ARCSEC, C, njit


@njit(cache=True, inline='always')
def dot3(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


@njit(cache=True)
def solve3(matrix, rhs):
    """Avoid a LAPACK dispatch for each tiny regularized velocity fit."""
    a = math.sqrt(matrix[0, 0]); b = matrix[1, 0]/a; c = matrix[2, 0]/a
    d2 = matrix[1, 1]-b*b
    if d2 <= 0: return np.linalg.solve(matrix, rhs)
    d = math.sqrt(d2); e = (matrix[2, 1]-c*b)/d
    f2 = matrix[2, 2]-c*c-e*e
    if f2 <= 0: return np.linalg.solve(matrix, rhs)
    f = math.sqrt(f2)
    y0 = rhs[0]/a; y1 = (rhs[1]-b*y0)/d; y2 = (rhs[2]-c*y0-e*y1)/f
    x2 = y2/f; x1 = (y1-e*x2)/d; x0 = (y0-b*x1-c*x2)/a
    return np.array([x0, x1, x2])


@njit(cache=True)
def gauss_starts(t, u, observer):
    """Same positive-root Gauss construction, with setup inside compiled code."""
    indices = np.array([0, len(t)//2, len(t)-1])
    l = u[indices].T.copy(); r = observer[indices].T.copy()
    tau1 = t[indices[0]]-t[indices[1]]; tau3 = t[indices[2]]-t[indices[1]]
    tau = tau3-tau1
    result = [(np.zeros(6), 0.)]; result.pop()
    if min(-tau1, tau3) < 1e-4 or abs(np.linalg.det(l)) < 1e-15:
        return result
    inv = np.linalg.inv(l); p = inv@r
    a1 = tau3/tau; a3 = -tau1/tau
    b1 = a1*(tau*tau-tau3*tau3)/6; b3 = a3*(tau*tau-tau1*tau1)/6
    a = p[1, 0]*a1-p[1, 1]+p[1, 2]*a3
    b = MU*(p[1, 0]*b1+p[1, 2]*b3)
    e = dot3(l[:, 1], r[:, 1]); r2 = dot3(r[:, 1], r[:, 1])
    coefficients = np.array([1., 0., -(a*a+2*a*e+r2), 0., 0., -2*b*(a+e), 0., 0., -b*b]).astype(np.complex128)
    for root in np.roots(coefficients):
        if abs(root.imag) > 1e-7 or root.real <= .03: continue
        radius = root.real
        c1 = a1+MU*b1/radius**3; c3 = a3+MU*b3/radius**3
        rho = (inv@(r@np.array([-c1, 1., -c3])))/np.array([c1, -1., c3])
        if np.any(rho <= 0): continue
        xyz = l*rho+r
        f1 = 1-.5*MU*tau1*tau1/radius**3; f3 = 1-.5*MU*tau3*tau3/radius**3
        g1 = tau1-MU*tau1**3/(6*radius**3); g3 = tau3-MU*tau3**3/(6*radius**3)
        denominator = f1*g3-f3*g1
        if abs(denominator) > 1e-12:
            v = (f1*xyz[:, 2]-f3*xyz[:, 0])/denominator
            result.append((np.concatenate((xyz[:, 1], v)), float(t[indices[1]])))
    return result


@njit(cache=True)
def trial(log_range, offsets, observers, u, east, north, middle):
    rho = math.exp(log_range)
    r = observers[middle]+rho*u[middle]
    radius = math.sqrt(dot3(r, r))
    if radius < .005:
        return 1e30, np.zeros(6)
    w = math.sqrt(MU/radius**3)
    f = np.cos(w*offsets); g = np.sin(w*offsets)/w
    prior = (observers[-1]-observers[0]+rho*(u[-1]-u[0]))/(offsets[-1]-offsets[0])
    normal = np.zeros((3, 3)); rhs = np.zeros(3)
    for i in range(len(offsets)):
        target = observers[i]-f[i]*r-g[i]*prior
        for coordinate in range(2):
            tangent = east[i] if coordinate == 0 else north[i]
            b = dot3(tangent, target)
            for j in range(3):
                a = g[i]*tangent[j]
                rhs[j] += a*b
                for k in range(3):
                    normal[j, k] += a*g[i]*tangent[k]
    ridge = max(np.trace(normal)*1e-12, 1e-30)
    for j in range(3): normal[j, j] += ridge
    velocity = prior+solve3(normal, rhs)
    cost = 0.
    for i in range(len(offsets)):
        delta = f[i]*r+g[i]*velocity-observers[i]
        radial = dot3(delta, u[i])
        if radial <= 0 or not math.isfinite(radial):
            return 1e30, np.concatenate((r, velocity))
        for tangent in (east[i], north[i]):
            angle = math.atan2(dot3(delta, tangent), radial)*ARCSEC
            cost += angle*angle
    return cost/len(offsets), np.concatenate((r, velocity))


@njit(cache=True)
def range_starts(t, observers, u, east, north, gate_arcsec=100.):
    middle = len(t)//2; offsets = t-t[middle]
    ranges = np.array([.005, .01, .03, .1, .3, 1., 3., 10., 30., 100., 300., 1000., 3000.])
    grid = np.log(ranges); scores = np.empty(len(grid))
    for i in range(len(grid)):
        scores[i], _ = trial(grid[i], offsets, observers, u, east, north, middle)
    states = np.empty((len(grid), 6)); epochs = np.empty(len(grid)); costs = np.empty(len(grid))
    used = 0; ratio = (math.sqrt(5.)-1)/2
    for i in range(len(grid)):
        if i > 0 and scores[i] > scores[i-1]: continue
        if i+1 < len(grid) and scores[i] > scores[i+1]: continue
        if scores[i] >= 1e29: continue
        lo = grid[max(0, i-1)]; hi = grid[min(len(grid)-1, i+1)]
        a = hi-ratio*(hi-lo); b = lo+ratio*(hi-lo)
        ca, _ = trial(a, offsets, observers, u, east, north, middle)
        cb, _ = trial(b, offsets, observers, u, east, north, middle)
        for _ in range(28):
            if ca < cb:
                hi = b; b = a; cb = ca; a = hi-ratio*(hi-lo)
                ca, _ = trial(a, offsets, observers, u, east, north, middle)
            else:
                lo = a; a = b; ca = cb; b = lo+ratio*(hi-lo)
                cb, _ = trial(b, offsets, observers, u, east, north, middle)
        best = a if ca < cb else b
        if scores[i] < min(ca, cb): best = grid[i]
        cost, state = trial(best, offsets, observers, u, east, north, middle)
        if cost <= gate_arcsec**2:
            states[used] = state
            # Approximate a constant range delay only for the initializer.
            # The exact final fit solves each observation's light time.
            epochs[used] = t[middle]-math.exp(best)/C
            costs[used] = math.sqrt(cost); used += 1
    return states[:used], epochs[:used], costs[:used]
