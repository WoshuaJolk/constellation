"""Research reference: safeguarded universal-variable two-body propagation.

All conic energies share one equation. This is a standard method, not a novel
result. No collisions, perturbations, or formal integration certificate.
"""

import numpy as np

from magictools.orbits import GM, C_AU_DAY


def stumpff(z):
    z = np.asarray(z, dtype=float)
    c, s = np.empty_like(z), np.empty_like(z)
    small, pos, neg = np.abs(z) < 1e-3, z >= 1e-3, z <= -1e-3
    x = z[small]
    c[small] = .5-x/24+x*x/720-x**3/40320+x**4/3628800
    s[small] = 1/6-x/120+x*x/5040-x**3/362880+x**4/39916800
    x = np.sqrt(z[pos])
    c[pos], s[pos] = 2*np.sin(x/2)**2/z[pos], (x-np.sin(x))/x**3
    x = np.sqrt(-z[neg])
    if np.any(x > 600):
        raise ValueError("Universal anomaly outside stable hyperbolic range")
    c[neg], s[neg] = 2*np.sinh(x/2)**2/(-z[neg]), (np.sinh(x)-x)/x**3
    return c, s


def propagate(states, offsets, mu=GM["sun"]):
    states, offsets = np.asarray(states, float), np.asarray(offsets, float)
    if states.ndim != 2 or states.shape[1] != 6 or not len(states) or not np.all(np.isfinite(states)):
        raise ValueError("Finite states must have shape (N,6)")
    if offsets.ndim == 1:
        offsets = np.broadcast_to(offsets, (len(states), len(offsets)))
    if offsets.ndim != 2 or offsets.shape[0] != len(states) or not offsets.size or not np.all(np.isfinite(offsets)):
        raise ValueError("Offsets must be finite (T,) or (N,T)")
    r0 = np.linalg.norm(states[:, :3], axis=1)[:, None]
    if np.any(r0 < 1e-5) or not np.isfinite(mu) or mu <= 0:
        raise ValueError("Positive mu and nonzero heliocentric radius required")
    rv = np.sum(states[:, :3]*states[:, 3:], axis=1)[:, None]/np.sqrt(mu)
    alpha = 2/r0-np.sum(states[:, 3:]**2, axis=1)[:, None]/mu

    def equation(chi):
        z = alpha*chi**2
        c, s = stumpff(z)
        f = rv*chi**2*c + (1-alpha*r0)*chi**3*s + r0*chi-np.sqrt(mu)*offsets
        derivative = rv*chi*(1-z*s)+(1-alpha*r0)*chi**2*c+r0
        return f, derivative, c, s

    edge = np.maximum(np.sqrt(mu)*np.abs(offsets)/r0, 1e-10)*np.sign(offsets)
    for _ in range(80):
        value = equation(edge)[0]
        outside = value*np.sign(offsets) < 0
        if not np.any(outside):
            break
        edge = np.where(outside, 2*edge, edge)
    else:
        raise ValueError("Could not bracket universal anomaly")
    low, high = np.minimum(0., edge), np.maximum(0., edge)
    chi = (low+high)/2
    active = offsets != 0
    for _ in range(120):
        f, df, _, _ = equation(chi)
        if np.any((df <= 0) & active):
            raise ValueError("Collision or invalid universal radial derivative")
        low = np.where((f < 0) & active, chi, low)
        high = np.where((f >= 0) & active, chi, high)
        step = f/df
        converged = np.abs(step) <= 3e-14*np.maximum(1., np.abs(chi))
        active &= ~converged
        if not np.any(active):
            break
        proposed = chi-step
        proposed = np.where((proposed > low) & (proposed < high), proposed, (low+high)/2)
        chi = np.where(active, proposed, chi)
    else:
        raise ValueError("Universal anomaly did not converge")
    _, _, c, s = equation(chi)
    f = 1-chi**2*c/r0
    g = offsets-chi**3*s/np.sqrt(mu)
    position = f[..., None]*states[:, None, :3]+g[..., None]*states[:, None, 3:]
    radius = np.linalg.norm(position, axis=2)
    if np.any(radius < 1e-5) or not np.all(np.isfinite(position)):
        raise ValueError("Singular/nonfinite propagated position")
    fdot = np.sqrt(mu)/(radius*r0)*(alpha*chi**3*s-chi)
    gdot = 1-chi**2*c/radius
    velocity = fdot[..., None]*states[:, None, :3]+gdot[..., None]*states[:, None, 3:]
    return np.concatenate((position, velocity), axis=2)


def apparent(states, offsets, observers):
    """Reception offsets/heliocentric observers; return emission states/times."""
    offsets, observers = np.asarray(offsets, float), np.asarray(observers, float)
    if offsets.ndim != 1 or observers.shape != (len(offsets), 3) or not np.all(np.isfinite(observers)):
        raise ValueError("One finite observer per reception time required")
    emission = np.broadcast_to(offsets, (len(states), len(offsets))).copy()
    for _ in range(40):
        state = propagate(states, emission)
        distance = np.linalg.norm(state[..., :3]-observers[None], axis=2)
        if np.any(distance < 1e-8):
            raise ValueError("Object too near observer")
        update = offsets-distance/C_AU_DAY
        if np.max(np.abs(update-emission)) < 1e-11:
            emission = update
            break
        emission = update
    else:
        raise ValueError("Light time did not converge")
    return propagate(states, emission), emission
