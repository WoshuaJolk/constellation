"""Exact constant-coefficient Kepler dynamics in Sundman time.

Established regularization identity, independently derived in
regularization_attempt.md. It does not remove the orbit-dependent observation
clock or supply an all-orbit image transform.
"""

import numpy as np

from magictools.orbits import GM
from .universal import stumpff


def regularized(states, anomaly, mu=GM["sun"]):
    states, anomaly = np.asarray(states, float), np.asarray(anomaly, float)
    if (states.ndim != 2 or states.shape[1] != 6 or not len(states)
            or anomaly.ndim != 1 or not len(anomaly)
            or not np.all(np.isfinite(states)) or not np.all(np.isfinite(anomaly))
            or not np.isfinite(mu) or mu <= 0):
        raise ValueError("Finite (N,6) states, Sundman samples and positive gravity required")
    r, v = states[:, :3], states[:, 3:]
    radius = np.linalg.norm(r, axis=1)
    if np.any(radius <= 0):
        raise ValueError("Initial radius must be positive")
    rv = np.sum(r*v, axis=1)
    energy2 = np.sum(v*v, axis=1)-2*mu/radius
    eccentricity = ((np.sum(v*v, axis=1)-mu/radius)[:, None]*r-rv[:, None]*v)/mu
    s = anomaly[None]
    lam = energy2[:, None]
    c, h = stumpff(-lam*s*s)
    cosine = 1+lam*s*s*c
    sine_over_frequency = s+lam*s*s*s*h
    forcing = energy2[:, None]*r-mu*eccentricity
    initial_derivative = radius[:, None]*v
    position = (r[:, None]+initial_derivative[:, None]*s[..., None]
                +forcing[:, None]*(s*s*c)[..., None]
                +initial_derivative[:, None]*(lam*s*s*s*h)[..., None])
    radius_s = (radius[:, None]+rv[:, None]*s
                +(lam*radius[:, None]+mu)*s*s*c+lam*rv[:, None]*s*s*s*h)
    time = radius[:, None]*s+rv[:, None]*s*s*c+(lam*radius[:, None]+mu)*s*s*s*h
    derivative = forcing[:, None]*sine_over_frequency[..., None]+initial_derivative[:, None]*cosine[..., None]
    if np.any(radius_s <= 0) or not np.all(np.isfinite(position)):
        raise ValueError("Regularized samples cross a collision or numerical range")
    velocity = derivative/radius_s[..., None]
    return np.concatenate((position, velocity), axis=2), time, {
        "twice_energy": energy2, "eccentricity_vector": eccentricity,
        "dt_ds": radius_s, "initial_clock_derivatives": np.column_stack(
            (radius, rv, energy2*radius+mu)),
    }
