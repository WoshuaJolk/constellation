"""Compiled small-array bookkeeping; float64, no fast-math or altered cuts.

These functions do not solve or choose orbital branches. The Python fitter keeps
seed ordering, acceptance limits, duplicate replacement and output semantics.
"""
import math
import numpy as np
from numba import njit
from .derivatives import state_geometry
from .kernels import MU, position


@njit(cache=True, nogil=True)
def element_values(state):
    r, v = state[:3], state[3:]
    radius = np.linalg.norm(r)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    if hn < 1e-14 or radius < .01:
        return np.full(6, np.nan)
    eccentricity = np.linalg.norm(np.cross(v, h)/MU-r/radius)
    energy = np.dot(v, v)/2-MU/radius
    obl = np.deg2rad(23.439291111)
    hz = -np.sin(obl)*h[1]+np.cos(obl)*h[2]
    inclination = np.rad2deg(np.arccos(min(1., max(-1., hz/hn))))
    return np.array([eccentricity, hn*hn/(MU*(1+eccentricity)),
                     -MU/(2*energy) if abs(energy) > 1e-16 else np.nan,
                     inclination, energy, hn])


@njit(cache=True, nogil=True)
def uncertainty_values(state, jacobian, epoch_offset, observer):
    # Same column-scaled SVD as the reference implementation. Never invert the
    # squared normal matrix or shrink the assumed error using fitted residuals.
    scales = np.empty(6)
    for k in range(6):
        scales[k] = max(np.linalg.norm(jacobian[:, k]), 1e-30)
    _, singular, vt = np.linalg.svd(jacobian/scales, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*max(jacobian.shape)*np.finfo(np.float64).eps))
    values = np.full(7, np.nan)
    values[0] = rank
    if rank < 6:
        return values
    factor = (vt.T/singular)/scales[:, None]
    p, _, position_jac = state_geometry(state, epoch_offset)
    delta = p-observer
    distance = np.linalg.norm(delta)
    range_sigma = np.linalg.norm((delta/distance)@position_jac@factor)
    r, v = state[:3], state[3:]
    radius = np.linalg.norm(r)
    energy = np.dot(v, v)/2-MU/radius
    de = np.empty(6)
    de[:3] = MU*r/radius**3
    de[3:] = v
    energy_sigma = np.linalg.norm(de@factor)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    obl = np.deg2rad(23.439291111)
    pole = np.array([0., -np.sin(obl), np.cos(obl)])
    cosine = min(1., max(-1., h@pole/hn))
    sine = np.sqrt(max(0., 1-cosine*cosine))
    inclination_sigma = np.nan
    if sine > 1e-10:
        dh = np.empty((3, 6))
        eye = np.eye(3)
        for k in range(3):
            dh[:, k] = np.cross(eye[k], v)
            dh[:, k+3] = np.cross(r, eye[k])
        gradient = -(pole-cosine*h/hn)/(hn*sine)*180/np.pi
        inclination_sigma = np.linalg.norm(gradient@dh@factor)
    values[1] = singular[0]/singular[-1]
    values[2] = range_sigma
    values[3] = inclination_sigma
    values[4] = energy_sigma
    values[5] = abs(energy) > 3*energy_sigma
    values[6] = range_sigma > .5*distance or np.isnan(inclination_sigma) or inclination_sigma > 10.
    return values


@njit(cache=True, nogil=True)
def candidate_summary(state, residual, sigma, rms_limit, labels, canonical_offset, observer):
    """Unchanged residual/per-tracklet/physical cuts, then canonical state."""
    count = len(labels)
    chi2 = np.dot(residual, residual)
    rms = sigma*math.sqrt(chi2/count)
    empty = np.full(3, np.nan)
    elements = np.full(6, np.nan)
    if rms > rms_limit:
        return False, chi2, rms, np.nan, np.nan, empty, empty, elements
    per_observation = np.empty(count)
    for k in range(count):
        per_observation[k] = sigma*math.sqrt(residual[2*k]**2+residual[2*k+1]**2)
    maximum = np.max(per_observation)
    if maximum > max(3*sigma, 2*rms_limit):
        return False, chi2, rms, maximum, np.nan, empty, empty, elements
    for label in np.unique(labels):
        if np.sqrt(np.mean(per_observation[labels == label]**2)) > rms_limit:
            return False, chi2, rms, maximum, np.nan, empty, empty, elements
    elements = element_values(state)
    if not np.isfinite(elements[5]) or elements[1] < .005:
        return False, chi2, rms, maximum, np.nan, empty, empty, elements
    canonical_position = position(state, canonical_offset)
    distance = np.linalg.norm(canonical_position-observer)
    canonical_velocity = (position(state, canonical_offset+.001)-position(state, canonical_offset-.001))/.002
    return True, chi2, rms, maximum, distance, canonical_position, canonical_velocity, elements


@njit(cache=True, nogil=True)
def duplicate_index(positions, velocities, length, position_value, velocity_value, distance):
    for k in range(length):
        if (np.linalg.norm(position_value-positions[k]) < max(1e-5, distance*1e-3)
                and np.linalg.norm(velocity_value-velocities[k]) < 1e-6):
            return k
    return -1
