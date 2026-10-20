"""Explicit, smooth Galactic potentials and vectorized stellar trajectories.

Internal units: pc, Myr, solar masses. Cartesian axes follow Astropy Galactocentric.
These potentials are user-specified models, not a fitted Milky Way mass model.
"""

import numpy as np
from astropy import constants as const, units as u

from .io import _fields

KMS_TO_PC_MYR = (1 * u.km / u.s).to_value(u.pc / u.Myr)
G_PC_MYR = const.G.to_value(u.pc**3 / u.Msun / u.Myr**2)


class GalacticPotential:
    """Sum of Plummer bulges, Miyamoto-Nagai disks, and spherical log halos."""

    def __init__(self, specification):
        _fields(specification, ["components"])
        if not isinstance(specification["components"], list):
            raise ValueError("Potential components must be a list; [] means free motion")
        self.components = []
        for component in specification["components"]:
            kind = component.get("type") if isinstance(component, dict) else None
            required = {
                "plummer": ["mass_msun", "scale_pc"],
                "miyamoto_nagai": ["mass_msun", "a_pc", "b_pc"],
                "log_halo": ["speed_kms", "core_pc"],
            }.get(kind)
            if required is None:
                raise ValueError("Unknown Galactic potential component")
            _fields(component, ["type", *required])
            if any(not np.isfinite(component[key]) or component[key] <= 0 for key in required):
                raise ValueError("Potential masses, lengths, and speeds must be finite and positive")
            self.components.append(component.copy())

    def acceleration(self, positions):
        r = np.asarray(positions, dtype=float)
        if r.shape[-1:] != (3,) or not np.all(np.isfinite(r)):
            raise ValueError("Positions must be finite vectors with last dimension three")
        acceleration = np.zeros_like(r)
        r2 = np.sum(r**2, axis=-1)
        for c in self.components:
            if c["type"] == "plummer":
                acceleration -= G_PC_MYR * c["mass_msun"] * r / (r2 + c["scale_pc"]**2)[..., None]**1.5
            elif c["type"] == "miyamoto_nagai":
                z_soft = np.sqrt(r[..., 2]**2 + c["b_pc"]**2)
                az = c["a_pc"] + z_soft
                denominator = (r[..., 0]**2 + r[..., 1]**2 + az**2)**1.5
                direction = r.copy()
                direction[..., 2] *= az / z_soft
                acceleration -= G_PC_MYR * c["mass_msun"] * direction / denominator[..., None]
            else:
                speed = c["speed_kms"] * KMS_TO_PC_MYR
                acceleration -= speed**2 * r / (r2 + c["core_pc"]**2)[..., None]
        return acceleration

    def value(self, positions):
        r = np.asarray(positions, dtype=float)
        r2 = np.sum(r**2, axis=-1)
        value = np.zeros(r.shape[:-1])
        for c in self.components:
            if c["type"] == "plummer":
                value -= G_PC_MYR * c["mass_msun"] / np.sqrt(r2 + c["scale_pc"]**2)
            elif c["type"] == "miyamoto_nagai":
                az = c["a_pc"] + np.sqrt(r[..., 2]**2 + c["b_pc"]**2)
                value -= G_PC_MYR * c["mass_msun"] / np.sqrt(r[..., 0]**2 + r[..., 1]**2 + az**2)
            else:
                value += 0.5 * (c["speed_kms"] * KMS_TO_PC_MYR)**2 * np.log1p(r2 / c["core_pc"]**2)
        return value


def integrate_stars(states_pc_kms, times_myr, potential, max_step_myr=0.02):
    """Kick-drift-kick leapfrog from t=0 to monotone signed times, including zero.

    Any leading state dimensions are supported. Returns (times, ..., 6), in
    pc and km/s. Output times are exact integration endpoints, not interpolated.
    """
    states = np.asarray(states_pc_kms, dtype=float)
    times = np.asarray(times_myr, dtype=float)
    if states.shape[-1:] != (6,) or not np.all(np.isfinite(states)):
        raise ValueError("Stellar states must be finite six-vectors in pc and km/s")
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or times[0] != 0:
        raise ValueError("Integration times must be a finite vector starting at zero")
    if len(times) > 1 and not (np.all(np.diff(times) > 0) or np.all(np.diff(times) < 0)):
        raise ValueError("Integration times must be strictly monotone")
    if not np.isfinite(max_step_myr) or max_step_myr <= 0:
        raise ValueError("max_step_myr must be finite and positive")
    # Avoid an extra substep when a nominally integral ratio is a few ulps high.
    n_steps = np.maximum(1, np.ceil(np.abs(np.diff(times)) / max_step_myr - 1e-12)).astype(int)
    if n_steps.sum() > 100000:
        raise ValueError("Integration exceeds 100,000 steps; shorten the interval")
    if states.size * len(times) > 30000000:
        raise ValueError("Trajectory output exceeds 30 million values; reduce catalogue, samples, or times")
    position, velocity = states[..., :3].copy(), states[..., 3:].copy() * KMS_TO_PC_MYR
    output = np.empty((len(times),) + states.shape)
    output[0] = states
    acceleration = potential.acceleration(position)
    for index, steps in enumerate(n_steps, 1):
        dt = (times[index] - times[index - 1]) / steps
        for _ in range(steps):
            velocity += dt * acceleration / 2
            position += dt * velocity
            acceleration = potential.acceleration(position)
            velocity += dt * acceleration / 2
        output[index, ..., :3] = position
        output[index, ..., 3:] = velocity / KMS_TO_PC_MYR
    if not np.all(np.isfinite(output)):
        raise ValueError("Trajectory integration produced nonfinite values")
    return output


def energy_diagnostics(paths, potential):
    kinetic = 0.5 * np.sum((paths[..., 3:] * KMS_TO_PC_MYR)**2, axis=-1)
    phi = potential.value(paths[..., :3])
    energy = kinetic + phi
    scale = np.maximum(kinetic[0] + np.abs(phi[0]), np.finfo(float).tiny)
    errors = np.abs(energy - energy[0]) / scale
    return {"max_scaled_energy_drift": float(errors.max()),
            "p95_scaled_energy_drift": float(np.quantile(errors, 0.95)),
            "scale": "initial kinetic energy + absolute initial potential, per trajectory"}
