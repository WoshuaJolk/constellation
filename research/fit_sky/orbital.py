"""FitSky adapter for Kepler's all-conic universal-variable propagator.

Numerical forward evaluation only: no orbit shortlist or posterior certificate.
The gravitational/light-speed constants are those of FitSky's interval model,
not the (slightly different) default constants in the universal propagator.
"""

import numpy as np

from research.kepler_transform.universal import propagate
from research.detection_limit.interval_orbits import MU, C_AU_DAY


STATE_PARAMETERS = ("rx", "ry", "rz", "vx", "vy", "vz")


class OrbitProjection:
    """Reception observers and tangent bases in a common heliocentric frame.

    Times are TDB days from the Cartesian-state epoch; positions are AU and
    velocities AU/day. Input observers are fixed at reception, including any
    site displacement. Exposure integration/trailing is not modeled.
    """

    def __init__(self, exposures):
        self.exposures = exposures
        for e in exposures:
            for key in ("observer", "center", "east", "north"):
                value = np.asarray(e.get(key), float)
                if value.shape != (3,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"Kepler exposure requires finite {key} vector")
            basis = np.array([e["center"], e["east"], e["north"]], float)
            if (not np.allclose(basis@basis.T, np.eye(3), rtol=0, atol=1e-10)
                    or np.linalg.det(basis) <= 0):
                raise ValueError("Kepler tangent basis must be right-handed and orthonormal")
            for key, default in (("pixel_scale_arcsec", np.nan), ("light_time_max", 1.)):
                value = e.get(key, default)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"Kepler exposure requires positive finite {key}")

    def coordinates(self, states, exposure):
        """Return tangent-plane pixels before affine/shared calibration.

        A numerical or projection failure aborts evaluation. It must not be
        converted into zero likelihood, which would silently remove prior mass.
        """
        states = np.asarray(states, float)
        e = self.exposures[exposure]
        time, observer = float(e["time"]), np.asarray(e["observer"], float)
        # Iterate delay rather than absolute emission time to avoid cancellation
        # when reception is far from the state epoch.
        delay = np.zeros((len(states), 1))
        for _ in range(40):
            emitted = propagate(states, time-delay, mu=float(MU))
            direction = emitted[:, 0, :3]-observer
            distance = np.linalg.norm(direction, axis=1)
            if (np.any(distance < 1e-8)
                    or np.any(np.linalg.norm(emitted[:, 0, 3:], axis=1) >= float(C_AU_DAY))):
                raise ValueError("Kepler light-time geometry is singular or non-subluminal")
            updated = distance[:, None]/float(C_AU_DAY)
            if np.max(np.abs(updated-delay)) < 1e-12:
                delay = updated
                break
            delay = updated
        else:
            raise ArithmeticError("Kepler light-time iteration did not converge")
        if np.any(delay > e.get("light_time_max", 1.)):
            raise ValueError("Kepler delay exceeds light_time_max; prior mass was not discarded")
        direction = propagate(states, time-delay, mu=float(MU))[:, 0, :3]-observer
        denominator = direction@e["center"]
        if np.any(denominator <= 0):
            raise ValueError("Kepler orbit crosses tangent horizon; prior mass was not discarded")
        scale = float(e["pixel_scale_arcsec"])*np.pi/648000
        result = np.column_stack((direction@e["east"], direction@e["north"]))/(denominator[:, None]*scale)
        if not np.all(np.isfinite(result)):
            raise ArithmeticError("Nonfinite Kepler projection")
        return result
