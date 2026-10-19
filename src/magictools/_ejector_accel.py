"""Optional compiled version of the fit integrator; identical KDK equations.

No fastmath or relaxed precision. The NumPy path remains the default fallback
when the optional ``accelerate`` extra is unavailable.
"""
import numpy as np

from .galactic import G_PC_MYR, KMS_TO_PC_MYR

try:
    from numba import njit
except ImportError:
    njit = None


def pack_potential(potential):
    rows = []
    for c in potential.components:
        if c['type'] == 'plummer':
            rows.append([0, G_PC_MYR*c['mass_msun'], c['scale_pc'], 0])
        elif c['type'] == 'miyamoto_nagai':
            rows.append([1, G_PC_MYR*c['mass_msun'], c['a_pc'], c['b_pc']])
        else:
            rows.append([2, (KMS_TO_PC_MYR*c['speed_kms'])**2, c['core_pc'], 0])
    return np.array(rows, dtype=float).reshape(-1, 4)


def _force(x, y, z, components):
    ax, ay, az = 0., 0., 0.
    for kind, amplitude, a, b in components:
        direction_z = z
        if kind == 0:
            denominator = (x*x+y*y+z*z+a*a)**1.5
        elif kind == 1:
            soft = np.sqrt(z*z+b*b)
            height = a+soft
            denominator = (x*x+y*y+height*height)**1.5
            direction_z *= height/soft
        else:
            denominator = x*x+y*y+z*z+a*a
        ax -= amplitude*x/denominator
        ay -= amplitude*y/denominator
        az -= amplitude*direction_z/denominator
    return ax, ay, az


def _positions(states, indices, ages, steps, components):
    output = np.empty((len(states), len(indices), 3))
    for i in range(len(states)):
        h = -ages[i]/steps
        for j in range(len(indices)):
            s = states[i, indices[j]]
            x, y, z = s[0], s[1], s[2]
            vx, vy, vz = s[3]*KMS_TO_PC_MYR, s[4]*KMS_TO_PC_MYR, s[5]*KMS_TO_PC_MYR
            ax, ay, az = _force(x, y, z, components)
            for _ in range(steps):
                vx += h*ax/2; vy += h*ay/2; vz += h*az/2
                x += h*vx; y += h*vy; z += h*vz
                ax, ay, az = _force(x, y, z, components)
                vx += h*ax/2; vy += h*ay/2; vz += h*az/2
            output[i, j, 0], output[i, j, 1], output[i, j, 2] = x, y, z
    return output


if njit is not None:
    _force = njit(cache=True)(_force)
    compiled_positions = njit(cache=True)(_positions)
else:
    compiled_positions = None
