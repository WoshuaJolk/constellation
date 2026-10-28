"""Experimental FP64 CUDA residual/Jacobian bank, NOT an orbit verifier.

The universal-variable and analytic-derivative functions are compiled from the
existing CPU Python bodies. One CUDA thread evaluates one state/observation pair.
Inputs stay resident across evaluations; outputs are explicitly copied on request.
Two-body dynamics only: planetary final fitting stays on the CPU.
"""
import math
from types import FunctionType

import numpy as np
from numba import cuda, float64

from . import derivatives, kernels


def _device(function, **bindings):
    body = function.py_func
    namespace = dict(body.__globals__, **bindings)
    clone = FunctionType(body.__code__, namespace, body.__name__)
    return cuda.jit(device=True, fastmath=False)(clone)


_stumpff = _device(kernels.stumpff)
_universal = _device(derivatives.universal_scalars, stumpff=_stumpff)
_geometry = _device(derivatives.geometry_from_values)
ARCSEC, C = kernels.ARCSEC, kernels.C


@cuda.jit(fastmath=False)
def _evaluate(states, case_index, offsets, observers, observed, east, north,
              sigma, residual, jacobian, valid):
    i = cuda.grid(1)
    if i >= len(offsets):
        return
    valid[i] = 0
    for coordinate in range(2):
        residual[i, coordinate] = math.nan
        for k in range(6):
            jacobian[i, coordinate, k] = math.nan
    state = states[case_index[i]]
    j = cuda.local.array((3, 6), float64)
    emitted = offsets[i]
    converged = False
    for _ in range(20):
        values = _universal(state, emitted)
        dx = values[0] - observers[i, 0]
        dy = values[1] - observers[i, 1]
        dz = values[2] - observers[i, 2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if not math.isfinite(distance) or distance < 1e-8:
            return
        slope = 1 + (dx*values[3]+dy*values[4]+dz*values[5])/(C*distance)
        updated = (emitted-(emitted-offsets[i]+distance/C)/slope
                   if math.isfinite(slope) and slope > 1e-8
                   else offsets[i]-distance/C)
        if not math.isfinite(updated):
            return
        if abs(updated-emitted) < 1e-11:
            converged = True
            break
        emitted = updated
    if not converged:
        return
    values = _geometry(state, values, j)
    dx, dy, dz = values[0]-observers[i, 0], values[1]-observers[i, 1], values[2]-observers[i, 2]
    vx, vy, vz = values[3], values[4], values[5]
    distance = math.sqrt(dx*dx+dy*dy+dz*dz)
    denominator = C*distance+dx*vx+dy*vy+dz*vz
    if not math.isfinite(distance) or distance < 1e-8 or abs(denominator) < 1e-8:
        return
    for k in range(6):
        shift = -(dx*j[0, k]+dy*j[1, k]+dz*j[2, k])/denominator
        j[0, k] += vx*shift
        j[1, k] += vy*shift
        j[2, k] += vz*shift
    radial = dx*observed[i, 0]+dy*observed[i, 1]+dz*observed[i, 2]
    factor = ARCSEC/sigma
    for coordinate in range(2):
        # Avoid dynamic selection of whole arrays in device code.
        ex, ey, ez = east[i, 0], east[i, 1], east[i, 2]
        if coordinate == 1:
            ex, ey, ez = north[i, 0], north[i, 1], north[i, 2]
        transverse = dx*ex+dy*ey+dz*ez
        norm = radial*radial+transverse*transverse
        if norm < 1e-20:
            return
        residual[i, coordinate] = math.atan2(transverse, radial)*factor
        for k in range(6):
            du = observed[i, 0]*j[0, k]+observed[i, 1]*j[1, k]+observed[i, 2]*j[2, k]
            de = ex*j[0, k]+ey*j[1, k]+ez*j[2, k]
            jacobian[i, coordinate, k] = (radial*de-transverse*du)/norm*factor
    for coordinate in range(2):
        if not math.isfinite(residual[i, coordinate]):
            return
        for k in range(6):
            if not math.isfinite(jacobian[i, coordinate, k]):
                return
    valid[i] = 1


def validate_bank(bank):
    """Ragged observations, shared state bank; no silent precision conversion."""
    names = ('states', 'offsets', 'observers', 'observed', 'east', 'north')
    arrays = {name: np.asarray(bank[name]) for name in names}
    n = len(arrays['offsets'])
    if arrays['states'].ndim != 2 or arrays['states'].shape[1] != 6:
        raise ValueError('states must be (cases, 6)')
    if arrays['offsets'].shape != (n,):
        raise ValueError('offsets must be a vector')
    for name in names:
        if arrays[name].dtype != np.float64 or not arrays[name].flags.c_contiguous:
            raise ValueError(name+' must be contiguous float64')
    for name in ('observers', 'observed', 'east', 'north'):
        if arrays[name].shape != (n, 3) or not np.isfinite(arrays[name]).all():
            raise ValueError(name+' must be finite (observations, 3)')
    for name in ('observed', 'east', 'north'):
        if not np.allclose(np.linalg.norm(arrays[name], axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError(name+' must contain unit vectors')
    if not np.allclose(np.cross(arrays['observed'], arrays['east']), arrays['north'], atol=1e-10, rtol=0):
        raise ValueError('tangent basis is inconsistent')
    case = np.asarray(bank['case_index'])
    if case.dtype != np.int64 or case.shape != (n,) or not case.flags.c_contiguous:
        raise ValueError('case_index must be contiguous int64')
    if n == 0 or case.min() < 0 or case.max() >= len(arrays['states']):
        raise ValueError('empty bank or invalid case index')
    if not np.array_equal(np.unique(case), np.arange(len(arrays['states']))):
        raise ValueError('Every state must have observations')
    if not np.isfinite(arrays['offsets']).all():
        raise ValueError('nonfinite observation time')
    sigma = float(bank['sigma'])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive')
    return arrays, case, sigma


class ResidentBank:
    def __init__(self, bank, *, allow_simulator=False):
        if cuda.config.ENABLE_CUDASIM and not allow_simulator:
            raise RuntimeError('CUDA simulator is not a GPU benchmark')
        if not cuda.is_available():
            raise RuntimeError('No CUDA device; no CPU fallback or timing claim')
        arrays, case, self.sigma = validate_bank(bank)
        self.n = len(case)
        self.case_index = case.copy()
        self.inputs = [cuda.to_device(arrays['states']), cuda.to_device(case)]
        self.inputs += [cuda.to_device(arrays[k]) for k in ('offsets', 'observers', 'observed', 'east', 'north')]
        self.residual = cuda.device_array((self.n, 2), np.float64)
        self.jacobian = cuda.device_array((self.n, 2, 6), np.float64)
        self.valid = cuda.device_array(self.n, np.uint8)

    def evaluate(self, *, download=True):
        _evaluate[(self.n+127)//128, 128](*self.inputs, self.sigma, self.residual, self.jacobian, self.valid)
        cuda.synchronize()
        if download:
            r, j, valid = self.residual.copy_to_host(), self.jacobian.copy_to_host(), self.valid.copy_to_host()
            self.last_point_valid = valid.copy()
            # CPU residual_jacobian invalidates a whole candidate when any
            # observation fails. Preserve that behavior and the raw point mask.
            bad_cases = np.unique(self.case_index[valid == 0])
            bad = np.isin(self.case_index, bad_cases)
            r[bad], j[bad], valid[bad] = np.nan, np.nan, 0
            return r, j, valid


def cpu_reference(bank):
    """Original analytic CPU implementation, evaluated per candidate."""
    arrays, case, sigma = validate_bank(bank)
    residual = np.full((len(case), 2), np.nan)
    jacobian = np.full((len(case), 2, 6), np.nan)
    valid = np.zeros(len(case), dtype=np.uint8)
    for c in np.unique(case):
        ix = np.flatnonzero(case == c)
        r, j = derivatives.residual_jacobian(arrays['states'][c], arrays['offsets'][ix],
            arrays['observers'][ix], arrays['observed'][ix], arrays['east'][ix], arrays['north'][ix], sigma)
        if np.isfinite(r).all() and np.isfinite(j).all() and not np.all(r == 1e12):
            residual[ix], jacobian[ix], valid[ix] = r.reshape(-1, 2), j.reshape(-1, 2, 6), 1
    return residual, jacobian, valid
