"""Experimental resident FP64 common-epoch proposal bank, not a final fitter.

One CUDA thread per (range/radial hypothesis, tracklet). Geometry is uploaded
once and reused across epochs. This mirrors distant_epoch_graph.projected's
approximate attribution model; it does not add light time or planetary forces.
CPU parity and real-device throughput must pass before production integration.
"""
import math
import time
import numpy as np
from numba import cuda, float64
from magictools.itf.gpu import _universal
from magictools.itf.kernels import MU


@cuda.jit(fastmath=False)
def _project(t, u, du, observer, velocity, ref, earth, hypotheses, out):
    i, h = cuda.grid(2)
    if i >= len(t) or h >= len(hypotheses):
        return
    for k in range(3):
        out[h, i, k] = math.nan
    dt = ref-t[i]
    distance, fraction = hypotheses[h, 0], hypotheses[h, 1]
    radial = fraction*math.sqrt(2*MU/distance)
    rho = distance-radial*dt
    if abs(dt) > 365 or rho <= .001:
        return
    state = cuda.local.array(6, float64)
    for k in range(3):
        state[k] = observer[i, k]+rho*u[i, k]
        state[k+3] = velocity[i, k]+rho*du[i, k]+radial*u[i, k]
    values = _universal(state, dt)
    dx, dy, dz = values[0]-earth[0], values[1]-earth[1], values[2]-earth[2]
    norm = math.sqrt(dx*dx+dy*dy+dz*dz)
    if math.isfinite(norm) and norm > .001:
        out[h, i, 0], out[h, i, 1], out[h, i, 2] = dx/norm, dy/norm, dz/norm


def _array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
        raise ValueError(name+' must be finite float64 with shape '+str(shape))
    return np.ascontiguousarray(a)


class ProjectionBank:
    def __init__(self, t, u, du, observers, velocities, *, allow_simulator=False):
        if cuda.config.ENABLE_CUDASIM and not allow_simulator:
            raise RuntimeError('CUDA simulator is not a hardware benchmark')
        t = np.asarray(t)
        if t.ndim != 1:
            raise ValueError('Times must be a vector')
        self.n = len(t)
        arrays = [_array(t, (self.n,), 'times')]
        arrays += [_array(a, (self.n, 3), name) for a, name in
                   zip((u, du, observers, velocities), ('directions', 'motion', 'observer', 'velocity'))]
        if not np.allclose(np.linalg.norm(arrays[1], axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError('Directions must be unit vectors')
        started = time.perf_counter()
        self.arrays = [cuda.to_device(a) for a in arrays]
        cuda.synchronize()
        self.upload_seconds = time.perf_counter()-started
        self.input_bytes = sum(a.nbytes for a in arrays)

    def project(self, ref, earth, hypotheses, *, max_output_bytes=256*1024**2):
        if not np.isfinite(ref):
            raise ValueError('Reference epoch must be finite')
        earth = _array(earth, (3,), 'earth')
        h = np.asarray(hypotheses)
        if h.ndim != 2 or h.shape[1] != 2:
            raise ValueError('Hypotheses must have distance/fraction columns')
        h = _array(h, h.shape, 'hypotheses')
        if np.any(h[:, 0] <= 0):
            raise ValueError('Distances must be positive')
        shape = (len(h), self.n, 3)
        size = math.prod(shape)*8
        if size > max_output_bytes:
            raise MemoryError('Projection bank exceeds explicit output bound; tile hypotheses')
        started = time.perf_counter()
        out = cuda.device_array(shape, dtype=np.float64)
        if self.n and len(h):
            _project[((self.n+127)//128, len(h)), (128, 1)](
                *self.arrays, float(ref), cuda.to_device(earth), cuda.to_device(h), out)
        cuda.synchronize()
        evaluated = time.perf_counter()-started
        started = time.perf_counter()
        result = out.copy_to_host()
        downloaded = time.perf_counter()-started
        return result, dict(evaluate_seconds=evaluated, download_seconds=downloaded,
            input_upload_seconds=self.upload_seconds, input_bytes=self.input_bytes,
            output_bytes=size, hypotheses=len(h), tracklets=self.n,
            valid_projections=int(np.isfinite(result).all(axis=2).sum()),
            simulated=bool(cuda.config.ENABLE_CUDASIM))
