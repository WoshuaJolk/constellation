"""Resident exact nightly cone retrieval; CPU-equivalent chord predicate.

No nearest-neighbor cap or fixed hit buffer: count first, bound allocation, then
fill. Overflow is an explicit failure, never a clipped successful search.
This accelerates a spatial screening primitive, not orbit uncertainty inference.
"""
import math
import time
import numpy as np
from numba import cuda


@cuda.jit(device=True, fastmath=False)
def _bounds(points, starts, ends, night, x, radius):
    lo, hi = starts[night], ends[night]
    while lo < hi:
        mid = (lo+hi)//2
        if points[mid, 0] < x-radius:
            lo = mid+1
        else:
            hi = mid
    first = lo
    hi = ends[night]
    while lo < hi:
        mid = (lo+hi)//2
        if points[mid, 0] <= x+radius:
            lo = mid+1
        else:
            hi = mid
    return first, lo


@cuda.jit(fastmath=False)
def _count(points, starts, ends, nights, predictions, radii, counts):
    q = cuda.grid(1)
    if q >= len(nights):
        return
    x, y, z = predictions[q]
    radius = radii[q]
    first, last = _bounds(points, starts, ends, nights[q], x, radius)
    total = 0
    for i in range(first, last):
        dx, dy, dz = points[i,0]-x, points[i,1]-y, points[i,2]-z
        if dx*dx+dy*dy+dz*dz <= radius*radius:
            total += 1
    counts[q] = total


@cuda.jit(fastmath=False)
def _fill(points, starts, ends, nights, predictions, radii, offsets, indices, queries, distances):
    q = cuda.grid(1)
    if q >= len(nights):
        return
    x, y, z = predictions[q]
    radius = radii[q]
    first, last = _bounds(points, starts, ends, nights[q], x, radius)
    out = offsets[q]
    for i in range(first, last):
        dx, dy, dz = points[i,0]-x, points[i,1]-y, points[i,2]-z
        squared = dx*dx+dy*dy+dz*dz
        if squared <= radius*radius:
            indices[out], queries[out], distances[out] = i, q, math.sqrt(squared)
            out += 1


def _require(value, dtype, shape, name):
    a = np.asarray(value)
    if a.dtype != np.dtype(dtype) or a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Invalid '+name)
    return np.ascontiguousarray(a)


class NightQueryBank:
    def __init__(self, points, starts, ends, *, allow_simulator=False):
        if cuda.config.ENABLE_CUDASIM and not allow_simulator:
            raise RuntimeError('Simulator timing is not a hardware benchmark')
        n = len(points); nights = len(starts)
        points = _require(points, np.float64, (n,3), 'points')
        starts = _require(starts, np.int64, (nights,), 'starts')
        ends = _require(ends, np.int64, (nights,), 'ends')
        if not nights or starts[0] != 0 or ends[-1] != n or np.any(starts > ends) or np.any(starts[1:] != ends[:-1]):
            raise ValueError('Night slices must partition the bank')
        for lo,hi in zip(starts,ends):
            if np.any(np.diff(points[lo:hi,0]) < 0):
                raise ValueError('Points must be x-sorted within nights')
        if not np.allclose(np.linalg.norm(points,axis=1),1,atol=1e-10,rtol=0):
            raise ValueError('Points must be unit vectors')
        self.nights = nights
        stamp = time.perf_counter()
        self.arrays = [cuda.to_device(x) for x in (points,starts,ends)]
        cuda.synchronize()
        self.upload_seconds = time.perf_counter()-stamp
        self.input_bytes = points.nbytes+starts.nbytes+ends.nbytes

    def query(self, nights, predictions, radii, *, max_output_bytes=256*1024**2):
        n = len(nights)
        nights = _require(nights,np.int64,(n,),'query nights')
        predictions = _require(predictions,np.float64,(n,3),'predictions')
        radii = _require(radii,np.float64,(n,),'radii')
        if np.any(nights < 0) or np.any(nights >= self.nights) or np.any(radii < 0) or np.any(radii > 2):
            raise ValueError('Night index or chord radius out of bounds')
        if not np.allclose(np.linalg.norm(predictions,axis=1),1,atol=1e-10,rtol=0):
            raise ValueError('Predictions must be unit vectors')
        if not n:
            return (np.empty(0,np.int64), np.empty(0,np.int64), np.empty(0)), dict(queries=0,hits=0)
        stamp = time.perf_counter()
        dn, dp, dr = [cuda.to_device(x) for x in (nights,predictions,radii)]
        dc = cuda.device_array(n,np.int64)
        _count[(n+127)//128,128](*self.arrays,dn,dp,dr,dc)
        counts = dc.copy_to_host()
        counted = time.perf_counter()-stamp
        offsets = np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
        total = int(offsets[-1]); size = total*24
        if size > max_output_bytes:
            raise MemoryError(f'{total} hits require {size} output bytes; no hits returned or clipped')
        stamp = time.perf_counter()
        di, dq, dd = cuda.device_array(total,np.int64),cuda.device_array(total,np.int64),cuda.device_array(total,np.float64)
        _fill[(n+127)//128,128](*self.arrays,dn,dp,dr,cuda.to_device(offsets),di,dq,dd)
        cuda.synchronize()
        filled = time.perf_counter()-stamp
        stamp = time.perf_counter()
        result = di.copy_to_host(),dq.copy_to_host(),dd.copy_to_host()
        return result,dict(queries=n,hits=total,output_bytes=size,count_upload_download_seconds=counted,
            fill_seconds=filled,result_download_seconds=time.perf_counter()-stamp,
            simulated=bool(cuda.config.ENABLE_CUDASIM))
