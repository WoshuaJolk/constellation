"""Optional exact three-axis index for the existing spherical night queries.

Select the narrowest coordinate slab, then use the original 3D chord-distance
predicate. No cone, uncertainty, physics or candidate limit changes. Results are
returned in the same per-query original-point order as query_nights. Index cost
and memory must be counted separately; sparse workloads may be slower.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _bound(units, order, start, end, axis, value, upper):
    while start < end:
        mid = (start+end)//2
        coord = units[order[mid], axis]
        if coord < value or (upper and coord == value):
            start = mid+1
        else:
            end = mid
    return start


@njit(cache=True, nogil=True)
def query_indexed_nights(units, orders, starts, ends, nights, predicted, radii):
    points, offsets, distances = [], [], []
    for offset in range(len(nights)):
        night = nights[offset]
        start, end = starts[night], ends[night]
        radius = radii[offset]
        axis, first, last = 0, start, end
        for k in range(3):
            lo = _bound(units, orders[k], start, end, k, predicted[offset,k]-radius, False)
            hi = _bound(units, orders[k], lo, end, k, predicted[offset,k]+radius, True)
            if hi-lo < last-first:
                axis, first, last = k, lo, hi
        selected = []
        x, y, z = predicted[offset]
        squared = radius*radius
        for j in range(first, last):
            point = orders[axis,j]
            dx,dy,dz = units[point,0]-x,units[point,1]-y,units[point,2]-z
            distance = dx*dx+dy*dy+dz*dz
            if distance <= squared:
                selected.append(point)
        # Original query scans x-sorted points, not selected-axis order.
        for point in np.sort(np.asarray(selected, dtype=np.int64)):
            dx,dy,dz = units[point,0]-x,units[point,1]-y,units[point,2]-z
            points.append(point)
            offsets.append(offset)
            distances.append(np.sqrt(dx*dx+dy*dy+dz*dz))
    return np.asarray(points,dtype=np.int64),np.asarray(offsets,dtype=np.int64),np.asarray(distances)


def build_axis_orders(units, starts, ends):
    """Units must already be finite and x-sorted inside each night (as before)."""
    units = np.asarray(units)
    if units.ndim != 2 or units.shape[1] != 3 or not np.isfinite(units).all():
        raise ValueError('Finite (N,3) unit vectors required')
    if (len(starts) != len(ends) or (len(starts) and
            (starts[0] != 0 or ends[-1] != len(units) or np.any(starts > ends)
             or np.any(starts[1:] != ends[:-1])))):
        raise ValueError('Night slices must partition the point bank')
    orders = np.tile(np.arange(len(units),dtype=np.int64),(3,1))
    for lo,hi in zip(starts,ends):
        if np.any(np.diff(units[lo:hi,0]) < 0):
            raise ValueError('Night points must be x-sorted')
        for k in (1,2):
            orders[k,lo:hi] = lo+np.argsort(units[lo:hi,k],kind='stable')
    return orders
