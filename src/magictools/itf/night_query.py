"""Exact batched spherical-cone queries over per-night sorted unit vectors."""
import numpy as np
from numba import njit


@njit(cache=True)
def query_nights(units, starts, ends, nights, predicted, radii):
    points, offsets, distances = [], [], []
    for offset in range(len(nights)):
        night = nights[offset]
        start, end = starts[night], ends[night]
        radius = radii[offset]
        x, y, z = predicted[offset]
        lo, hi = start, end
        while lo < hi:
            mid = (lo+hi)//2
            if units[mid, 0] < x-radius:
                lo = mid+1
            else:
                hi = mid
        first = lo
        lo, hi = first, end
        while lo < hi:
            mid = (lo+hi)//2
            if units[mid, 0] <= x+radius:
                lo = mid+1
            else:
                hi = mid
        squared = radius*radius
        for point in range(first, lo):
            dx, dy, dz = units[point, 0]-x, units[point, 1]-y, units[point, 2]-z
            distance = dx*dx+dy*dy+dz*dz
            if distance <= squared:
                points.append(point)
                offsets.append(offset)
                distances.append(np.sqrt(distance))
    return np.asarray(points, dtype=np.int64), np.asarray(offsets, dtype=np.int64), np.asarray(distances)
