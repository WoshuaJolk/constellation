"""Shared-subpath transform with bounds for a finite continuous trajectory bank.

The template is a bilinear translation of a fixed native-pixel PSF in each
exposure. FFTs supply its linear numerator and quadratic information. Bernstein
bounds screen cells; all possible detections/top-k entries are evaluated at the
original, unquantized coordinates. See docs/fold-kepler.md for the proof/scope.
"""

from time import perf_counter

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import log_ndtr, ndtri_exp

from .photometry import validate_psf_kernel


def _mix(x, y):
    return np.stack(((1-x)*(1-y), x*(1-y), (1-x)*y, x*y), axis=-1)


class FilterFrame:
    """Known background, independent known variances, fixed PSF; no fitted sky.

    Valid centers have 0 <= x < width-1 and 0 <= y < height-1. Partial PSFs
    at these edges are retained without renormalization. Other centers supply
    zero information. A nonzero mask excludes pixels, not whole centroids.
    """

    def __init__(self, image, variance, kernel, background, mask=None, scale=1.0):
        image = np.asarray(image, dtype=float)
        variance = np.broadcast_to(np.asarray(variance, dtype=float), image.shape)
        if image.ndim != 2 or min(image.shape) < 2 or image.size > 2_000_000:
            raise ValueError("Fold images must be 2D, at least 2x2, at most 2 million pixels")
        if not np.all(np.isfinite([background, scale])) or scale <= 0:
            raise ValueError("background must be finite and flux_scale positive")
        invalid = ~np.isfinite(image) | ~np.isfinite(variance) | (variance <= 0)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != image.shape:
                raise ValueError("Mask shape must match the image")
            invalid |= mask != 0
        weight = np.zeros(image.shape)
        np.divide(1., variance, out=weight, where=~invalid)
        data = np.where(invalid, 0., image - background) * weight
        kernel = validate_psf_kernel(kernel)
        self.shape, self.scale = image.shape, float(scale)
        self.numerator = fftconvolve(data, kernel[::-1, ::-1], mode="same") * scale
        # Four integer-center templates on their common support, indexed
        # (00, 10, 01, 11). Padding gives maps indexed by the lower-left center.
        kh, kw = kernel.shape
        basis = np.zeros((4, kh + 1, kw + 1))
        for k, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            basis[k, dy:dy+kh, dx:dx+kw] = kernel
        self._basis, self._weight, self._data = basis, weight, data
        # Integer support counts let truly empty footprints stay exactly empty
        # despite tiny nonlocal FFT roundoff in their information maps.
        self._support = np.rint(fftconvolve((weight > 0).astype(float),
                                          (kernel[::-1, ::-1] != 0).astype(float), mode="same")) > 0
        padded = np.pad(weight, ((kh//2, kh//2), (kw//2, kw//2)))
        self.gram = {}
        for i in range(4):
            for j in range(i, 4):
                product = basis[i] * basis[j]
                self.gram[i, j] = fftconvolve(padded, product[::-1, ::-1], mode="valid") * scale**2
        if not np.all(np.isfinite(self.numerator)) or any(not np.all(np.isfinite(g)) for g in self.gram.values()):
            raise ValueError("Non-finite filter maps: rescale image/variance/flux units")
        # Conservative numerical padding, not a formally verified FFT error
        # certificate. Global scales protect small cells against FFT leakage.
        self.n_pad = 1e-10 * max(1., float(np.max(np.abs(self.numerator))))
        self.i_pad = 1e-10 * max(1., max(float(np.max(np.abs(g))) for g in self.gram.values()))

    def _maps(self, x, y):
        n = np.stack([self.numerator[y+dy, x+dx] for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))], axis=-1)
        g = np.empty((len(x), 4, 4))
        for (i, j), plane in self.gram.items():
            g[:, i, j] = g[:, j, i] = plane[y, x]
        return n, g

    def valid(self, xy):
        return ((xy[:, 0] >= 0) & (xy[:, 0] < self.shape[1]-1)
                & (xy[:, 1] >= 0) & (xy[:, 1] < self.shape[0]-1))

    def exact(self, xy):
        """Return numerator/information at original continuous positions."""
        xy = np.asarray(xy, dtype=float)
        output = np.zeros((len(xy), 2))
        valid = self.valid(xy)
        p = xy[valid]
        base = np.floor(p).astype(np.int64)
        n, g = self._maps(base[:, 0], base[:, 1])
        a = _mix(*(p-base).T)
        output[valid, 0] = np.sum(a*n, axis=1)
        output[valid, 1] = np.maximum(0., np.einsum("bi,bij,bj->b", a, g, a))
        # Do not turn FFT leakage into tiny positive information and enormous
        # fitted fluxes in masked holes. Recompute ill-conditioned footprints
        # directly in pixel space; there is no sensitivity cutoff.
        valid_indices = np.flatnonzero(valid)
        suspect = np.flatnonzero(output[valid, 1] <= self.i_pad)
        if len(suspect):
            bx, by = base[suspect].T
            support = np.stack([self._support[by+dy, bx+dx] for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))], axis=1)
            nonempty = np.any(support & (a[suspect] > 0), axis=1)
            output[valid_indices[suspect[~nonempty]]] = 0
            height, width = self.shape
            bh, bw = self._basis.shape[1:]
            for i in suspect[nonempty]:
                template = np.einsum("i,ijk->jk", a[i], self._basis)*self.scale
                left, bottom = base[i]-np.array([(bw-1)//2, (bh-1)//2])
                x0, y0 = max(0, left), max(0, bottom)
                x1, y1 = min(width, left+bw), min(height, bottom+bh)
                stamp = template[y0-bottom:y1-bottom, x0-left:x1-left]
                output[valid_indices[i]] = [np.sum(stamp*self._data[y0:y1, x0:x1]),
                                            np.sum(stamp**2*self._weight[y0:y1, x0:x1])]
        return output

    def bounds(self, cells, subdivisions):
        """Analytic min/max numerator and biquadratic Bernstein information."""
        result = np.zeros((len(cells), 4))
        valid = cells[:, 0] >= 0  # (-1,-1) is the off-detector sentinel.
        p = cells[valid]
        base = p // subdivisions
        fraction = (p % subdivisions) / subdivisions
        n, g = self._maps(base[:, 0], base[:, 1])
        if subdivisions == 1:
            vertices, vg = n, g
        else:
            delta = 1. / subdivisions
            a = np.stack([_mix(fraction[:, 0]+dx*delta, fraction[:, 1]+dy*delta)
                          for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))], axis=1)
            vertices = np.einsum("bvi,bi->bv", a, n)
            vg = a @ g @ np.swapaxes(a, 1, 2)
        # The nine degree-(2,2) Bernstein coefficients of squared norm.
        coeff = np.stack([vg[:, 0, 0], vg[:, 1, 1], vg[:, 2, 2], vg[:, 3, 3],
                          vg[:, 0, 1], vg[:, 0, 2], vg[:, 1, 3], vg[:, 2, 3],
                          (vg[:, 0, 3] + vg[:, 1, 2])/2], axis=1)
        result[valid] = np.stack([vertices.min(axis=1)-self.n_pad,
                                  vertices.max(axis=1)+self.n_pad,
                                  np.maximum(0., coeff.min(axis=1)-self.i_pad),
                                  np.maximum(0., coeff.max(axis=1)+self.i_pad)], axis=1)
        bx, by = base.T
        supported = np.any(np.stack([self._support[by+dy, bx+dx]
                                    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))], axis=1), axis=1)
        result[np.flatnonzero(valid)[~supported]] = 0
        return result


class FoldPlan:
    """Compile recurring time-local cell subpaths into a balanced DAG.

    Compilation examines every supplied position. The plan can be reused for
    new images with the same shapes and trajectory coordinates (e.g. controls).
    Evaluation cost is proportional to distinct leaves/subpaths plus refinements;
    no worst-case sublinear-in-bank guarantee is claimed.
    """

    def __init__(self, positions, shapes, subdivisions=1):
        positions = np.asarray(positions, dtype=float)
        if (positions.ndim != 3 or positions.shape[2] != 2 or min(positions.shape[:2]) < 1
                or positions.shape[0] > 100_000 or positions.shape[1] > 256
                or np.prod(positions.shape[:2]) > 4_000_000
                or not np.all(np.isfinite(positions)) or np.max(np.abs(positions)) > 1e8):
            raise ValueError("Finite positions must be (1..100000 tracks,1..256 frames,2), <=4 million pairs, |xy|<=1e8")
        if isinstance(subdivisions, bool) or subdivisions not in (1, 2, 4, 8, 16):
            raise ValueError("cell_subdivisions must be one of 1,2,4,8,16")
        if len(shapes) != positions.shape[1] or any(len(s) != 2 or min(s) < 2 for s in shapes):
            raise ValueError("One valid image shape is required per frame")
        self.positions = positions.copy()
        self.positions.setflags(write=False)
        self.shapes, self.subdivisions = [tuple(s) for s in shapes], int(subdivisions)
        self.nodes = []
        self.leaf_count, self.add_count = 0, 0

        def compile_node(start, stop):
            if stop-start == 1:
                xy, shape = positions[:, start], self.shapes[start]
                cells = np.floor(xy*subdivisions).astype(np.int64)
                valid = ((xy[:, 0] >= 0) & (xy[:, 0] < shape[1]-1)
                         & (xy[:, 1] >= 0) & (xy[:, 1] < shape[0]-1))
                cells[~valid] = -1
                unique, inverse = np.unique(cells, axis=0, return_inverse=True)
                node = {"frame": start, "cells": unique}
                self.leaf_count += len(unique)
            else:
                middle = (start+stop)//2
                left, li = compile_node(start, middle)
                right, ri = compile_node(middle, stop)
                pairs, inverse = np.unique(np.column_stack((li, ri)), axis=0, return_inverse=True)
                node = {"left": left, "right": right, "pairs": pairs}
                self.add_count += len(pairs)
            index = len(self.nodes)
            self.nodes.append(node)
            return index, inverse

        self.root, self.inverse = compile_node(0, positions.shape[1])

    def bounds(self, frames):
        if [f.shape for f in frames] != self.shapes:
            raise ValueError("Filter frames must match the compiled image shapes")
        values = []
        for node in self.nodes:
            if "frame" in node:
                value = frames[node["frame"]].bounds(node["cells"], self.subdivisions)
            else:
                p = node["pairs"]
                value = values[node["left"]][p[:, 0]] + values[node["right"]][p[:, 1]]
                # Each child has one parent in the time tree; release its maps.
                values[node["left"]] = values[node["right"]] = None
            values.append(value)
        return values[self.root][self.inverse]


def direct_statistics(frames, positions):
    """Vectorized direct baseline, using the same FFT maps and exact PSF model."""
    result = np.zeros((len(positions), 2))
    for start in range(0, len(positions), 4096):
        for j, frame in enumerate(frames):
            result[start:start+4096] += frame.exact(positions[start:start+4096, j])
    return result


def _snr_limits(bounds):
    nl, nu, il, iu = bounds.T
    lower, upper = np.full(len(bounds), -np.inf), np.full(len(bounds), np.inf)
    # A negative numerator reverses which denominator gives the extreme.
    for output, numerator, denominator in ((lower, nl, np.where(nl >= 0, iu, il)),
                                            (upper, nu, np.where(nu >= 0, il, iu))):
        np.divide(numerator, np.sqrt(denominator), out=output, where=denominator > 0)
    upper[iu <= 0] = -np.inf
    return lower, upper


def search_fold(frames, plan, ids=None, alpha=0.01, top_k=10, validate_direct=False):
    """Return exact top-k/detections, refining all potentially qualifying cells."""
    n = len(plan.positions)
    ids = [str(i) for i in range(n)] if ids is None else list(ids)
    if len(ids) != n or len(set(ids)) != n or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("One unique nonempty string ID is required per trajectory")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= min(n, 1000):
        raise ValueError("top_k must be in 1..min(bank size,1000)")
    threshold = float(-ndtri_exp(np.log(alpha)-np.log(n)))
    start = perf_counter()
    bounds = plan.bounds(frames)
    lower, upper = _snr_limits(bounds)
    cutoff = min(threshold, np.partition(lower, n-top_k)[n-top_k])
    # Equality is retained, so ties cannot disappear at the screening step.
    selected = np.flatnonzero(upper >= cutoff)
    stats = direct_statistics(frames, plan.positions[selected])
    eligible = stats[:, 1] > 0
    snr = np.full(len(selected), -np.inf)
    snr[eligible] = stats[eligible, 0] / np.sqrt(stats[eligible, 1])
    order = np.lexsort((selected, -snr))
    top = [i for i in order if eligible[i]][:top_k]
    detected = [i for i in order if eligible[i] and snr[i] >= threshold]
    elapsed = perf_counter()-start

    def report(i):
        numerator, information = stats[i]
        log_p = min(0., float(log_ndtr(-snr[i]))+np.log(n))
        return {"id": ids[selected[i]], "index": int(selected[i]), "snr": float(snr[i]),
                "flux": float(numerator/information), "flux_error": float(1/np.sqrt(information)),
                "log_p_family_upper_bound": log_p, "p_family_upper_bound": float(np.exp(log_p)),
                "passes_gaussian_threshold": bool(snr[i] >= threshold)}

    result = {"trials": n, "alpha": alpha, "threshold_snr": threshold,
              "top": [report(i) for i in top], "detections": [report(i) for i in detected],
              "work": {"direct_track_frame_pairs": int(np.prod(plan.positions.shape[:2])),
                       "distinct_cell_leaves": plan.leaf_count, "shared_subpath_additions": plan.add_count,
                       "refined_tracks": len(selected), "screened_tracks": n-len(selected),
                       "fold_evaluation_seconds": elapsed},
              "screening": {"model": "exact bilinear translation of fixed native-pixel PSF; known background",
                            "method": "Bernstein cell bounds; exact refinement at original coordinates",
                            "arithmetic": "float64 with 1e-10 map-scale padding; not an interval-arithmetic certificate",
                            "scope": "finite supplied bank only; no orbit discretization or physical PSF accuracy guarantee"}}
    if validate_direct:
        start = perf_counter()
        direct = direct_statistics(frames, plan.positions)
        seconds = perf_counter()-start
        good = direct[:, 1] > 0
        z = np.full(n, -np.inf)
        z[good] = direct[good, 0] / np.sqrt(direct[good, 1])
        ranked = np.lexsort((np.arange(n), -z))
        expected_top = [int(i) for i in ranked if good[i]][:top_k]
        expected_detections = [int(i) for i in ranked if good[i] and z[i] >= threshold]
        violations = np.flatnonzero(good & ((z < lower-1e-9) | (z > upper+1e-9)))
        if (len(violations) or expected_top != [int(selected[i]) for i in top]
                or expected_detections != [int(selected[i]) for i in detected]):
            raise ValueError("Fold validation failed: bounds, detections or top-k differ from direct evaluation")
        error = float(np.max(np.abs(snr[eligible]-z[selected[eligible]]))) if np.any(eligible) else 0.
        result["validation"] = {"all_bounds_contain_direct_snr": True, "identical_top_and_detections": True,
                                "max_refined_snr_difference": error, "direct_evaluation_seconds": seconds,
                                "evaluation_speedup": seconds/elapsed}
    return result
