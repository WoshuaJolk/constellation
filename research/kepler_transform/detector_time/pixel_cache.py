"""Strong direct baselines: cache separable PSFs or the full pixel response.

Information is independent of image values when masks, variance, PSF and
astrometry stay fixed. Both baselines cache it explicitly.
"""

import numpy as np
from scipy.special import ndtr


class PixelCache:
    def __init__(self, detector, positions, *, backend="factor", max_bytes=1_000_000_000):
        p = detector.pixel_positions(positions)
        self.detector, self.queries = detector, len(p)
        if backend not in ("factor", "dense", "stream"):
            raise ValueError("Unknown pixel cache backend")
        self.backend = backend
        if backend == "stream":
            if p.nbytes > max_bytes:
                raise ValueError("Pixel position cache exceeds budget")
            self.positions = p
            self.persistent_bytes = p.nbytes
            return
        count = (self.queries * detector.frames * detector.n**2 if backend == "dense"
                 else 2 * self.queries * detector.frames * detector.n)
        if count * 8 > max_bytes:
            raise ValueError(f"Pixel response cache exceeds budget: {count * 8}")
        axis = np.arange(detector.n) - (detector.n - 1) / 2
        self.information = np.zeros(self.queries)
        if backend == "dense":
            self.response = np.empty((self.queries, np.prod(detector.shape_pixels)), float)
        else:
            self.px = np.empty((detector.frames, self.queries, detector.n))
            self.py = np.empty_like(self.px)
        for j, (sx, sy) in enumerate(detector.sigma):
            x, y = p[:, j, 0, None], p[:, j, 1, None]
            px = ndtr((axis + .5 - x) / sx) - ndtr((axis - .5 - x) / sx)
            py = ndtr((axis + .5 - y) / sy) - ndtr((axis - .5 - y) / sy)
            w, scale = detector.weight[j], detector.scales[j]
            overlap = np.sum((py @ w) * px, axis=1)
            info = np.sum((py**2 @ w) * px**2, axis=1)
            total = np.sum(w)
            if detector.profile and total > 0:
                info -= overlap**2 / total
            self.information += scale**2 * info
            if backend == "factor":
                self.px[j], self.py[j] = px, py
            else:
                response = py[:, :, None] * px[:, None, :]
                if detector.profile and total > 0:
                    response -= overlap[:, None, None] / total
                response *= w * scale
                start = j * detector.n**2
                self.response[:, start:start + detector.n**2] = response.reshape(self.queries, -1)
        self.persistent_bytes = count * 8 + self.information.nbytes

    def apply(self, images):
        images = np.asarray(images, float)
        d = self.detector
        if images.shape != d.shape_pixels or not np.all(np.isfinite(images)):
            raise ValueError("Finite images matching fixed detector geometry required")
        if self.backend == "dense":
            return self.response @ images.ravel(), self.information.copy()
        n = np.zeros(self.queries)
        information = np.zeros(self.queries) if self.backend == "stream" else self.information.copy()
        axis = np.arange(d.n) - (d.n - 1) / 2
        for j in range(d.frames):
            w, values = d.weight[j], images[j].copy()
            if d.profile and np.sum(w) > 0:
                values -= np.sum(values * w) / np.sum(w)
            if self.backend == "stream":
                x, y = self.positions[:, j, 0, None], self.positions[:, j, 1, None]
                sx, sy = d.sigma[j]
                px = ndtr((axis + .5 - x) / sx) - ndtr((axis - .5 - x) / sx)
                py = ndtr((axis + .5 - y) / sy) - ndtr((axis - .5 - y) / sy)
                info = np.sum((py**2 @ w) * px**2, axis=1)
                if d.profile and np.sum(w) > 0:
                    info -= np.sum((py @ w) * px, axis=1)**2 / np.sum(w)
                information += d.scales[j]**2 * info
            else:
                px, py = self.px[j], self.py[j]
            n += d.scales[j] * np.sum((py @ (w * values)) * px, axis=1)
        return n, information
