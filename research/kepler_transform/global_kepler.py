"""Continuous-time Kepler phase oracle for the global mathematical attempt."""

from types import SimpleNamespace

import numpy as np

from .compiled_reference import projected_chart
from .factorization import observer_path
from .universal import apparent


class FrequencyGate:
    """Exact separation of exp(i*(kx*X(t)+ky*Y(t))) into two factors."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def apply(self, values, *, adjoint=False):
        x, y = (self.x.conj(), self.y.conj()) if adjoint else (self.x, self.y)
        return values*x[:, :, None]*y[:, None, :]

    def dense(self):
        return (self.x[:, :, None]*self.y[:, None, :]).ravel()


class ContinuousKeplerPhase:
    def __init__(self, chart, center, widths, *, time_half_span, frequency_scale):
        self.chart = chart
        self.center, self.widths = np.asarray(center, float), np.asarray(widths, float)
        if (self.center.shape != (6,) or self.widths.shape != (6,)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.widths))
                or np.any(self.widths <= 0) or self.center[0] <= self.widths[0]
                or not np.isfinite(time_half_span) or time_half_span <= 0
                or not np.isfinite(frequency_scale) or frequency_scale <= 0):
            raise ValueError("Positive finite chart widths, span and bandwidth required")
        self.span, self.frequency_scale = time_half_span, frequency_scale
        self.geometry_pairs = 0

    def geometry(self, times):
        count = len(times)
        return SimpleNamespace(times=times, observers=observer_path(times),
            normal=np.tile(self.chart.axes[0], (count, 1)),
            east=np.tile(self.chart.axes[1], (count, 1)),
            north=np.tile(self.chart.axes[2], (count, 1)))

    def __call__(self, targets, sources):
        targets, sources = np.asarray(targets, float), np.asarray(sources, float)
        if (targets.ndim != 2 or targets.shape[1] != 6 or sources.ndim != 2
                or sources.shape[1] != 3 or not np.all(np.isfinite(targets))
                or not np.all(np.isfinite(sources)) or np.any(abs(targets) > 1)
                or np.any(abs(sources) > 1)):
            raise ValueError("Normalized six-dimensional targets and three-dimensional sources required")
        times, inverse = np.unique(sources[:, 0]*self.span, return_inverse=True)
        parameters = self.center + targets*self.widths
        positions = projected_chart(self.chart, parameters, self.geometry(times))
        self.geometry_pairs += len(targets)*len(times)
        return np.einsum("nmi,mi->nm", positions[:, inverse], sources[:, 1:]*self.frequency_scale)

    def independent(self, targets, sources):
        """Independent Python universal solver with epoch conversion."""
        times, inverse = np.unique(sources[:, 0]*self.span, return_inverse=True)
        g = self.geometry(times)
        states = self.chart.states(self.center + np.asarray(targets)*self.widths)
        emitted, _ = apparent(states, times, g.observers)
        delta = emitted[..., :3]-g.observers
        denominator = np.einsum("nti,ti->nt", delta, g.normal)
        if np.any(denominator <= 0):
            raise ValueError("Independent check crosses tangent horizon")
        positions = np.stack([np.einsum("nti,ti->nt", delta, axis)/denominator
                              for axis in [g.east, g.north]], axis=-1)
        return np.einsum("nmi,mi->nm", positions[:, inverse], sources[:, 1:]*self.frequency_scale)

    def factorized_gate(self, targets, nodes):
        """Kernel at one center, or its ratio at two centers, on a tensor grid."""
        geometry = self.geometry(np.asarray(nodes[0])*self.span)
        positions = projected_chart(self.chart, self.center+np.asarray(targets)*self.widths, geometry)
        self.geometry_pairs += len(targets)*len(nodes[0])
        delta = positions[0] if len(targets) == 1 else positions[0]-positions[1]
        return FrequencyGate(
            np.exp(1j*delta[:, 0, None]*(np.asarray(nodes[1])*self.frequency_scale)),
            np.exp(1j*delta[:, 1, None]*(np.asarray(nodes[2])*self.frequency_scale)))

    def evaluate_expansion(self, targets, nodes, coefficients, center, *, chunk=8192, eps=1e-12):
        """Exploit exact linearity in spatial frequency at each source-time node.

        This evaluates the same final field with 2D NUFFTs. No additional
        orbit interpolation or fitted physical model is introduced.
        """
        import finufft
        from .factorization import ARCSEC
        targets = np.asarray(targets, float)
        geometry = self.geometry(np.asarray(nodes[0])*self.span)
        center_positions = projected_chart(self.chart, (self.center+center*self.widths)[None], geometry)[0]
        self.geometry_pairs += len(nodes[0])
        xx, yy = np.meshgrid(nodes[1], nodes[2], indexing="ij")
        sx = np.ascontiguousarray(xx.ravel()*self.frequency_scale*ARCSEC)
        sy = np.ascontiguousarray(yy.ravel()*self.frequency_scale*ARCSEC)
        result = np.zeros(len(targets), complex)
        for start in range(0, len(targets), chunk):
            selected = targets[start:start+chunk]
            positions = projected_chart(self.chart, self.center+selected*self.widths, geometry)
            self.geometry_pairs += len(selected)*len(nodes[0])
            positions = (positions-center_positions)/ARCSEC
            if len(selected) <= 16:
                x = np.exp(1j*positions[:, :, 0, None]*np.asarray(nodes[1])*self.frequency_scale*ARCSEC)
                y = np.exp(1j*positions[:, :, 1, None]*np.asarray(nodes[2])*self.frequency_scale*ARCSEC)
                result[start:start+len(selected)] = np.einsum("ntx,txy,nty->n", x, coefficients, y,
                                                             optimize=True)
                continue
            for visit in range(len(nodes[0])):
                result[start:start+len(selected)] += finufft.nufft2d3(
                    sx, sy, np.ascontiguousarray(coefficients[visit].ravel()),
                    np.ascontiguousarray(positions[:, visit, 0]),
                    np.ascontiguousarray(positions[:, visit, 1]),
                    eps=eps, isign=1, nthreads=1)
        return result
