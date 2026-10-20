"""Continuous-time Kepler phase oracle for the global mathematical attempt."""

from types import SimpleNamespace

import numpy as np

from .compiled_reference import projected_chart
from .factorization import observer_path
from .universal import apparent


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
