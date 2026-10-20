"""Counted, lazy Kepler/Fourier kernel and independent fast-sum baseline.

Positions use all-conic universal variables, a moving reception observer and
orientation-dependent light time. Input coefficients describe finite Fourier
response fields; they are NOT automatically a calibrated detector likelihood.
"""

from time import perf_counter

import numpy as np

from magictools.orbits import GM
from .factorization import ARCSEC, observer_path
from .universal import apparent


class KeplerKernel:
    def __init__(self, states, times, observers, visits, frequencies, reference):
        self.states = np.asarray(states, float)
        self.times = np.asarray(times, float)
        self.observers = np.asarray(observers, float)
        self.visits = np.asarray(visits, int)
        self.frequencies = np.asarray(frequencies, float)
        if (self.states.ndim != 2 or self.states.shape[1] != 6 or not len(self.states)
                or self.times.ndim != 1 or not len(self.times)
                or self.observers.shape != (len(self.times), 3)
                or self.visits.ndim != 1 or not len(self.visits)
                or self.frequencies.shape != (len(self.visits), 2)
                or np.any(self.visits < 0) or np.any(self.visits >= len(self.times))
                or any(not np.all(np.isfinite(a)) for a in
                       [self.states, self.times, self.observers, self.frequencies])):
            raise ValueError("Invalid finite Kepler/Fourier inputs")
        reference = np.asarray(reference, float)
        if reference.shape != (6,) or not np.all(np.isfinite(reference)):
            raise ValueError("Finite six-dimensional reference state required")
        origin, _ = apparent(reference[None], self.times, self.observers)
        self.normal = origin[0, :, :3]-self.observers
        self.normal /= np.linalg.norm(self.normal, axis=1)[:, None]
        axes = np.eye(3)[np.argmin(abs(self.normal), axis=1)]
        self.east = np.cross(axes, self.normal)
        self.east /= np.linalg.norm(self.east, axis=1)[:, None]
        self.north = np.cross(self.normal, self.east)
        self.positions = np.full((len(states), len(times), 2), np.nan)
        self.geometry_pairs, self.entries, self.geometry_seconds = 0, 0, 0.

    def projected(self, rows, visits):
        for visit in np.unique(visits):
            missing = np.unique(rows[np.isnan(self.positions[rows, visit, 0])])
            if not len(missing):
                continue
            start = perf_counter()
            state, _ = apparent(self.states[missing], self.times[visit:visit+1],
                                self.observers[visit:visit+1])
            delta = state[:, 0, :3]-self.observers[visit]
            denominator = delta @ self.normal[visit]
            if np.any(denominator <= 0):
                raise ValueError("Requested orbit crosses the gnomonic horizon")
            self.positions[missing, visit] = np.column_stack((
                delta @ self.east[visit]/denominator, delta @ self.north[visit]/denominator))
            self.geometry_pairs += len(missing)
            self.geometry_seconds += perf_counter()-start
        return self.positions[np.asarray(rows)[:, None], np.asarray(visits)[None]]

    def __call__(self, rows, columns):
        rows, columns = np.asarray(rows, int), np.asarray(columns, int)
        self.entries += len(rows)*len(columns)
        positions = self.projected(rows, self.visits[columns])
        return np.exp(1j*np.einsum("nmi,mi->nm", positions, self.frequencies[columns]))

    def direct(self, data, chunk=256):
        output = np.empty(len(self.states), complex)
        for start in range(0, len(self.states), chunk):
            rows = np.arange(start, min(start+chunk, len(self.states)))
            output[rows] = self(rows, np.arange(len(self.visits))) @ data
        return output

    def nufft(self, data, eps=1e-12):
        """Type-3 NUFFT per exposure; no dense phase matrix. Includes all modes."""
        import finufft
        rows = np.arange(len(self.states))
        output = np.zeros(len(rows), complex)
        for visit in np.unique(self.visits):
            selected = self.visits == visit
            positions = self.projected(rows, np.array([visit]))[:, 0]
            # Rescale reciprocal coordinates to avoid extreme radian magnitudes.
            frequency = np.ascontiguousarray(self.frequencies[selected]*ARCSEC)
            target = np.ascontiguousarray(positions/ARCSEC)
            output += finufft.nufft2d3(np.ascontiguousarray(frequency[:, 0]),
                                      np.ascontiguousarray(frequency[:, 1]),
                                      np.ascontiguousarray(data[selected], dtype=complex),
                                      np.ascontiguousarray(target[:, 0]),
                                      np.ascontiguousarray(target[:, 1]), eps=eps,
                                      isign=1, nthreads=1)
        return output


def physical_problem(n=512, modes=1024, *, case="bound", width=1., seed=20261020):
    """Six-dimensional volumes, not only rotations of a single radial shape."""
    rng = np.random.default_rng(seed)
    parameters = rng.uniform(-1, 1, (n, 6))
    if case == "bound":
        radius, speed, span = 3., 1., 8.
    elif case == "mixed_energy":
        radius, speed, span = 3., np.sqrt(2), 8.
    elif case == "close_observer":
        radius, speed, span = 1.02, 1.8, .5
    else:
        raise ValueError("Unknown physical case")
    center = np.array([radius, 0., 0., 0., speed*np.sqrt(GM["sun"]/radius), 0.])
    distance = radius-1
    widths = np.array([distance*.025, distance*10*ARCSEC, distance*10*ARCSEC,
                       .0002, distance*2*ARCSEC, distance*2*ARCSEC])*width
    states = center+parameters*widths
    times = np.linspace(-span/2, span/2, 16)
    observers = observer_path(times)
    visits = rng.integers(len(times), size=modes)
    frequencies = rng.uniform(-1, 1, (modes, 2))/ARCSEC
    source = np.column_stack((times[visits]/span, frequencies*ARCSEC))
    # These are data-independent test coefficients, not observations of stars.
    data = rng.normal(size=modes)+1j*rng.normal(size=modes)
    return KeplerKernel(states, times, observers, visits, frequencies, center), parameters, source, data


def phase_coordinates(kernel, parameters):
    """Local phase-gradient coordinates for an alternative partition.

    Finite differences guide partitioning only; they never replace exact
    geometry in the kernel or supply an error certificate.
    """
    started = perf_counter()
    # Recover the affine physical parameter map, without assuming component
    # widths from a particular synthetic experiment.
    affine = np.linalg.lstsq(np.column_stack((np.ones(len(parameters)), parameters)),
                            kernel.states, rcond=None)[0]
    center, basis = affine[0], affine[1:]
    perturbation = 1e-3
    states = np.concatenate((center+perturbation*basis, center-perturbation*basis))
    emitted, _ = apparent(states, kernel.times, kernel.observers)
    delta = emitted[..., :3]-kernel.observers
    denominator = np.einsum("nti,ti->nt", delta, kernel.normal)
    if np.any(denominator <= 0):
        raise ValueError("Phase-gradient stencil crosses gnomonic horizon")
    projected = np.stack([np.einsum("nti,ti->nt", delta, axis)/denominator
                          for axis in (kernel.east, kernel.north)], axis=-1)
    gradient = (projected[:6]-projected[6:])/(2*perturbation)
    phase = np.einsum("dmi,mi->md", gradient[:, kernel.visits], kernel.frequencies)
    _, singular, rotation = np.linalg.svd(phase-phase.mean(axis=0), full_matrices=False)
    scale = np.maximum(np.sqrt(singular/max(1., np.sqrt(len(phase)))), 1e-12)
    row_coordinates = (parameters @ rotation.T)*scale
    column_coordinates = (phase @ rotation.T)/scale
    return row_coordinates, column_coordinates, {
        "seconds": perf_counter()-started, "additional_geometry_pairs": 12*len(kernel.times),
        "phase_gradient_singular_values": singular.tolist(),
        "role": "partition heuristic only; kernel retains exact retarded geometry"}
