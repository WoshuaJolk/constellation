"""Attempts A/B: physical-template rank and complementary-block diagnostics.

The block prototype deliberately counts dense construction and exact SVDs.
It is a falsifiable test of a proposed representation, not a fast construction
algorithm or an implementation of the cited butterfly papers.
"""

from time import perf_counter

import numpy as np
from scipy.linalg import svd

from magictools.orbits import GM
from .universal import apparent

ARCSEC = np.pi/(180*3600)


def observer_path(times):
    phase = np.sqrt(GM["sun"])*np.asarray(times)
    return np.column_stack((np.cos(phase), np.sin(phase), np.zeros(len(phase))))


def tangent_positions(states, times, observers, reference=None):
    if reference is None:
        reference = np.mean(states, axis=0)
    reference_state, _ = apparent(np.asarray(reference)[None], times, observers)
    normal = reference_state[0, :, :3]-observers
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    east = np.cross(np.array([0., 0., 1.]), normal)
    east /= np.linalg.norm(east, axis=1)[:, None]
    north = np.cross(normal, east)
    state, _ = apparent(states, times, observers)
    delta = state[..., :3]-observers
    denominator = np.einsum("nti,ti->nt", delta, normal)
    if np.any(denominator <= 0):
        raise ValueError("Research field crosses its tangent horizon")
    return np.stack([np.einsum("nti,ti->nt", delta, direction)/denominator
                     for direction in [east, north]], axis=-1)


def global_rank(side=16):
    times = np.linspace(-2, 2, 16)
    yy, xx = np.meshgrid(np.arange(side)-(side-1)/2, np.arange(side)-(side-1)/2)
    rows = []
    for label, spacing_arcsec in [("tight_neighbors", .008), ("resolved_orbits", 8.)]:
        states = np.tile([3., 0., 0., 0., np.sqrt(GM["sun"]/3), 0.], (side**2, 1))
        states[:, 1] = xx.ravel()*spacing_arcsec*ARCSEC*2
        states[:, 2] = yy.ravel()*spacing_arcsec*ARCSEC*2
        positions = tangent_positions(states, times, observer_path(times))
        gram = np.zeros((len(states), len(states)))
        for j in range(len(times)):
            distance2 = np.sum((positions[:, None, j]-positions[None, :, j])**2, axis=2)
            gram += np.exp(-distance2/(4*ARCSEC**2))/len(times)
        eigenvalues = np.maximum(0., np.linalg.eigvalsh(gram)[::-1])
        cumulative = np.cumsum(eigenvalues)/eigenvalues.sum()
        rank99 = int(np.searchsorted(cumulative, .99))+1
        off_diagonal = gram-np.diag(np.diag(gram))
        rows.append({"case": label, "orbits": len(states), "frames": len(times),
                     "rank_for_99_percent_template_energy": rank99,
                     "minimum_gram_eigenvalue": float(eigenvalues[-1]),
                     "gershgorin_eigenvalue_lower_bound": float(np.min(np.diag(gram)-np.sum(np.abs(off_diagonal), axis=1))),
                     "rank_16_best_relative_frobenius_error": float(np.sqrt(eigenvalues[16:].sum()/eigenvalues.sum()))})
    return rows


def kd_groups(points, count):
    """Balanced median tree on normalized coordinates; count is a power of 2."""
    groups = [np.arange(len(points))]
    while len(groups) < count:
        divided = []
        for indices in groups:
            axis = np.argmax(np.ptp(points[indices], axis=0))
            order = indices[np.argsort(points[indices, axis], kind="stable")]
            middle = len(order)//2
            divided.extend((order[:middle], order[middle:]))
        groups = divided
    return groups


class BlockFactorization:
    def __init__(self, matrix, row_points, column_points, relative_tolerance=1e-8):
        start = perf_counter()
        nr, nc = matrix.shape
        level = int(np.log2(min(nr, nc)))//2
        rows = kd_groups(row_points, 2**level)
        columns = kd_groups(column_points, 2**(int(np.log2(min(nr, nc)))-level))
        self.shape, self.blocks = matrix.shape, []
        ranks, errors, stored, compressed = [], [], 0, 0
        for r in rows:
            for c in columns:
                block = matrix[np.ix_(r, c)]
                u, s, vh = svd(block, full_matrices=False, check_finite=False)
                tail = np.r_[np.cumsum(s[::-1]**2)[::-1], 0.]
                rank = int(np.flatnonzero(tail <= relative_tolerance**2*tail[0])[0])
                ranks.append(rank)
                if rank*(len(r)+len(c)) < block.size:
                    self.blocks.append((r, c, u[:, :rank]*s[:rank], vh[:rank]))
                    stored += rank*(len(r)+len(c))
                    errors.append(tail[rank])
                    compressed += 1
                else:
                    self.blocks.append((r, c, block.copy(), None))
                    stored += block.size
        self.report = {"construction_seconds": perf_counter()-start, "blocks": len(self.blocks),
                       "compressed_blocks": compressed, "median_block_rank": float(np.median(ranks)),
                       "maximum_block_rank": int(max(ranks)), "stored_complex_numbers": stored,
                       "dense_complex_numbers": matrix.size, "storage_ratio": stored/matrix.size,
                       "relative_frobenius_error_bound": float(np.sqrt(sum(errors))/np.linalg.norm(matrix)),
                       "construction_kernel_entries": matrix.size,
                       "construction_method": "full dense phase kernel followed by block SVD; not fast construction"}

    def apply(self, data, adjoint=False):
        result = np.zeros(self.shape[1 if adjoint else 0], dtype=complex)
        for rows, columns, left, right in self.blocks:
            if adjoint:
                value = left.conj().T @ data[rows]
                result[columns] += value if right is None else right.conj().T @ value
            else:
                result[rows] += left @ (data[columns] if right is None else right @ data[columns])
        return result


def hierarchical_probe(n=256, case="bound", seed=2026, frequency_fraction=1., relative_tolerance=1e-8):
    rng = np.random.default_rng(seed)
    parameters = rng.uniform(-1, 1, (n, 6))
    if case == "bound":
        center = np.array([3., 0., 0., 0., np.sqrt(GM["sun"]/3), 0.])
        widths, baseline = [.1, .001, .001, .001, .001, .001], 8.
    elif case == "mixed_energy":
        center = np.array([3., 0., 0., 0., np.sqrt(2*GM["sun"]/3), 0.])
        widths, baseline = [.1, .001, .001, .001, .003, .001], 8.
    elif case == "close_observer":
        center = np.array([1.01, 0., 0., 0., .025, 0.])
        widths, baseline = [.002, .0002, .0002, .001, .001, .001], .5
    else:
        raise ValueError("Unknown phase-kernel probe")
    times = np.linspace(-baseline/2, baseline/2, 32)
    states = center+parameters*np.asarray(widths)
    start = perf_counter()
    positions = tangent_positions(states, times, observer_path(times), center)
    projection_seconds = perf_counter()-start
    source = rng.uniform(-1, 1, (n, 3))
    visits = np.minimum(31, ((source[:, 0]+1)*16).astype(int))
    # Physical Fourier modes at up to one inverse PSF sigma: these modes have
    # not been exponentially erased by a one-arcsecond Gaussian PSF.
    frequencies = source[:, 1:]/ARCSEC*frequency_fraction
    start = perf_counter()
    kernel = np.exp(1j*np.einsum("nsi,si->ns", positions[:, visits], frequencies))
    kernel_seconds = perf_counter()-start
    factor = BlockFactorization(kernel, parameters, source, relative_tolerance)
    data = rng.normal(size=n)+1j*rng.normal(size=n)
    target = rng.normal(size=n)+1j*rng.normal(size=n)
    start = perf_counter()
    result = factor.apply(data)
    apply_seconds = perf_counter()-start
    start = perf_counter()
    direct = kernel @ data
    direct_seconds = perf_counter()-start
    adjoint_error = abs(np.vdot(result, target)-np.vdot(data, factor.apply(target, adjoint=True)))
    return {"case": case, "size": n, "frequency_fraction": frequency_fraction,
            "relative_tolerance": relative_tolerance, "projection_seconds": projection_seconds,
            "geometry_state_time_pairs": n*len(times), "kernel_seconds": kernel_seconds,
            **factor.report, "apply_seconds": apply_seconds, "dense_matvec_seconds": direct_seconds,
            "max_score_error": float(np.max(np.abs(result-direct))), "adjoint_absolute_error": float(adjoint_error)}
