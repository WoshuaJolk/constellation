"""Stronger baseline: reuse the SAME polynomial geometry, sum per exposure.

This separates cheap orbit generation from genuine time-accumulation savings.
Neither a slower physical integrator nor dense Fourier multiplication may
serve as the only baseline for claiming a fast transform.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from magictools.orbits import GM
from .factorization import ARCSEC, observer_path
from .kernel import KeplerKernel
from .local_transform import LocalTransform, monomials


def polynomial_nufft(plan, data, targets):
    import finufft
    output = np.zeros(len(targets), complex)
    basis = monomials(targets)
    geometry = plan.geometry
    for visit in np.unique(geometry.visits):
        selected = geometry.visits == visit
        positions = basis @ plan.position_coefficients[:, visit, :]/ARCSEC
        frequencies = geometry.frequencies[selected]*ARCSEC
        output += finufft.nufft2d3(np.ascontiguousarray(frequencies[:, 0]),
            np.ascontiguousarray(frequencies[:, 1]), np.ascontiguousarray(data[selected]),
            np.ascontiguousarray(positions[:, 0]), np.ascontiguousarray(positions[:, 1]),
            eps=1e-12, isign=1, nthreads=1)
    return output


def experiment(frames, n, modes_per_frame=64, repetitions=3):
    rng = np.random.default_rng(20261021)
    center = np.array([3., 0., 0., 0., np.sqrt(GM["sun"]/3), 0.])
    inner = np.zeros((3, 6))
    inner[0, 1] = inner[1, 2] = 20*ARCSEC
    inner[2, 4] = 4*ARCSEC
    times = np.linspace(-4, 4, frames)
    visits = np.repeat(np.arange(frames), modes_per_frame)
    frequencies = rng.uniform(-1, 1, (len(visits), 2))/ARCSEC
    geometry = KeplerKernel(center[None], times, observer_path(times), visits, frequencies, center)
    plan = LocalTransform(center, inner, geometry)
    targets = rng.uniform(-1, 1, (n, 3))
    data = rng.normal(size=len(visits))+1j*rng.normal(size=len(visits))
    local_times, baseline_times, maximum_error = [], [], 0.
    # Alternate ordering to reduce systematic warm-cache/order bias.
    for trial in range(repetitions):
        results = {}
        for name in (["local", "baseline"] if trial % 2 == 0 else ["baseline", "local"]):
            start = perf_counter()
            results[name] = plan.apply(data, targets) if name == "local" else polynomial_nufft(plan, data, targets)
            (local_times if name == "local" else baseline_times).append(perf_counter()-start)
        maximum_error = max(maximum_error, float(max(abs(results["local"]-results["baseline"]))))
    return {"frames": frames, "queries": n, "modes_per_frame": modes_per_frame,
            "construction_seconds_common_to_both": plan.report["construction_seconds"],
            "local_seconds": local_times, "same_geometry_baseline_seconds": baseline_times,
            "median_time_accumulation_speedup": float(np.median(baseline_times)/np.median(local_times)),
            "median_construction_inclusive_speedup": float((plan.report["construction_seconds"]+np.median(baseline_times))
                /(plan.report["construction_seconds"]+np.median(local_times))),
            "maximum_score_difference": maximum_error,
            "analytic_phase_tail_bound": plan.phase_tail_bound(data),
            "excludes_certificate_common_to_both": True,
            "baseline": "same fitted polynomial geometry + separate 2D type-3 NUFFT per frame",
            "full_objective_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be new")
    results = []
    for frames, n, modes in [(8, 32768, 64), (32, 32768, 64), (64, 32768, 64), (64, 131072, 64), (32, 32768, 512)]:
        result = experiment(frames, n, modes)
        results.append(result)
        print(json.dumps(result), flush=True)
    value = {"results": results,
             "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    args.output.write_text(json.dumps(value, indent=2)+"\n")


if __name__ == "__main__":
    main()
