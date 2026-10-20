"""Compare identical Fourier-field queries; never extrapolate unallocated graphs."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter
from importlib.metadata import version

import numpy as np
from flint import ctx

from ..compiled_reference import projected
from ..constructive_experiments import domain
from ..factorization import ARCSEC
from ..ordered.experiments import hashes
from ..ordered.hierarchy import OrderedHierarchyPlan
from ..reception_flow import CircularObserver
from ..segment_experiments import geometry


def measure(function, repeats=5, minimum_seconds=.02):
    """Independent batches, with calibrated inner loops for short operations."""
    started = perf_counter()
    result = function()
    pilot = perf_counter() - started
    loops = min(1000, max(1, int(np.ceil(minimum_seconds / max(pilot, 1e-9)))))
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        for _ in range(loops):
            result = function()
        samples.append((perf_counter() - started) / loops)
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite output in timing trial")
    return {
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "loops_per_sample": loops,
    }


def run(directory, counts, modes, repeats):
    directory.mkdir(parents=True, exist_ok=False)
    ctx.prec = 100
    source = hashes()
    root = Path(__file__).resolve().parents[3]
    for path in Path(__file__).parent.glob("*.py"):
        source[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rng = np.random.default_rng(20261020)
    *_, center, widths = domain("near")
    observer = CircularObserver()
    times = np.array([-1e-7, 1e-7])
    started = perf_counter()
    chart = observer.chart(0.)
    geom = geometry(times, observer.axes)
    direct_geometry_seconds = perf_counter() - started
    queries = center + rng.uniform(-.999, .999, (max(counts), 6)) * widths
    started = perf_counter()
    # One genuine first invocation: record JIT plus the small call, separately.
    projected(chart.states(queries[:1]), geom)
    direct_jit_first_call_seconds = perf_counter() - started
    started = perf_counter()
    plan = OrderedHierarchyPlan(
        observer, center, widths, times, [1 / ARCSEC] * 2,
        neighborhood_ellipses=(2., 8., 32.),
    )
    planning_seconds = perf_counter() - started
    (directory / "plan.json").write_text(json.dumps(plan.record(), indent=2) + "\n")
    result = {
        "source_sha256": source,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in
                         ["numpy", "scipy", "numba", "python-flint", "matplotlib"]},
            "threads": {name: os.environ.get(name) for name in
                        ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]},
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        },
        "workload": {
            "case": "original near domain; synthetic Fourier coefficients",
            "time_span_days": float(np.ptp(times)),
            "exposures": len(times),
            "frequency_bound_radians_inverse": [1 / ARCSEC] * 2,
            "requested_error_per_coefficient_l1": 1e-8,
            "certified_exact_arithmetic_error_per_coefficient_l1": plan.error,
            "query_counts": counts,
            "seed": 20261020,
            "query_batch_size_pfkt": 64,
            "query_batch_size_direct": 1024,
            "floating_point_error_certified": False,
            "survey_or_archive_benchmark": False,
        },
        "plan_stats": plan.stats,
        "setup": {
            "planning_seconds_single_run": planning_seconds,
            "direct_geometry_seconds_single_run": direct_geometry_seconds,
            "direct_jit_plus_first_call_seconds_single_run": direct_jit_first_call_seconds,
            "imports_and_detector_preparation_included": False,
        },
        "cases": [],
        "no_unmeasured_query_count_extrapolation": True,
    }
    print(json.dumps({"planning_seconds": planning_seconds, "plan_stats": plan.stats}), flush=True)
    for mode_count in modes:
        frequencies = [rng.uniform(-1, 1, (mode_count, 2)) / ARCSEC for _ in times]
        # Exercise the declared frequency boundary in both workloads.
        for k in frequencies:
            k[0] = [1 / ARCSEC, -1 / ARCSEC]
        data = rng.normal(size=2 * mode_count) + 1j * rng.normal(size=2 * mode_count)
        started = perf_counter()
        graph = plan.materialize(frequencies, max_nodes=2_000_000, max_entries=30_000_000)
        materialization_seconds = perf_counter() - started
        transform_timing = measure(lambda: graph.transform(data), repeats)
        values = graph.transform(data)

        def direct(q):
            output = []
            for start in range(0, len(q), 1024):
                pos = projected(chart.states(q[start:start + 1024]), geom)
                scores = np.zeros(len(pos), complex)
                for visit, freq in enumerate(frequencies):
                    scores += np.exp(1j * pos[:, visit] @ freq.T) @ data[
                        visit * mode_count:(visit + 1) * mode_count]
                output.append(scores)
            return np.concatenate(output)

        def partial(q):
            # Build fresh query stencils each time. No same-orbit query cache.
            return np.concatenate([graph.evaluate(values, q[start:start + 64])
                                   for start in range(0, len(q), 64)])

        rows = []
        for count in counts:
            q = queries[:count]
            actual, reference = partial(q), direct(q)
            error = float(np.max(abs(actual - reference)) / np.sum(abs(data)))
            if not np.isfinite(error) or error > plan.error + 1e-12:
                raise ValueError(f"Query regression failed at {count}: {error}")
            # Alternate timing order to reduce systematic ordering bias.
            tasks = [("direct", lambda: direct(q)), ("pfkt_query", lambda: partial(q))]
            if len(rows) % 2:
                tasks.reverse()
            timings = {name: measure(function, repeats) for name, function in tasks}
            row = {"orbits": count, "sampled_error_per_coefficient_l1": error, **timings}
            rows.append(row)
            print(json.dumps({"modes": mode_count, **row}), flush=True)
        result["cases"].append({
            "modes_per_exposure": mode_count,
            "materialization_seconds_single_run": materialization_seconds,
            "transform_timing": transform_timing,
            "materialized_stats": graph.stats,
            "measurements": rows,
            "frequency_sha256": hashlib.sha256(np.array(frequencies).tobytes()).hexdigest(),
            "coefficient_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        })
        (directory / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        del graph
    for name, digest in source.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during timing campaign: {name}")
    print(json.dumps({"saved": str(directory)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024, 4096])
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.counts + args.modes + [args.repeats]) < 1:
        parser.error("Counts, modes and repeats must be positive")
    run(args.directory, sorted(set(args.counts)), sorted(set(args.modes)), args.repeats)


if __name__ == "__main__":
    main()
