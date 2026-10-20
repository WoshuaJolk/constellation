"""Measure scoring subsets of the transform's predefined output grid."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from flint import ctx

from .benchmark import measure
from ..compiled_reference import projected
from ..constructive_experiments import domain
from ..factorization import ARCSEC
from ..ordered.experiments import hashes
from ..ordered.hierarchy import OrderedHierarchyPlan
from ..reception_flow import CircularObserver
from ..segment_experiments import geometry


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    ctx.prec = 100
    source = hashes()
    root = Path(__file__).resolve().parents[3]
    for path in Path(__file__).parent.glob("*.py"):
        source[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    *_, center, widths = domain("near")
    observer = CircularObserver()
    chart = observer.chart(0.)
    times = [-1e-7, 1e-7]
    geom = geometry(times, observer.axes)
    # Compilation is warm for the comparison, and not charged as orbit work.
    started = perf_counter()
    projected(chart.states(np.array([center])), geom)
    jit_seconds = perf_counter() - started
    started = perf_counter()
    plan = OrderedHierarchyPlan(observer, center, widths, times, [1 / ARCSEC] * 2,
                                neighborhood_ellipses=(2., 8., 32.))
    planning = perf_counter() - started
    rng = np.random.default_rng(20261021)
    result = {"source_sha256": source, "planning_seconds_single_run": planning,
              "direct_jit_first_call_seconds_single_run": jit_seconds,
              "plan_stats": plan.stats, "time_span_days": 2e-7,
              "query_type": "random subsets of existing native root-grid nodes",
              "fixed_root_grid_size": plan.stats["root_nodes"],
              "no_extrapolation_beyond_native_grid": True, "cases": []}
    for modes in [1, 64]:
        frequencies = [rng.uniform(-1, 1, (modes, 2)) / ARCSEC for _ in times]
        for k in frequencies:
            k[0] = [1 / ARCSEC, -1 / ARCSEC]
        data = rng.normal(size=2 * modes) + 1j * rng.normal(size=2 * modes)
        started = perf_counter()
        graph = plan.materialize(frequencies, max_nodes=2_000_000, max_entries=30_000_000)
        materialization = perf_counter() - started
        points = graph.grids[id(plan.root)].points()
        indices = rng.permutation(len(points))
        counts = sorted(set([1, 4, 16, 64, 256, 1024, 4096, len(points)]))

        def direct(q):
            blocks = []
            for start in range(0, len(q), 1024):
                pos = projected(chart.states(q[start:start + 1024]), geom)
                output = np.zeros(len(pos), complex)
                for i, k in enumerate(frequencies):
                    output += np.exp(1j * pos[:, i] @ k.T) @ data[i * modes:(i + 1) * modes]
                blocks.append(output)
            return np.concatenate(blocks)

        def direct_kernel(q):
            blocks = []
            for start in range(0, len(q), 1024):
                pos = projected(chart.states(q[start:start + 1024]), geom)
                blocks.append(np.concatenate(
                    [np.exp(1j * pos[:, i] @ k.T) for i, k in enumerate(frequencies)],
                    axis=1,
                ))
            return np.concatenate(blocks)

        values = graph.transform(data)
        rows = []
        for count in counts:
            chosen = indices[:count]
            q = points[chosen]
            reference = direct(q)
            error = float(np.max(abs(values[chosen] - reference)) / np.sum(abs(data)))
            if not np.isfinite(error) or error > plan.error + 1e-12:
                raise ValueError(f"Native-grid regression failed: {error}")
            started = perf_counter()
            kernel = direct_kernel(q)
            kernel_setup = perf_counter() - started
            cached_error = float(np.max(abs(kernel @ data - reference)) / np.sum(abs(data)))
            if not np.isfinite(cached_error) or cached_error > 1e-12:
                raise ValueError("Cached direct reference differs from the direct sum")
            # Always apply to fresh coefficients in the timed call; no cached
            # result is substituted for the transform. The geometry is reused.
            operations = [("direct", lambda: direct(q)),
                          ("pfkt_apply_and_select", lambda: graph.transform(data)[chosen]),
                          ("direct_cached_apply", lambda: kernel @ data)]
            if len(rows) % 2:
                operations.reverse()
            row = {"orbits": count, "sampled_error_per_coefficient_l1": error,
                   "direct_kernel_setup_seconds_single_run": kernel_setup,
                   "direct_kernel_entries": int(kernel.size),
                   "direct_cached_error_per_coefficient_l1": cached_error,
                   **{name: measure(fn) for name, fn in operations}}
            rows.append(row)
            print(json.dumps({"modes": modes, **row}), flush=True)
        result["cases"].append({"modes_per_exposure": modes,
                                "materialization_seconds_single_run": materialization,
                                "measurements": rows})
        (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        del graph
    for name, digest in source.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during native-grid benchmark: {name}")
    print(json.dumps({"saved": str(directory)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory)
