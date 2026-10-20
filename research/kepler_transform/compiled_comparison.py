"""Final comparisons against optimized polynomial and compiled direct physics.

Fixed phase operator tolerance: 1e-4 times the input coefficient L1 norm.
Geometry tolerances/domains stay unchanged. All outer points and all queries
are evaluated by each method, with three rotated-order trials.
"""

import argparse
import hashlib
import importlib.metadata
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .chart_cover import positions_nufft
from .chart_experiments import save
from .compiled_reference import projected_chart
from .geometry_certificate import score_error_bound
from .phase_budget import choose_order
from .replay_chart import load, restore


METHODS = ["order_one", "budgeted", "polynomial", "compiled_direct"]


def experiment(cover, data, case, n, repetitions, compile_seconds, checkpoint):
    trials, worst, policies = [], {name: 0. for name in METHODS[:-1]}, {}
    geometry_bound, phase_bound = 0., 0.
    for path, atlas in cover.atlases.items():
        policies[path] = choose_order(atlas, 1e-4)
        order = policies[path]["selected_order"]
        if order is None:
            raise ValueError("No affordable phase order for this declared benchmark")
        phase_bound = max(phase_bound, policies[path]["operator_bounds_by_order"][order]*sum(abs(data)))
        geometry_bound = max(geometry_bound, score_error_bound(atlas.slice(np.zeros(3)), cover.certificates[path], data))
    for trial in range(repetitions):
        rng = np.random.default_rng(20261026)
        seconds = {name: 0. for name in METHODS}
        query_count = 0
        for leaf in cover.leaves:
            atlas = cover.atlases[leaf["path"]]
            for index, outer in enumerate(product([-.7, .7], repeat=3)):
                inner = rng.uniform(-.99, .99, (n, 3))
                parameters = atlas.center+np.column_stack((np.tile(outer, (n, 1)), inner))*atlas.widths
                shift = (trial+index) % len(METHODS)
                values = {}
                for method in METHODS[shift:]+METHODS[:shift]:
                    started = perf_counter()
                    if method == "compiled_direct":
                        values[method] = positions_nufft(cover.geometry,
                            projected_chart(cover.chart, parameters, cover.geometry), data)
                    else:
                        cover.phase_tolerance = 1e-4 if method == "budgeted" else None
                        values[method], _ = cover.apply(data, parameters,
                            route="polynomial" if method == "polynomial" else "slices")
                    seconds[method] += perf_counter()-started
                truth = values["compiled_direct"]
                for method in METHODS[:-1]:
                    worst[method] = max(worst[method], float(max(abs(values[method]-truth))))
                if max(abs(values["budgeted"]-truth)) > geometry_bound+phase_bound+1e-8:
                    raise ValueError("Observed query error exceeds geometry/analytic bounds plus numerical allowance")
                query_count += n
            print(case, "trial", trial+1, "completed leaf", leaf["path"], flush=True)
        trials.append(seconds)
        checkpoint({"case": case, "trial_seconds": trials, "maximum_errors": worst})
        print(case, "trial", trial+1, seconds, flush=True)
    medians = {name: float(np.median([row[name] for row in trials])) for name in METHODS}
    common = cover.construction_seconds
    total = common+medians["budgeted"]
    return {"case": case, "queries_per_trial": query_count, "repetitions": repetitions,
            "trial_seconds": trials, "median_seconds": medians,
            "common_geometry_construction_and_certification_seconds": common,
            "compiled_startup_seconds": compile_seconds,
            "budgeted_total_seconds": total,
            "speedup_vs_compiled_direct_including_compile": (compile_seconds+medians["compiled_direct"])/total,
            "speedup_vs_compiled_direct_excluding_compile": medians["compiled_direct"]/total,
            "speedup_vs_same_geometry": (common+medians["polynomial"])/total,
            "application_speedup_vs_compiled_direct": medians["compiled_direct"]/medians["budgeted"],
            "maximum_score_errors_vs_compiled_direct": worst,
            "uniform_geometry_score_contribution": geometry_bound,
            "uniform_phase_score_contribution": phase_bound, "phase_policies": policies,
            "same_geometry_baseline_reuses_outer_coefficients": True,
            "compiled_direct_uses_chart_emission_epoch_without_extra_propagation": True,
            "formal_full_arithmetic_score_certificate": False, "full_objective_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quadratic_directory", type=Path)
    parser.add_argument("quartic_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--queries-per-outer", type=int, default=131072)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.queries_per_outer < 1 or args.repetitions < 1:
        raise ValueError("New output and positive workload counts required")
    covers = {}
    for case in ["near", "distant"]:
        cover, data, _ = load(args.quadratic_directory, case)
        covers[case] = (cover, data)
    quartic = json.loads((args.quartic_directory/"complete.json").read_text())
    report = quartic["attempts"][-1]
    if not report["all_cells_resolved"]:
        raise ValueError("Quartic broad-domain cover must be complete")
    certificate_path = args.quartic_directory/f"certificates-{report['max_leaves']}.json"
    certificates = json.loads(certificate_path.read_text())
    with np.load(args.quadratic_directory/"stress-inputs.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cover, _ = restore(report, certificates, arrays)
    covers["quartic_stress"] = (cover, arrays["data"])
    started = perf_counter()
    projected_chart(covers["near"][0].chart, covers["near"][0].center[None], covers["near"][0].geometry)
    compilation = perf_counter()-started
    result = {"cases": {}, "compiled_first_call_seconds": compilation,
              "numba_version": importlib.metadata.version("numba"), "serial_no_fast_math": True}
    for case, (cover, data) in covers.items():
        result["cases"][case] = experiment(cover, data, case, args.queries_per_outer,
            args.repetitions, compilation, lambda value: save(args.output.with_suffix(".progress.json"), value))
        save(args.output, result)
    result["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(Path(__file__).parent.glob("*.py"))}
    save(args.output, result)


if __name__ == "__main__":
    main()
