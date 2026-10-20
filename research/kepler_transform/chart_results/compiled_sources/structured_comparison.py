"""Tightened baseline: reuse repeated outer coefficients before summing frames.

The initial campaign's general 6D polynomial evaluator did unnecessary work
when outer coordinates repeated. Both current routes now reuse each slice.
Run this comparison after other computational jobs finish; three alternating
trials, including routing and slice construction, with the common archived
geometry-construction/certification cost charged to both totals.
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .chart_experiments import save
from .replay_chart import load


def experiment(directory, case, n=131072, repetitions=3):
    cover, data, _ = load(directory, case)
    if not cover.report()["all_cells_resolved"]:
        raise ValueError("Complete cover required for this comparison")
    trials, maximum_error = [], 0.
    for trial in range(repetitions):
        rng = np.random.default_rng(20261023)
        seconds = {"slices": 0., "polynomial": 0.}
        queries = 0
        for leaf in cover.leaves:
            atlas = cover.atlases[leaf["path"]]
            for j, outer in enumerate(product([-.7, .7], repeat=3)):
                inner = rng.uniform(-.99, .99, (n, 3))
                parameters = atlas.center+np.column_stack((np.tile(outer, (n, 1)), inner))*atlas.widths
                values = {}
                for route in (["slices", "polynomial"] if (trial+j) % 2 == 0 else ["polynomial", "slices"]):
                    start = perf_counter()
                    values[route], _ = cover.apply(data, parameters, route=route)
                    seconds[route] += perf_counter()-start
                maximum_error = max(maximum_error, float(max(abs(values["slices"]-values["polynomial"]))))
                queries += n
        trials.append(seconds)
        print(case, "trial", trial+1, seconds, flush=True)
    fast = float(np.median([row["slices"] for row in trials]))
    baseline = float(np.median([row["polynomial"] for row in trials]))
    return {"case": case, "queries_per_trial": queries, "repetitions": repetitions,
            "trial_seconds": trials, "median_slice_seconds": fast,
            "median_optimized_polynomial_seconds": baseline,
            "common_geometry_construction_and_certification_seconds": cover.construction_seconds,
            "application_speedup": baseline/fast,
            "total_speedup": (cover.construction_seconds+baseline)/(cover.construction_seconds+fast),
            "maximum_score_difference": maximum_error,
            "baseline_reuses_fixed_outer_coefficients": True,
            "full_arithmetic_error_certified": False, "full_objective_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--queries-per-outer", type=int, default=131072)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.queries_per_outer < 1 or args.repetitions < 1:
        raise ValueError("New output path and positive workload/trial counts required")
    result = {"cases": {}}
    for case in ["near", "distant"]:
        result["cases"][case] = experiment(args.directory, case, args.queries_per_outer, args.repetitions)
        save(args.output, result)
    result["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(Path(__file__).parent.glob("*.py"))}
    save(args.output, result)


if __name__ == "__main__":
    main()
