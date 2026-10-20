"""Attempt the unchanged wide stress domain with fifth-derivative bounds.

First budget: 16 cells / 600 seconds. If unresolved, make a second full-domain
attempt with 64 cells / 600 seconds, retaining the failed first result.
Queries here validate the construction; they are not a large speed benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .chart_cover import ChartCover, audit_scores, physical_kernel
from .chart_experiments import setup, save
from .continuous import verify_partition
from .factorization import ARCSEC
from .geometry_certificate import score_error_bound


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    chart, center, widths, geometry, data = setup("stress")
    result = {"purpose": "higher-order attempt on the same widest domain and same accuracy tolerance",
              "attempts": [], "full_objective_complete": False}
    for budget in [16, 64]:
        print("Starting quartic budget", budget, flush=True)
        def progress(report):
            save(output/f"progress-{budget}.json", report)
            print("quartic", budget, "leaves", len(report["leaves"]),
                  "resolved", sum(x["status"] == "resolved" for x in report["leaves"]), flush=True)
        cover = ChartCover(chart, center, widths, geometry, position_tolerance=.0005*ARCSEC,
                           max_leaves=budget, max_seconds=600., outer_degree=4, progress=progress)
        report = cover.report()
        assert verify_partition(report)
        result["attempts"].append(report)
        save(output/f"cover-{budget}.json", report)
        save(output/f"certificates-{budget}.json", cover.certificates)
        if report["all_cells_resolved"]:
            break
    if report["all_cells_resolved"]:
        rng = np.random.default_rng(20261025)
        parameters = []
        for leaf in cover.leaves:
            bounds = np.asarray(leaf["normalized_bounds"])
            points = bounds.mean(axis=1)+np.diff(bounds, axis=1).ravel()/2*rng.uniform(-.99, .99, (512, 6))
            parameters.append(center+points*widths)
        parameters = np.concatenate(parameters)
        start = perf_counter()
        actual, application = cover.apply(data, parameters)
        evaluation = perf_counter()-start
        start = perf_counter()
        truth = physical_kernel(geometry, chart.states(parameters)).nufft(data)
        direct = perf_counter()-start
        bounds = [score_error_bound(cover.atlases[leaf["path"]].slice(np.zeros(3)),
                                    cover.certificates[leaf["path"]], data) for leaf in cover.leaves]
        checks = np.arange(0, len(parameters), 512)
        result.update(queries=len(parameters), application=application,
            maximum_score_difference=float(max(abs(actual-truth))),
            relative_score_difference=float(np.linalg.norm(actual-truth)/np.linalg.norm(truth)),
            uniform_geometry_score_contribution=max(bounds),
            selected_full_arithmetic_audits=audit_scores(chart, geometry, parameters[checks], data, actual[checks]),
            query_seconds=evaluation, direct_query_seconds=direct,
            validation_workload_total_speedup=direct/(cover.construction_seconds+evaluation),
            large_workload_speed_advantage_demonstrated=False)
        np.savez_compressed(output/"validation.npz", parameters=parameters, approximate=actual, reference=truth)
    result["all_cells_resolved"] = report["all_cells_resolved"]
    result["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(Path(__file__).parent.glob("*.py"))}
    save(output/"complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"all_cells_resolved": result["all_cells_resolved"],
                      "attempts": len(result["attempts"]), "queries": result.get("queries")}, indent=2))


if __name__ == "__main__":
    main()
