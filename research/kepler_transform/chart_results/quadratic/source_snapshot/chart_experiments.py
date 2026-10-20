"""Reproduce broad observer-chart coverage and total-cost benchmarks."""

import argparse
import hashlib
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from magictools.orbits import GM
from .chart_cover import ChartCover, audit_scores, physical_kernel
from .continuous import verify_partition
from .factorization import ARCSEC, observer_path
from .geometry_certificate import score_error_bound
from .kernel import KeplerKernel
from .observer_chart import ObserverChart


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def setup(case, quick=False):
    radius, span, frames = (41., 30., 32) if case == "distant" else (3., 8., 8)
    if quick:
        frames = 2
    chart = ObserverChart([1, 0, 0], [0, np.sqrt(GM["sun"]), 0],
                          [1, 0, 0], [0, 1, 0], [0, 0, 1], retarded_epoch=True)
    reference = np.array([radius, 0., 0., 0., np.sqrt(2*GM["sun"]/radius), 0.])
    center = chart.parameters(reference[None])[0]
    width_rho, width_eta = {"near": (.05, .001), "distant": (.00625, .0001), "stress": (.25, .01)}[case]
    widths = np.array([width_rho, width_eta, 2*ARCSEC, 10*ARCSEC, 10*ARCSEC, 2*ARCSEC])
    times = np.linspace(-span/2, span/2, frames)
    visits = np.repeat(np.arange(frames), 32)
    rng = np.random.default_rng(20261022)
    geometry = KeplerKernel(reference[None], times, observer_path(times), visits,
                            rng.uniform(-1, 1, (len(visits), 2))/ARCSEC, reference)
    data = rng.normal(size=len(visits))+1j*rng.normal(size=len(visits))
    data /= np.linalg.norm(data)
    return chart, center, widths, geometry, data


def experiment(case, output, quick=False):
    chart, center, widths, geometry, data = setup(case, quick)
    def progress(report):
        save(output/(case+"-progress.json"), report)
        print(case, "leaves", len(report["leaves"]),
              "resolved", sum(x["status"] == "resolved" for x in report["leaves"]), flush=True)
    cover = ChartCover(chart, center, widths, geometry, position_tolerance=.0005*ARCSEC,
                       max_leaves=4 if quick else 16, max_seconds=180 if case == "stress" else 600,
                       progress=progress)
    report = cover.report()
    assert verify_partition(report)
    save(output/(case+"-cover.json"), report)
    save(output/(case+"-certificates.json"), cover.certificates)
    n = 128 if quick else 131072
    rng = np.random.default_rng(20261023)
    result = {"case": case, "quick": quick, "frames": len(geometry.times), "modes": len(data),
              "cover": report, "input_data_l1": float(sum(abs(data))),
              "full_objective_complete": False, "real_survey_data": False}
    np.savez_compressed(output/(case+"-inputs.npz"), times=geometry.times,
                        observers=geometry.observers, normal=geometry.normal, east=geometry.east,
                        north=geometry.north, visits=geometry.visits, frequencies=geometry.frequencies,
                        data=data, center=center, widths=widths)
    if not report["all_cells_resolved"]:
        points = center+rng.uniform(-.99, .99, (128, 6))*widths
        try:
            cover.apply(data, points)
        except ValueError as failure:
            result["strict_evaluation_refused"] = str(failure)
        actual, fallback = cover.apply(data, points, allow_direct_fallback=True)
        result["fallback"] = fallback
        result["audits"] = audit_scores(chart, geometry, points[:2], data, actual[:2])
        result["benchmark_status"] = "continuous_geometry_budget_failure"
        return result
    times = {"slices": 0., "polynomial": 0., "physical": 0., "chart_mapping": 0.}
    max_error, max_same_error, norm2, error2, queries, bound, tail = 0., 0., 0., 0., 0, 0., 0.
    blocks, counts, audits = [], {"bound": 0, "unbound": 0}, []
    outer_points = np.array(list(product([-.7, .7], repeat=3)))
    for leaf in cover.leaves:
        atlas = cover.atlases[leaf["path"]]
        certificate = cover.certificates[leaf["path"]]
        bound = max(bound, score_error_bound(atlas.slice(np.zeros(3)), certificate, data))
        tail = max(tail, atlas.phase_tail_bound(data))
        for outer_index, outer in enumerate(outer_points):
            inner = rng.uniform(-.99, .99, (n, 3))
            local = np.column_stack((np.tile(outer, (n, 1)), inner))
            parameters = atlas.center+local*atlas.widths
            values, block_times = {}, {}
            for route in (["slices", "polynomial"] if outer_index % 2 == 0 else ["polynomial", "slices"]):
                started = perf_counter()
                values[route], application = cover.apply(data, parameters, route=route)
                block_times[route] = perf_counter()-started
                times[route] += block_times[route]
            started = perf_counter()
            states = chart.states(parameters)
            mapping = perf_counter()-started
            times["chart_mapping"] += mapping
            energy = np.sum(states[:, 3:]**2, axis=1)/2-GM["sun"]/np.linalg.norm(states[:, :3], axis=1)
            counts["bound"] += int(np.sum(energy < 0))
            counts["unbound"] += int(np.sum(energy >= 0))
            started = perf_counter()
            truth = physical_kernel(geometry, states).nufft(data)
            block_times["physical"] = perf_counter()-started+mapping
            times["physical"] += block_times["physical"]
            maximum = float(np.max(abs(values["slices"]-truth)))
            max_error = max(max_error, maximum)
            max_same_error = max(max_same_error, float(np.max(abs(values["slices"]-values["polynomial"]))))
            norm2 += float(np.linalg.norm(truth)**2)
            error2 += float(np.linalg.norm(values["slices"]-truth)**2)
            queries += n
            if outer_index == 0:
                audits.extend(audit_scores(chart, geometry, parameters[:1], data, values["slices"][:1]))
            blocks.append({"leaf": leaf["path"], "outer": outer.tolist(), "queries": n,
                           "seconds": block_times, "maximum_score_difference": maximum})
            save(output/(case+"-blocks.json"), blocks)
            print(case, "block", len(blocks), "queries", queries, "max error", max_error, flush=True)
    # Independent full-6D, nonrepeated outer coordinates: no per-query local NUFFT.
    parameters = center+rng.uniform(-.99, .99, (128 if quick else 32768, 6))*widths
    arbitrary, arbitrary_report = cover.apply(data, parameters)
    start = perf_counter()
    arbitrary_truth = physical_kernel(geometry, chart.states(parameters)).nufft(data)
    arbitrary_report["physical_reference_seconds"] = perf_counter()-start
    arbitrary_report["maximum_score_difference"] = float(max(abs(arbitrary-arbitrary_truth)))
    arbitrary_report["relative_score_difference"] = float(np.linalg.norm(arbitrary-arbitrary_truth)/np.linalg.norm(arbitrary_truth))
    result.update(benchmark_status="completed", queries=queries, sampled_energy_counts=counts,
        seconds=times, blocks=blocks, arbitrary_queries=arbitrary_report, audits=audits,
        maximum_score_difference=max_error, relative_score_difference=float(np.sqrt(error2/norm2)),
        maximum_difference_same_geometry=max_same_error,
        uniform_geometry_score_contribution=bound,
        uniform_analytic_phase_tail_contribution=tail,
        full_arithmetic_score_certified=False,
        total_slice_seconds=cover.construction_seconds+times["slices"],
        total_same_geometry_seconds=cover.construction_seconds+times["polynomial"],
        speedup_physical=times["physical"]/(cover.construction_seconds+times["slices"]),
        speedup_same_geometry=(cover.construction_seconds+times["polynomial"])/(cover.construction_seconds+times["slices"]),
        application_speedup_same_geometry=times["polynomial"]/times["slices"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=["near", "distant", "stress"], default=["near", "distant", "stress"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"quick": args.quick, "cases": {}, "full_objective_complete": False}
    for case in args.cases:
        print("Starting", case, flush=True)
        result["cases"][case] = experiment(case, args.output, args.quick)
        save(args.output/(case+".json"), result["cases"][case])
    result["provenance"] = {"python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "finufft", "python-flint"]},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob("*.py"))}}
    save(args.output/"complete.json", result)


if __name__ == "__main__":
    main()
