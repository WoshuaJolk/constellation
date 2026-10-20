"""Replay archived proofs with their recorded interval implementations.

Run from a research checkout. Numerical refactoring can change interval widths;
historical bounds must be checked with the original expression trees.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .chart_experiments import save
from .phase_budget import choose_order
from .replay_chart import load, restore


def verify(root, output):
    root = Path(root)
    for relative in ["quadratic/source_snapshot", "compiled_sources"]:
        snapshot = root / relative
        for name, expected in json.loads((snapshot / "snapshot.json").read_text())["sources"].items():
            actual = hashlib.sha256((snapshot / Path(name).name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Source snapshot mismatch: {name}")
    old_quadratic = importlib.import_module(
        ".chart_results.quadratic.source_snapshot.chart_atlas", __package__).ChartAtlas
    old_quartic = importlib.import_module(
        ".chart_results.compiled_sources.quartic_atlas", __package__).QuarticChartAtlas
    result = {
        "checks": "snapshot hashes, exact partitions, outward physical cells, coefficient bindings, interval proofs",
        "historical_proofs_use_archived_expression_trees": True,
        "full_objective_complete": False,
        "cases": {},
    }
    covers = {}
    for name in ["near", "distant", "stress"]:
        cover, data, checks = load(root / "quadratic", name, replay=True, atlas_class=old_quadratic)
        result["cases"][name] = {"replayed": checks, "all_cells_resolved": cover.report()["all_cells_resolved"]}
        if name != "stress":
            covers[name] = (cover, data)
        save(output, result)
        print("Replayed", name, flush=True)
    with np.load(root / "quadratic/stress-inputs.npz", allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    quartic = json.loads((root / "quartic/complete.json").read_text())
    translated = json.loads((root / "translated/complete.json").read_text())
    for name, report, certificate_file, atlas_class in [
        ("quartic_stress", quartic["attempts"][-1], root / "quartic/certificates-16.json", old_quartic),
        ("translated_stress", translated["cover"], root / "translated/certificates.json", None),
    ]:
        certificates = json.loads(certificate_file.read_text())
        cover, checks = restore(report, certificates, arrays, replay=True, atlas_class=atlas_class)
        result["cases"][name] = {"replayed": checks, "all_cells_resolved": cover.report()["all_cells_resolved"]}
        covers[name] = (cover, arrays["data"])
        save(output, result)
        print("Replayed", name, flush=True)
    # Planning was cached in median application trials. Measure and charge it
    # separately rather than presenting that median as a fully cold command.
    timing = json.loads((root / "compiled-comparison.json").read_text())
    result["planning_supplement"] = {}
    for name, (cover, data) in covers.items():
        if name not in timing["cases"]:
            continue
        trials = []
        for _ in range(3):
            start = perf_counter()
            policies = {path: choose_order(atlas, 1e-4) for path, atlas in cover.atlases.items()}
            trials.append(perf_counter() - start)
        phase_bound = max(atlas.phase_tail_bound(data, policies[path]["selected_order"])
                          for path, atlas in cover.atlases.items())
        row = timing["cases"][name]
        planning = float(np.median(trials))
        total = row["budgeted_total_seconds"] + planning
        result["planning_supplement"][name] = {
            "planning_trial_seconds": trials,
            "planning_median_seconds": planning,
            "total_including_geometry_and_phase_planning_seconds": total,
            "speedup_vs_warm_compiled_direct": row["median_seconds"]["compiled_direct"] / total,
            "analytic_phase_score_bound_outward": phase_bound,
            "note": "Post-run planning measurement; original timing file unchanged. Analytic bound excludes fast arithmetic.",
        }
        save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "chart_results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.directory, args.output)


if __name__ == "__main__":
    main()
