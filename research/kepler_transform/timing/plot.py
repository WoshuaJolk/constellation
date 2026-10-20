"""Render measured timings only, with initialization costs shown separately."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(directory):
    result = json.loads((directory / "results.json").read_text())
    cases = result["cases"]
    fig, axes = plt.subplots(2, len(cases), figsize=(12, 8.4), squeeze=False,
                             sharex=True, sharey="row")
    fig.subplots_adjust(top=.84, bottom=.16, left=.09, right=.98, hspace=.25, wspace=.18)
    blue, orange = "#2864a5", "#b95114"
    setup = result["setup"]
    summary = []
    for column, case in enumerate(cases):
        rows = case["measurements"]
        q = np.array([row["orbits"] for row in rows])
        direct = np.array([row["direct"]["median_seconds"] for row in rows])
        query = np.array([row["pfkt_query"]["median_seconds"] for row in rows])
        apply = case["transform_timing"]["median_seconds"]
        pfkt_setup = setup["planning_seconds_single_run"] + case["materialization_seconds_single_run"]
        direct_setup = setup["direct_geometry_seconds_single_run"] + setup["direct_jit_plus_first_call_seconds_single_run"]
        first, ready = axes[:, column]
        first.plot(q, pfkt_setup + apply + query, "o-", color=orange, label="PFKT: plan + build + apply + query")
        first.plot(q, direct_setup + direct, "s--", color=blue, label="Direct: JIT startup + evaluation")
        first.plot(q, direct, "o-", color=blue, label="Direct: compiler already warm")
        first.set_title(f'{case["modes_per_exposure"]} Fourier mode(s) per exposure', fontsize=11)
        ready.plot(q, query, "o-", color=orange, label="PFKT: existing root field, new queries")
        ready.plot(q, apply + query, "s--", color=orange, label="PFKT: existing geometry, new coefficients")
        ready.plot(q, direct, "o-", color=blue, label="Direct: compiler already warm")
        for name, color in [("direct", blue), ("pfkt_query", orange)]:
            ready.fill_between(q, [r[name]["min_seconds"] for r in rows],
                               [r[name]["max_seconds"] for r in rows], color=color, alpha=.12)
        for ax in (first, ready):
            ax.set(xscale="log", yscale="log")
            ax.grid(which="major", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        ready.set_xlabel("Number of queried orbits")
        summary.append({
            "modes_per_exposure": case["modes_per_exposure"],
            "pfkt_plan_and_build_seconds": pfkt_setup,
            "pfkt_apply_seconds": apply,
            "largest_query_count": int(q[-1]),
            "direct_seconds_at_largest_count": float(direct[-1]),
            "pfkt_query_seconds_at_largest_count": float(query[-1]),
            "pfkt_total_seconds_at_largest_count": float(pfkt_setup + apply + query[-1]),
            "pfkt_query_to_direct_ratio_at_largest_count": float(query[-1] / direct[-1]),
            "measured_counts_pfkt_query_faster": [int(n) for n, p, d in zip(q, query, direct) if p < d],
            "maximum_sampled_error_per_coefficient_l1": max(r["sampled_error_per_coefficient_l1"] for r in rows),
        })
    axes[0, 0].set_ylabel("First-use total (seconds)")
    axes[1, 0].set_ylabel("Reuse runtime (seconds)")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[1, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Measured partial FKT vs direct orbit evaluation", fontsize=16, y=.97)
    fig.text(.5, .905, "Same orbit queries and Fourier fields • 2 exposures spanning 0.01728 seconds\n"
             "Small executable regression; not an eight-day or archive-scale forecast",
             ha="center", fontsize=11)
    repeats = len(cases[0]["measurements"][0]["direct"]["samples_seconds"])
    fig.text(.5, .055, f"Points: medians of {repeats} timing batches; shading: observed min–max. No extrapolation.\n"
             "First-use totals sum measured phases; setup timed once. Imports and detector preparation excluded.\n"
             "PFKT queries construct fresh interpolation stencils. Direct uses the compiled all-conic Kepler solver.",
             ha="center", fontsize=9)
    fig.savefig(directory / "timing_comparison.png", dpi=170)
    fig.savefig(directory / "timing_comparison.svg")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
