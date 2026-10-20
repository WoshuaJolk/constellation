"""Plot native output-grid measurements without extending beyond that grid."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(directory):
    result = json.loads((directory / "results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.5), sharey=True)
    fig.subplots_adjust(top=.67, bottom=.23, left=.09, right=.98, wspace=.16)
    summary = []
    for ax, case in zip(axes, result["cases"]):
        rows = case["measurements"]
        q = np.array([r["orbits"] for r in rows])
        direct = np.array([r["direct"]["median_seconds"] for r in rows])
        partial = np.array([r["pfkt_apply_and_select"]["median_seconds"] for r in rows])
        for key, label, color in [
            ("direct", "Direct: compiled orbit + field evaluation", "#2864a5"),
            ("pfkt_apply_and_select", "PFKT: apply + select; geometry already built", "#b95114"),
            ("direct_cached_apply", "Direct: same orbit responses already cached", "#18815c"),
        ]:
            ax.plot(q, [r[key]["median_seconds"] for r in rows], "o-", label=label, color=color)
            ax.fill_between(q, [r[key]["min_seconds"] for r in rows],
                            [r[key]["max_seconds"] for r in rows], color=color, alpha=.15)
        setup = result["planning_seconds_single_run"] + case["materialization_seconds_single_run"]
        ax.set(xscale="log", yscale="log", xlabel="Number of native grid orbits scored",
               title=f'{case["modes_per_exposure"]} mode(s)/exposure\n'
               f'Full-grid setup: PFKT {setup:.2f} s; direct cache '
               f'{rows[-1]["direct_kernel_setup_seconds_single_run"]:.3f} s')
        ax.grid(which="major", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        faster = q[partial < direct]
        summary.append({
            "modes_per_exposure": case["modes_per_exposure"],
            "pfkt_setup_seconds": setup,
            "first_measured_count_pfkt_faster": int(faster[0]) if len(faster) else None,
            "full_grid_orbits": int(q[-1]),
            "full_grid_direct_seconds": float(direct[-1]),
            "full_grid_pfkt_apply_seconds": float(partial[-1]),
            "full_grid_reuse_speedup": float(direct[-1] / partial[-1]),
            "full_grid_pfkt_first_use_seconds": float(setup + partial[-1]),
            "full_grid_direct_cached_apply_seconds": rows[-1]["direct_cached_apply"]["median_seconds"],
            "full_grid_direct_cache_setup_seconds": rows[-1]["direct_kernel_setup_seconds_single_run"],
            "max_sampled_error_per_coefficient_l1": max(r["sampled_error_per_coefficient_l1"] for r in rows),
        })
    axes[0].set_ylabel("Runtime with geometry / compiler ready (seconds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .87), frameon=False, fontsize=9)
    fig.suptitle("At the transform's own output grid: measured reuse cost", fontsize=15, y=.98)
    fig.text(.5, .91, "2 exposures spanning 0.01728 seconds • 13,464 fixed output nodes • synthetic Fourier fields",
             ha="center", fontsize=10)
    fig.text(.5, .055, "PFKT computes its entire fixed root grid on every call, then selects the requested nodes.\n"
             "Setup is excluded from curves and reported above each panel. A new/larger grid requires new work.\n"
             "Medians of 5 timing batches; shading is min–max. No extrapolation or archive-scale claim.",
             ha="center", fontsize=9)
    fig.savefig(directory / "native_grid_comparison.png", dpi=170)
    fig.savefig(directory / "native_grid_comparison.svg")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
