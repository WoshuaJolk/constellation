"""Compact comparison of arbitrary queries and fixed output-grid queries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(directory):
    arbitrary_path = directory / "arbitrary/results.json"
    native_path = directory / "native/results.json"
    arbitrary = json.loads(arbitrary_path.read_text())
    native = json.loads(native_path.read_text())
    case_a = next(c for c in arbitrary["cases"] if c["modes_per_exposure"] == 64)
    case_n = next(c for c in native["cases"] if c["modes_per_exposure"] == 64)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.2), sharey=True)
    fig.subplots_adjust(top=.74, bottom=.27, left=.09, right=.98, wspace=.2)
    for ax, case, is_native in [(axes[0], case_a, False), (axes[1], case_n, True)]:
        rows = case["measurements"]
        q = [row["orbits"] for row in rows]
        series = [("direct", "Direct evaluation", "#2864a5"),
                  ("pfkt_apply_and_select" if is_native else "pfkt_query",
                   "PFKT: apply + select" if is_native else "PFKT: apply + interpolate", "#b95114")]
        if is_native:
            series.append(("direct_cached_apply", "Direct: cached orbit responses", "#18815c"))
        for key, label, color in series:
            offset = case_a["transform_timing"]["median_seconds"] if key == "pfkt_query" else 0.
            ax.plot(q, [r[key]["median_seconds"] + offset for r in rows], "o-", color=color, label=label)
            ax.fill_between(q, [r[key]["min_seconds"] + offset for r in rows],
                            [r[key]["max_seconds"] + offset for r in rows], color=color, alpha=.12)
        ax.set(xscale="log", yscale="log", xlabel="Number of orbits scored",
               title="Fixed native output grid" if is_native else "Arbitrary orbit queries")
        ax.grid(which="major", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="best")
    axes[0].set_ylabel("Runtime with setup already done (seconds)")
    fig.suptitle("Partial FKT vs direct evaluation: measured compute time", fontsize=15, y=.97)
    fig.text(.5, .88, "64 Fourier modes per exposure • 2 exposures spanning 0.01728 seconds\n"
             "A small executable test, not an archive-scale forecast", ha="center", fontsize=11)
    a_setup = arbitrary["setup"]["planning_seconds_single_run"] + case_a["materialization_seconds_single_run"]
    n_setup = native["planning_seconds_single_run"] + case_n["materialization_seconds_single_run"]
    direct_cache = case_n["measurements"][-1]["direct_kernel_setup_seconds_single_run"]
    fig.text(.5, .12, f"Setup excluded above: PFKT {a_setup:.2f}–{n_setup:.2f} s; "
             f"direct cache for all 13,464 native nodes {direct_cache:.3f} s.\n"
             "PFKT applies the transform on every call. Native-grid queries select existing nodes; arbitrary queries interpolate.\n"
             "Direct caching requires those same fixed orbits and frequencies. No extrapolation beyond measured points.",
             ha="center", fontsize=9)
    fig.savefig(directory / "comparison.png", dpi=170)
    fig.savefig(directory / "comparison.svg")
    (directory / "figure_inputs.json").write_text(json.dumps({
        "inputs_sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [arbitrary_path, native_path]},
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
