"""Plot recorded end-to-end timings; no new timings or extrapolated curves."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=Path(__file__).parent / "results/dense-1024/results.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "results/timing_comparison",
        help="Output stem for PNG, SVG, PDF and plotted-data JSON.",
    )
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    rows = sorted(data["measurements"], key=lambda row: row["queries"])
    queries = [row["queries"] for row in rows]
    series = {
        key: [row["comparison"][key] for row in rows]
        for key in (
            "first_direct_seconds", "first_compressed_seconds",
            "new_image_direct_seconds", "new_image_compressed_seconds",
        )
    }
    workload = data["workload"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#172033",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "svg.hashsalt": "detector-time-comparison",
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    fig.subplots_adjust(left=.075, right=.975, bottom=.27, top=.74, wspace=.22)
    fig.suptitle("Partial FKT vs direct pixel evaluation", x=.075, y=.965,
                 ha="left", fontsize=22, weight="bold")
    fig.text(
        .075, .89,
        f"{workload['frames']:,} simulated {workload['image_side']}×"
        f"{workload['image_side']} exposures · {workload['span_days']:g} days"
        " · both likelihood statistics",
        fontsize=12, color="#475569",
    )
    colors = {"compressed": "#7C3AED", "direct": "#087F8C"}
    panels = (
        ("First use", "first", 34),
        ("New images, reusing preparation", "new_image", 3.05),
    )
    for ax, (title, prefix, upper) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=14, pad=16, weight="bold")
        for method in ("compressed", "direct"):
            values = series[f"{prefix}_{method}_seconds"]
            label = "Partial FKT" if method == "compressed" else "Best measured direct"
            ax.plot(queries, values, marker="o", markersize=7, linewidth=2.4,
                    color=colors[method], label=label, zorder=3)
            for x, y in zip(queries, values):
                # Keep labels above the zero baseline and clear of each curve.
                offset = 13 if method == "compressed" else 10
                ax.annotate(f"{y:.2f} s", (x, y), xytext=(0, offset),
                            textcoords="offset points", ha="center",
                            fontsize=10, color=colors[method], weight="bold")
        ax.set_xlim(0, 4450)
        ax.set_ylim(0, upper)
        ax.set_xticks(queries)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_xlabel("Number of orbit queries", labelpad=12)
        ax.set_ylabel("Compute time (seconds) · lower is better", labelpad=10)
        ax.grid(axis="y", color="#E2E8F0", linewidth=.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.075, .85),
               ncol=2, frameon=False, borderaxespad=0)
    first_ratio = series["first_compressed_seconds"][-1] / series["first_direct_seconds"][-1]
    reuse_ratio = series["new_image_compressed_seconds"][-1] / series["new_image_direct_seconds"][-1]
    fig.text(.075, .12,
             f"At {queries[-1]:,} queries, partial FKT takes {first_ratio:.2f}× as long"
             f" on first use and {reuse_ratio:.2f}× as long on new images.",
             fontsize=12, weight="bold")
    fig.text(.075, .055,
             "Measured points only; connecting lines are visual guides. No extrapolation."
             " Panel scales differ.\n"
             "First use includes preparation and output access. Reuse keeps geometry and"
             " information fixed for both methods; direct caches ≤3 GB.",
             fontsize=9, color="#475569", linespacing=1.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=180,
                    facecolor="white")
    plt.close(fig)
    record = {
        "source": "dense-1024/results.json",
        "source_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "queries": queries, "seconds": series,
        "workload": workload,
        "notes": [
            "Recorded complete comparison times; no new benchmark measurements.",
            "Three measured query counts; connecting lines are not a scaling claim.",
            "Synthetic eight-day stack; not the three-exposure real ZTF replay.",
            "Application timings use recorded medians; setup stages are single measurements.",
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
