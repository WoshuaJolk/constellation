"""Plot measured totals; never extrapolate beyond measured workloads."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory):
    datasets = [(p.parent.name, json.loads(p.read_text()))
                for p in sorted(directory.glob("*/results.json"))]
    if not datasets:
        raise ValueError("No benchmark result directories")
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 3.1 * len(datasets)), squeeze=False)
    for row, (name, result) in enumerate(datasets):
        w = result["workload"]
        for column, (direct, compressed, title) in enumerate([
            ("first_use_direct", "first_use_compressed", "First use: setup + images + outputs"),
            ("changed_input_fixed_queries_direct", "changed_input_fixed_queries_compressed",
             "New images: both methods reuse query caches"),
        ]):
            ax = axes[row, column]
            x = [r["queries"] for r in result["measurements"]]
            for field, label, color in [(direct, "Best measured direct method", "#c45e24"),
                                        (compressed, "Best measured time compression", "#227c9d")]:
                y = [1000 * r["comparison"][field]["seconds"] for r in result["measurements"]]
                ax.loglog(x, y, "o-", label=label, color=color)
            ax.set_title(f"{w['case']}, {w['span_days']:g} days, {w['exposures']} images, "
                         f"{w['image_side']}×{w['image_side']} pixels\n{title}", fontsize=10)
            ax.set_xlabel("Orbit queries")
            ax.set_ylabel("Milliseconds")
            ax.grid(alpha=.2, which="both")
            if row == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Certified time compression: measured common-grid synthetic workloads\n"
                 "Entire original orbit boxes • all input FFTs charged • no orbit grid",
                 fontsize=13, y=1.002)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"comparison.{suffix}", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot(parser.parse_args().directory)
