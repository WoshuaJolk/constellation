"""Standalone figure from the archived benchmark results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    narrow = json.loads((root/"atlas-narrow.json").read_text())
    broad = json.loads((root/"atlas-broad.json").read_text())
    comparison = json.loads((root/"stronger-baseline.json").read_text())["results"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    baseline = narrow["direct_geometry_plus_nufft_seconds"]
    fast = narrow["one_shot_seconds_without_certificate"]
    certification = narrow["certificate"]["construction_seconds"]
    ax = axes[0]
    ax.bar([0], [baseline], color="#64748b", label="Exact geometry + frame NUFFTs")
    ax.bar([1], [fast], color="#167d8d", label="Construction + transform")
    ax.bar([1], [certification], bottom=[fast], color="#88c9c1", label="Geometry certification")
    ax.set_xticks([0, 1], ["Exact geometry", "Local atlas"])
    ax.set_ylabel("One-shot time (seconds)")
    ax.set_title("1,048,576 queries · 8 exposures")
    for x, y in [(0, baseline), (1, fast+certification)]:
        ax.text(x, y+1, f"{y:.1f}s", ha="center")
    ax.set_ylim(0, baseline*1.3)
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[1]
    x = [row["frames"] for row in comparison[:3]]
    y = [row["median_construction_inclusive_speedup"] for row in comparison[:3]]
    ax.plot(x, y, "o-", color="#167d8d", linewidth=2)
    ax.axhline(1, color="#64748b", linestyle="--", linewidth=1)
    for a, b in zip(x, y):
        ax.annotate(f"{b:.2f}×", (a, b), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set_xticks(x)
    ax.set_ylim(0, max(y)*1.35)
    ax.set_xlabel("Exposures")
    ax.set_ylabel("Speedup; below 1 is slower")
    ax.set_title("Stronger baseline: same geometry")
    ax.text(.03, .95, "32,768 queries · 64 modes/frame\nCommon certificate cost excluded",
            transform=ax.transAxes, va="top", fontsize=9)
    ax = axes[2]
    errors = [narrow["relative_score_error"], broad["relative_score_error"]]
    ax.bar([0, 1], errors, color=["#167d8d", "#bb5949"])
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["Narrow cell", "Wider cell"])
    ax.set_ylabel("Measured relative score error")
    ax.set_title("Broad coverage remains unresolved")
    ax.set_ylim(1e-12, .2)
    for a, b in enumerate(errors):
        ax.text(a, b*1.7, f"{b:.1e}", ha="center")
    fig.suptitle("fold_kepler: a faster local construction, not the full all-orbit transform", fontsize=15)
    fig.text(.5, .025, "Synthetic tests; fixed two-body model and bounded domains. Geometry bounds are validated; NUFFT arithmetic is not interval-certified.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .92), w_pad=2.5)
    fig.savefig(root/"construction.png", dpi=170)
    fig.savefig(root/"construction.pdf")


if __name__ == "__main__":
    main()
