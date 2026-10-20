"""Two-curve view of measured first-use PFKT versus direct real-pixel costs."""

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
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    rows = report["measurements"]
    assert report["input_exposures"] == 3 and report["compact"]["order"] == 2
    queries = [row["queries"] for row in rows]
    direct = [row["first_direct_seconds"] for row in rows]
    pfkt = [row["first_compact_seconds"] for row in rows]
    error = next(row["max_snr_error"] for row in report["orders"] if row["order"] == 2)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=.11, right=.94, bottom=.25, top=.76)
    fig.text(.11, .93, "Partial FKT vs direct evaluation", fontsize=22, weight="bold")
    fig.text(.11, .86, "Same 3 real ZTF exposures • Increasing number of candidate orbits",
             fontsize=12, color="#475569")
    for values, label, color in [
        (direct, "Direct pixels", "#087F8C"),
        (pfkt, "Partial FKT — 2 time fields", "#DB2777"),
    ]:
        ax.plot(queries, values, "o-", color=color, linewidth=2.7, markersize=5, label=label)
        ax.annotate(f"{values[-1]:.3f} s", (queries[-1], values[-1]), xytext=(-8, 12),
                    textcoords="offset points", ha="right", color=color, weight="bold", fontsize=12)
    ax.set_xlim(0, 70000)
    ax.set_ylim(0, max(direct + pfkt) * 1.18)
    ax.set_xticks([0, 16384, 32768, 49152, 65536])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Number of candidate orbits tested", labelpad=9, fontsize=12)
    ax.set_ylabel("First-use compute time (seconds)", labelpad=10, fontsize=12)
    ax.grid(axis="y", color="#E2E8F0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.11, .075,
             "Includes preparation and access to all scores. Measured points; lines connect them.\n"
             f"PFKT maximum sampled S/N difference: {error:.3g}. Three exposures span 12.3 minutes.\n"
             "This is first use: cached direct evaluation wins when reusing the same geometry.",
             fontsize=10, color="#475569", linespacing=1.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        path = args.output.with_suffix("."+extension)
        fig.savefig(path, dpi=180, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({
        "source": str(args.result),
        "source_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
        "queries": queries, "first_direct_seconds": direct, "first_pfkt_seconds": pfkt,
        "pfkt_time_fields": 2, "maximum_sampled_snr_error": error,
        "scope": "First use including preparation; same three real ZTF exposures.",
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
