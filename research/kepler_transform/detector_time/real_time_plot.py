"""Plot measured real-pixel PFKT accuracy and cost, without extrapolation."""

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
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.3))
    fig.subplots_adjust(left=.065, right=.98, top=.76, bottom=.30, wspace=.31)
    fig.suptitle("PFKT time representation on actual ZTF pixels", x=.065, ha="left",
                 y=.97, fontsize=20, weight="bold")
    fig.text(.065, .87,
             f"3 real 9×9 exposures over 12.3 minutes · recorded observer positions · "
             f"{report['queries']:,} tested orbits + 8,192 holdout orbits", fontsize=11)
    orders = report["orders"]
    axes[0].semilogy([r["order"] for r in orders], [r["max_snr_error"] for r in orders],
                     "o-", color="#7C3AED", label="Max sampled S/N error")
    axes[0].axhline(report["threshold_max_absolute_snr_error"], color="#64748B",
                   linestyle="--", label="Sampled acceptance threshold")
    axes[0].axvline(3, color="#C36216", linestyle=":", label="Input exposure count")
    axes[0].set_title("Accuracy against direct pixels", loc="left")
    axes[0].set_xlabel("Number of time fields")
    axes[0].set_ylabel("Maximum absolute S/N difference")
    axes[0].legend(frameon=False, fontsize=8)
    rows = report["measurements"]
    x = [r["queries"] for r in rows]
    for ax, prefix, title in zip(axes[1:], ["first", "reuse"],
                                ["First use, including preparation", "New pixels, reusing preparation"]):
        for method, name, color in [
            ("direct", "Direct pixels (cache when useful)", "#087F8C"),
            ("adapter", "Fourier adapter only", "#C36216"),
            ("pfkt", f"PFKT ({report['selected_order']} time fields)", "#7C3AED"),
            ("compact", f"PFKT ({report['compact']['order']} fields, looser tolerance)", "#DB2777"),
        ]:
            y = [r[f"{prefix}_{method}_seconds"] for r in rows]
            ax.plot(x, y, "o-", color=color, label=name, linewidth=2, markersize=4)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Number of orbit queries")
        ax.set_ylabel("Compute time (seconds)")
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 32768, 65536])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", color="#E2E8F0")
        ax.set_axisbelow(True)
    selected = report["selected_order"]
    outcome = "reduces" if selected < 3 else "does not reduce"
    fig.text(.065, .06,
             f"Sampled S/N targets: 1e-8 needs {selected} fields ({outcome} input count); "
             f"1e-4 needs {report['compact']['order']} fields.\n"
             "Accuracy is empirical on this short stack; no whole-orbit-domain time certificate or long-span real-survey result.\n"
             "Three application samples; setup measured once. Timings exclude research order selection and shared input loading/JIT.",
             fontsize=10, color="#475569", linespacing=1.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(args.output.with_suffix("."+ext), dpi=180, facecolor="white")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({
        "source": str(args.result), "source_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
        "selected_order": selected, "orders": orders,
        "measurements": [{k:v for k,v in row.items() if k.endswith("_seconds") or k=="queries"}
                         for row in rows],
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
