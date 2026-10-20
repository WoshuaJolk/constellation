"""Summarize retained runs without treating replicate spread as confidence."""

import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
GROUPS = {
    "baseline": list(range(20261110, 20261114)),
    "conditional_flux": list(range(20261120, 20261124)),
    "split_merge_and_flux": list(range(20261140, 20261144)),
}


def validate_inventory(directory):
    inventory = json.loads((directory/"inventory.json").read_text())
    for name, expected in inventory.items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, name
    encoding_path = directory/"encoding.json"
    if encoding_path.exists():
        encoding = json.loads(encoding_path.read_text())
        original = json.loads((directory/"execution-inventory.json").read_text())
        for name, info in encoding.items():
            digest = hashlib.sha256(gzip.decompress((directory/name).read_bytes())).hexdigest()
            assert digest == info["uncompressed_sha256"] == original[info["uncompressed_name"]]
        for name, expected in original.items():
            path = directory/name
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    return len(inventory)


def summarize():
    broad = json.loads((ROOT.parent/"results/broad-inputs.json").read_text())["scene"]
    result = {
        "full_objective_complete": False, "mixed_scene_accuracy_target_met": False,
        "replicate_spread_is_not_a_confidence_interval": True,
        "all_original_priors_and_first_three_images_preserved": True,
        "all_campaign_inputs_identical": True,
        "groups": {}, "audits": {},
    }
    reports = {}
    reference_scene = None
    for group, seeds in GROUPS.items():
        rows, work = {}, []
        reports[group] = []
        for seed in seeds:
            directory = ROOT/"runs"/str(seed)
            checked = validate_inventory(directory)
            report = json.loads((directory/"report.json").read_text())
            assert report["summary"]["assimilated_exposures"] == 30
            assert not report["accuracy_target_met"]
            source = json.loads((directory/"input.json").read_text())["scene"]
            if reference_scene is None:
                reference_scene = source
            assert source == reference_scene
            assert source["exposures"][:3] == broad["exposures"]
            for key in ("count_prior", "source_types", "nuisance"):
                assert source[key] == broad[key]
            history = {h["assimilated_exposures"]: h for h in report["history"]}
            reports[group].append(report)
            work.append({
                "seed": seed, "seconds": report["summary"]["seconds"],
                "stages": report["summary"]["stages"],
                "maximum_sampled_count": max(s["maximum_sampled_count"] for s in report["stages"]),
                "inventory_files_checked": checked, **report["work"],
            })
            for endpoint in (3, 10, 30):
                rows.setdefault(endpoint, []).append(history[endpoint])
        aggregate = {}
        for endpoint, summaries in rows.items():
            log_evidence = [s["log_evidence_estimate"] for s in summaries]
            query_ranges = {q: [min(s["queries"][q]["estimate"] for s in summaries),
                                max(s["queries"][q]["estimate"] for s in summaries)]
                            for q in summaries[0]["queries"]}
            aggregate[str(endpoint)] = {
                "log_evidence_estimates": log_evidence,
                "max_min_evidence_ratio": math.exp(max(log_evidence)-min(log_evidence)),
                "query_estimate_ranges": query_ranges,
                "largest_query_estimate_range": max(hi-lo for lo, hi in query_ranges.values()),
            }
        result["groups"][group] = {"prefixes": aggregate, "work": work}
    for endpoint in (10, 30):
        directory = ROOT/f"audit-{endpoint}"
        validate_inventory(directory)
        report = json.loads((directory/"report.json").read_text())
        audit_input = json.loads((directory/"input.json").read_text())["scene"]
        assert audit_input["exposures"] == source["exposures"][:endpoint]
        assert all(audit_input[k] == source[k] for k in ("count_prior", "source_types", "nuisance"))
        result["audits"][str(endpoint)] = {k: report[k] for k in (
            "status", "relative_evidence_gap", "queries", "samples", "seed",
            "simultaneous_confidence", "seconds", "maximum_sampled_source_count")}
    assert all(a["status"] == "unresolved" for a in result["audits"].values())
    (ROOT/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    table = ["| Method | Exposures | Largest query spread (percentage points) | Evidence estimate max/min |",
             "| --- | ---: | ---: | ---: |"]
    labels = {"baseline": "Baseline", "conditional_flux": "Conditional flux",
              "split_merge_and_flux": "Flux + split/merge"}
    for group, values in result["groups"].items():
        for endpoint, row in values["prefixes"].items():
            table.append(f"| {labels[group]} | {endpoint} | "
                         f"{100*row['largest_query_estimate_range']:.2f} | "
                         f"{row['max_min_evidence_ratio']:.3f} |")
    readme = ROOT/"README.md"
    text = readme.read_text()
    before, rest = text.split("<!-- RESULTS -->", 1)
    _, after = rest.split("<!-- END RESULTS -->", 1)
    readme.write_text(before+"<!-- RESULTS -->\n\n"+"\n".join(table)
                      +"\n\n<!-- END RESULTS -->"+after)
    return result, reports


def plot(reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = ["Birth/death + random walks", "+ conditional flux", "+ split/merge"]
    columns = ["has:moving", "has:stationary", "has:galaxy"]
    column_titles = ["P(at least one moving source)", "P(at least one stationary source)",
                     "P(at least one galaxy)"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), sharex=True, sharey=True)
    colors = ["#225ea8", "#d95f0e", "#238b45", "#88419d"]
    for row, (group, runs) in enumerate(reports.items()):
        for column, query in enumerate(columns):
            ax = axes[row, column]
            for color, report in zip(colors, runs):
                history = report["history"]
                ax.plot([h["assimilated_exposures"] for h in history],
                        [h["queries"][query]["estimate"] for h in history],
                        color=color, linewidth=1.1, alpha=.85)
            ax.set_ylim(-.02, 1.02)
            ax.set_xlim(1, 30)
            ax.set_xticks([1, 10, 20, 30])
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(column_titles[column], fontsize=10)
            if column == 0:
                ax.set_ylabel(titles[row]+"\nEstimated probability")
            if row == 2:
                ax.set_xlabel("Assimilated exposures")
    fig.suptitle("Mixed-sky posterior estimates across independent populations", fontsize=14)
    fig.text(.5, .02, "Each line is one 2,048-particle run. Run agreement is not an accuracy certificate.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .96))
    fig.savefig(ROOT/"posterior-runs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    summary, reports = summarize()
    plot(reports)
    print(json.dumps({g: v["prefixes"] for g, v in summary["groups"].items()}, indent=2))
