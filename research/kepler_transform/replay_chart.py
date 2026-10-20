"""Reconstruct archived chart covers and independently replay their checks."""

import argparse
import json
from pathlib import Path

import numpy as np
from flint import arb, ctx

from .chart_atlas import ChartAtlas
from .chart_cover import ChartCover
from .chart_experiments import save
from .continuous import verify_partition
from .geometry_certificate import geometry_digest
from .kernel import KeplerKernel
from .observer_chart import ObserverChart
from .quartic_atlas import QuarticChartAtlas


def restore(report, certificates, arrays, *, replay=False, atlas_class=None):
    verify_partition(report)
    chart = ObserverChart(**report["chart"])
    center, widths = np.asarray(report["center"]), np.asarray(report["widths"])
    reference = chart.states(center[None])[0]
    geometry = KeplerKernel(reference[None], arrays["times"], arrays["observers"],
                            arrays["visits"], arrays["frequencies"], reference)
    geometry.normal, geometry.east, geometry.north = (arrays[name] for name in ("normal", "east", "north"))
    cover = ChartCover.__new__(ChartCover)
    cover.chart, cover.geometry, cover.center, cover.widths = chart, geometry, center, widths
    cover.position_tolerance = report["position_tolerance_radians"]
    cover.outer_degree = report.get("outer_degree", 2)
    cover.phase_tolerance, cover.phase_orders = report.get("phase_tolerance"), {}
    cover.translated_certificate = report.get("translated_certificate", False)
    cover.max_leaves, cover.max_seconds = report.get("max_leaves", 16), report.get("max_seconds", 600.)
    cover.leaves, cover.history = report["leaves"], report["history"]
    cover.construction_seconds = report["construction_seconds"]
    cover.atlases, cover.certificates = {}, certificates
    atlas_class = atlas_class or (QuarticChartAtlas if cover.outer_degree == 4 else ChartAtlas)
    comparisons = []
    with ctx.workprec(128):
        for leaf in cover.leaves:
            bounds = leaf["normalized_bounds"]
            # Physical cell must enclose both EXACT root-affine dyadic
            # endpoints, not merely their rounded binary64 approximations.
            for c, w, limits, cell_c, cell_w in zip(center, widths, bounds, leaf["center"], leaf["widths"]):
                enclosure = arb(float(cell_c), float(cell_w))
                for limit in limits:
                    if not enclosure.contains(arb(float(c))+arb(float(w))*arb(float(limit))):
                        raise ValueError("Physical cell does not enclose the logical partition")
            if leaf["status"] != "resolved":
                continue
            options = {"translated": True} if cover.translated_certificate else {}
            atlas = atlas_class(chart, leaf["center"], leaf["widths"], geometry, **options)
            certificate = certificates[leaf["path"]]
            if atlas.digest != leaf["atlas_sha256"] or atlas.digest != certificate["atlas_sha256"]:
                raise ValueError("Archived atlas digest does not match reconstruction")
            if certificate["geometry_sha256"] != geometry_digest(geometry):
                raise ValueError("Certificate belongs to different exposure geometry")
            if np.max(certificate["position_error_radians"]) > cover.position_tolerance:
                raise ValueError("Resolved cell exceeds requested tolerance")
            cover.atlases[leaf["path"]] = atlas
            if replay:
                current = atlas.certify()
                if np.any(np.asarray(current["position_error_radians"]) >
                          np.asarray(certificate["position_error_radians"])*(1+1e-12)+1e-24):
                    raise ValueError("Current validated certificate is weaker than archived bound")
                comparisons.append({"path": leaf["path"], "certificate_replayed": True,
                                    "current_position_bounds_radians": current["position_error_radians"]})
    return cover, comparisons


def load(directory, case, *, replay=False, atlas_class=None):
    directory = Path(directory)
    report = json.loads((directory/f"{case}-cover.json").read_text())
    certificates = json.loads((directory/f"{case}-certificates.json").read_text())
    with np.load(directory/f"{case}-inputs.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    cover, comparisons = restore(report, certificates, arrays, replay=replay, atlas_class=atlas_class)
    return cover, arrays["data"], comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"cases": {}, "checks": "exact partition, physical box enclosure, input/coefficient binding, interval certificate replay"}
    for case in ["near", "distant", "stress"]:
        cover, _, comparisons = load(args.directory, case, replay=True)
        result["cases"][case] = {"leaves": len(cover.leaves),
                                 "all_cells_resolved": cover.report()["all_cells_resolved"],
                                 "replayed": comparisons}
    save(args.output, result)
    print(json.dumps({case: value["all_cells_resolved"] for case, value in result["cases"].items()}))


if __name__ == "__main__":
    main()
