"""Run an explicit bounded-domain chart search from a JSON manifest + NPZ.

python -m research.kepler_transform.chart_search manifest.json NEW_OUTPUT_DIR

Research checkout and the research extra are required. This is a separate
experimental command; it does not change the production fold_kepler defaults.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .chart_cover import ChartCover, audit_scores
from .factorization import ARCSEC
from .kernel import KeplerKernel
from .observer_chart import ObserverChart


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def run(manifest_path, output_path):
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Chart-search schema_version must be 1")
    arrays_path = manifest_path.parent/manifest["arrays"]
    with np.load(arrays_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    chart = ObserverChart(**manifest["chart"])
    center, widths = np.asarray(manifest["center"], float), np.asarray(manifest["widths"], float)
    reference = chart.states(center[None])[0]
    geometry = KeplerKernel(reference[None], arrays["times"], arrays["observers"],
                            arrays["visits"], arrays["frequencies"], reference)
    if any(name in arrays for name in ("normal", "east", "north")):
        axes = [np.asarray(arrays[name], float) for name in ("normal", "east", "north")]
        if any(a.shape != (len(geometry.times), 3) or not np.all(np.isfinite(a)) for a in axes):
            raise ValueError("Finite exposure tangent bases required")
        matrices = np.stack(axes, axis=1)
        if not np.allclose(matrices @ matrices.transpose(0, 2, 1), np.eye(3), atol=2e-12, rtol=0):
            raise ValueError("Exposure tangent bases must be orthonormal")
        geometry.normal, geometry.east, geometry.north = axes
    data, parameters = np.asarray(arrays["data"], complex), np.asarray(arrays["parameters"], float)
    if (data.shape != (len(geometry.visits),) or not np.all(np.isfinite(data))
            or parameters.ndim != 2 or parameters.shape[1] != 6 or not len(parameters)
            or not np.all(np.isfinite(parameters))):
        raise ValueError("Finite compatible data and (N,6) parameters required")
    fallback = manifest.get("allow_direct_fallback", False)
    audit_count = manifest.get("audit_count", 3)
    if not isinstance(fallback, bool) or isinstance(audit_count, bool) or not isinstance(audit_count, int) or not 0 <= audit_count <= 32:
        raise ValueError("Boolean fallback and integer audit_count in 0..32 required")
    output_path.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    cover = ChartCover(chart, center, widths, geometry,
        position_tolerance=float(manifest["position_tolerance_arcsec"])*ARCSEC,
        max_leaves=manifest.get("max_leaves", 16), max_seconds=manifest.get("max_seconds", 600.),
        progress=lambda report: save(output_path/"progress.json", report))
    save(output_path/"cover.json", cover.report())
    save(output_path/"certificates.json", cover.certificates)
    report = {"schema_version": 1, "scope": "supplied finite Fourier fields over a bounded observer-chart domain",
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
              "all_geometry_cells_resolved": cover.report()["all_cells_resolved"],
              "full_objective_complete": False, "full_score_error_certified": False}
    try:
        scores, application = cover.apply(data, parameters, allow_direct_fallback=fallback,
                                           route=manifest.get("route", "auto"))
    except ValueError as failure:
        report.update(status="evaluation_refused", reason=str(failure), total_seconds=perf_counter()-started)
        save(output_path/"report.json", report)
        return report
    np.savez_compressed(output_path/"scores.npz", scores=scores)
    indices = np.linspace(0, len(scores)-1, min(audit_count, len(scores))).astype(int)
    audits = audit_scores(chart, geometry, parameters[indices], data, scores[indices]) if len(indices) else []
    report.update(status="evaluated_all_queries", application=application,
                  selected_score_audits=[dict(index=int(i), **a) for i, a in zip(indices, audits)],
                  total_seconds=perf_counter()-started)
    save(output_path/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.manifest, args.output)
    except (ValueError, KeyError, OSError) as failure:
        print(json.dumps({"status": "error", "reason": str(failure)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "evaluated_all_queries" else 2


if __name__ == "__main__":
    raise SystemExit(main())
