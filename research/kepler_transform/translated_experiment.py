"""Try observer-relative interval coordinates on the unchanged widest domain.

This tests proof cost, not a different physical model or relaxed tolerance.
The earlier large benchmark remains the comparison if this attempt does not
improve the cover. Sampled root-polynomial errors are diagnostics, not proofs.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .chart_cover import ChartCover, polynomial_positions, positions_nufft
from .chart_experiments import setup, save
from .compiled_reference import projected_chart
from .continuous import verify_partition
from .factorization import ARCSEC
from .quartic_atlas import QuarticChartAtlas


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    chart, center, widths, geometry, data = setup("stress")
    root = QuarticChartAtlas(chart, center, widths, geometry, translated=True)
    rng = np.random.default_rng(20261027)
    normalized = rng.uniform(-.99, .99, (2048, 6))
    parameters = center+normalized*widths
    exact_positions = projected_chart(chart, parameters, geometry)
    sampled_error = float(np.max(abs(polynomial_positions(root, normalized)-exact_positions))/ARCSEC)
    def progress(report):
        save(output/"progress.json", report)
        print("translated leaves", len(report["leaves"]), "resolved",
              sum(x["status"] == "resolved" for x in report["leaves"]), flush=True)
    cover = ChartCover(chart, center, widths, geometry, position_tolerance=.0005*ARCSEC,
                       max_leaves=16, max_seconds=600., outer_degree=4, phase_tolerance=1e-4,
                       translated_certificate=True, progress=progress)
    assert verify_partition(cover.report())
    result = {"cover": cover.report(),
              "sampled_unsplit_quartic_position_error_arcsec": sampled_error,
              "sampled_error_is_not_a_certificate": True, "full_objective_complete": False}
    if cover.report()["all_cells_resolved"]:
        start = perf_counter()
        actual, application = cover.apply(data, parameters)
        result["query_seconds"] = perf_counter()-start
        truth = positions_nufft(geometry, exact_positions, data)
        result.update(queries=len(parameters), application=application,
                      maximum_score_difference=float(max(abs(actual-truth))))
    result["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(Path(__file__).parent.glob("*.py"))}
    save(output/"certificates.json", cover.certificates)
    save(output/"complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"resolved": result["cover"]["all_cells_resolved"],
                      "leaves": len(result["cover"]["leaves"]),
                      "construction_seconds": result["cover"]["construction_seconds"],
                      "sampled_unsplit_error_arcsec": result["sampled_unsplit_quartic_position_error_arcsec"]}))


if __name__ == "__main__":
    main()
