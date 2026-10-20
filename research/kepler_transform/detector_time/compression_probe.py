"""Check both time-moment backends at the measured detector bandwidth.

Reuses a saved physical workload, not its outputs. All pixel-to-numerator work
is repeated in the changed-image measurement. Information remains fixed.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..temporal.benchmark import source_hashes
from ..timing.benchmark import measure
from .fields import CommonDetector
from .plan import DetectorTimePlan


def run(source, output):
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((source / "results.json").read_text())
    hashes = source_hashes()
    with np.load(source / "inputs.npz") as a:
        images, times, queries = a["images"], a["times"], a["queries"]
        detector = CommonDetector(a["variances"], a["sigma"], a["matrices"], offsets=a["offsets"],
                                  masks=a["masks"], flux_scales=a["flux_scales"],
                                  source_half_width=original["detector_report"]["source_half_width_arcsec"],
                                  max_bytes=4_000_000_000)
    cutoff = original["numerator_report"]["cutoff"]
    n, report = detector.prepare_numerator(images, cutoff)
    proof = original["time_proof"]
    result = {"source_sha256": hashes, "source_workload": str(source),
              "original_workload": original["workload"], "backends": {},
              "pixel_preparation": measure(lambda: detector.prepare_numerator(images, cutoff)[0], 3)}
    expected = None
    evaluator = None
    for backend in ("dense", "nufft"):
        started = perf_counter()
        plan = DetectorTimePlan("near", times, n.shape[1:], detector.spacing,
                                tolerance=proof["requested_tolerance"], compression_backend=backend)
        setup = perf_counter() - started
        if evaluator is None:
            evaluator = plan.query_plan(queries, backend="nufft")
        fields = plan.transform(n)
        actual = evaluator.apply(fields)
        if expected is None:
            expected = actual
        discrepancy = float(max(abs(actual - expected)))
        if discrepancy > plan.score_error_bound(n) + 1e-8:
            raise ValueError("Time moment backends disagree beyond budget")

        def repeat():
            coefficients, budget = detector.prepare_numerator(images, cutoff)
            if budget["numerator_model_error_bound"] + plan.score_error_bound(coefficients) > 2e-5:
                raise ValueError("New input exceeds error budget")
            return evaluator.apply(plan.transform(coefficients))

        timings = {
            "plan_setup_seconds": setup,
            "compression_only": measure(lambda: plan.transform(n), 3),
            "output_only": measure(lambda: evaluator.apply(fields), 3),
            "new_image_numerator_to_scores": measure(repeat, 3),
            "maximum_score_difference_from_dense": discrepancy,
        }
        result["backends"][backend] = timings
        print(json.dumps({"backend": backend, **timings}), flush=True)
    with np.load(source / "scores.npz") as scores:
        count = len(queries)
        result["maximum_error_against_original_direct_pixels"] = float(
            max(abs(expected - scores[f"direct_n_{count}"])))
    if source_hashes() != hashes:
        raise ValueError("Source changed during probe")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
