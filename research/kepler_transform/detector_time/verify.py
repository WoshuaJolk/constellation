"""Replay detector/time certificates and selected independent pixel scores."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..factorization import ARCSEC
from .fields import CommonDetector, trim_fields
from .plan import DetectorTimePlan, domain_source_box


def verify(directory, recompute=False):
    result = json.loads((directory / "results.json").read_text())
    root = Path(__file__).resolve().parents[3]
    for name, digest in result["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed: {name}")
    for name, digest in result.get("input_sha256", {}).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input changed: {name}")
    for name, digest in result.get("artifact_sha256", {}).items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact changed: {name}")
    if "real_exposures" in result:
        with np.load(directory / "scores.npz") as a:
            nerror = float(max(abs(a["numerator"] - a["affine_numerator"])))
            ierror = float(max(abs(a["information"] - a["affine_information"])))
            if nerror > result["numerator_analytic_model_budget"] + 1e-10 or ierror > result["information_analytic_model_budget"] + 1e-11:
                raise ValueError("Real-pixel score discrepancy exceeds model budget")
        return {"directory": str(directory), "real_pixel_vectors_checked": result["queries"],
                "input_hashes_checked": len(result["input_sha256"])}
    replay = domain_source_box("near")
    if replay != result["source_domain_proof"]:
        raise ValueError("Source-domain proof changed")
    with np.load(directory / "inputs.npz") as a, np.load(directory / "scores.npz") as scores:
        detector = CommonDetector(a["variances"], a["sigma"], a["matrices"], offsets=a["offsets"],
                                  masks=a["masks"], flux_scales=a["flux_scales"],
                                  source_half_width=result["detector_report"]["source_half_width_arcsec"],
                                  max_bytes=4_000_000_000)
        if detector.template_reports != result["detector_report"]["exposures"] and any(
                detector.template_reports[j][key] != r[key]
                for j, r in enumerate(result["detector_report"]["exposures"])
                for key in detector.template_reports[j]):
            raise ValueError("Detector template bound changed")
        plans = []
        for name, side in [("time_proof", result["workload"]["retained_fourier_side"]),
                            ("information_time_proof", result["workload"]["information_fourier_side"])]:
            proof = result[name]
            plan = DetectorTimePlan("near", a["times"], (side, side), detector.spacing,
                                    tolerance=proof["requested_tolerance"],
                                    rho=proof["ellipse_parameter"],
                                    compression_backend=result["workload"]["compression_backend"])
            for key in proof:
                if key != "construction_seconds" and plan.proof[key] != proof[key]:
                    raise ValueError(f"Time proof changed: {name}/{key}")
            plans.append(plan)
        recomputed = 0
        for row in result["measurements"]:
            count = row["queries"]
            for label, budget in [("n", result["error_budgets"]["numerator"]),
                                   ("i", result["error_budgets"]["information"])]:
                error = max(abs(scores[f"fast_{label}_{count}"] - scores[f"direct_{label}_{count}"]))
                if error > budget + 1e-8:
                    raise ValueError("Saved whole-vector likelihood comparison failed")
        if recompute:
            n, _ = detector.prepare_numerator(a["images"], result["numerator_report"]["cutoff"])
            full_n, full_i, _ = detector.prepare(a["images"])
            _, i, trim = trim_fields(full_n, full_i, [1e-5, 1e-7])
            if trim["retained_side"] != result["trim"]["retained_side"]:
                raise ValueError("Information cutoff changed")
            fields = [p.transform(c) for p, c in zip(plans, (n, i))]
            count = max(r["queries"] for r in result["measurements"])
            index = np.unique(np.r_[np.arange(min(65, count)),
                                     np.linspace(0, count - 1, min(16, count), dtype=int)])
            q = a["queries"][index]
            actual = [p.evaluate(f, q, backend="nufft") for p, f in zip(plans, fields)]
            direct = detector.direct(a["images"],
                                     (plans[0].positions(q, a["times"]) - plans[0].center_input) / ARCSEC)
            for k, label in enumerate(["n", "i"]):
                budget = result["error_budgets"]["numerator" if k == 0 else "information"]
                if max(abs(actual[k] - direct[k])) > budget + 1e-8:
                    raise ValueError("Recomputed score comparison failed")
                if max(abs(actual[k] - scores[f"fast_{label}_{count}"][index])) > 1e-7:
                    raise ValueError("Recomputed scores differ from retained scores")
            recomputed = len(index)
    return {"directory": str(directory), "source_and_two_time_proofs_replayed": True,
            "detector_template_bounds_replayed": detector.frames,
            "whole_score_vectors_checked": len(result["measurements"]) * 2,
            "recomputed_queries": recomputed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    for directory in args.directories:
        print(json.dumps(verify(directory, args.recompute)), flush=True)
