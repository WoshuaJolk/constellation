"""Replay interval certificates, input/output checks, and recorded provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .transform import TimePlan, certificate, image_coefficients


def verify(directory, *, recompute=False):
    result = json.loads((directory / "results.json").read_text())
    proof = json.loads((directory / "certificate.json").read_text())
    root = Path(__file__).resolve().parents[3]
    for name, digest in result["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source mismatch: {name}")
    for name, digest in result["artifact_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact mismatch: {name}")
    replay = certificate(proof["case"], proof["tolerance"], proof["ellipse_parameter"])
    for key in proof:
        if key != "construction_seconds" and replay[key] != proof[key]:
            raise ValueError(f"Certificate replay mismatch: {key}")
    with np.load(directory / "inputs.npz") as inputs, np.load(directory / "scores.npz") as scores:
        images, times, queries = inputs["images"], inputs["times"], inputs["queries"]
        coefficients = image_coefficients(images)
        if not np.all(np.isfinite(coefficients)) or np.any(abs(queries) > 1):
            raise ValueError("Invalid inputs or query box")
        l1 = float(np.sum(abs(coefficients)))
        plan = TimePlan(proof["case"], times, images.shape[1:], rho=proof["ellipse_parameter"],
                        tolerance=proof["tolerance"], compression_backend="nufft")
        max_error, recomputed = 0., 0
        fields = plan.transform(coefficients) if recompute else None
        for row in result["measurements"]:
            count = row["queries"]
            if row["counts"] != TimePlan.counts(plan, count) | {
                # The benchmark stores dense-plan counts while executing both.
                "compression_backend": "dense", "dense_compression_count_is_executed": True,
                "interpolation_real_entries": len(times) * plan.order,
            }:
                raise ValueError("Integer cost count mismatch")
            a, d = scores[f"compressed_{count}"], scores[f"direct_{count}"]
            if a.shape != (count,) or d.shape != (count,) or not np.all(np.isfinite(a + d)):
                raise ValueError("Invalid score arrays")
            error = float(max(abs(a - d)) / l1)
            if error > proof["exact_arithmetic_kernel_error_bound"] + 1e-11:
                raise ValueError("Saved score check failed")
            max_error = max(error, max_error)
            if recompute:
                # Full saved vectors were compared above; recompute all corners
                # plus spread-out interior queries through the executable paths.
                index = np.unique(np.r_[np.arange(min(64, count)),
                                         np.linspace(0, count - 1, min(16, count), dtype=int)])
                q = queries[index]
                fresh = plan.evaluate(fields, q, backend="nufft")
                direct = plan.direct_plan(q, backend="stream").apply(coefficients)
                if max(abs(fresh - direct)) / l1 > proof["exact_arithmetic_kernel_error_bound"] + 1e-11:
                    raise ValueError("Recomputed score check failed")
                if max(abs(fresh - a[index])) / l1 > 1e-10 or max(abs(direct - d[index])) / l1 > 1e-10:
                    raise ValueError("Recomputed outputs do not match retained outputs")
                recomputed += len(index)
    return {"directory": str(directory), "certificate_replayed": True,
            "source_files_verified": len(result["source_sha256"]),
            "whole_saved_vectors_checked": len(result["measurements"]),
            "recomputed_queries": recomputed, "maximum_saved_error_per_l1": max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    for directory in args.directories:
        print(json.dumps(verify(directory, recompute=args.recompute)), flush=True)


if __name__ == "__main__":
    main()
