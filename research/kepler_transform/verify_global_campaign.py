"""Check archived artifacts, proof coverage and independently recomputed scores.

With --recompute-proofs, rebuild every interval certificate from its physical
domain. This does not turn the ordinary floating-point transform into an
interval-certified implementation.
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import tempfile

import numpy as np

from .certify_global import certify, exact_error


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory, *, recompute=False):
    repository = Path(__file__).resolve().parents[2]
    manifest = json.loads((directory/"manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        require(hashlib.sha256((repository/name).read_bytes()).hexdigest() == digest,
                f"Hash mismatch: {name}")
    proof_count, execution_count, analysis_cells = 0, 0, 0
    for path in sorted((directory/"proofs").glob("*-rho*.json")):
        value = json.loads(path.read_text())
        require(value["all_hypotheses_certified"] and not value["failures"], path.name)
        depth = value["depth"]
        expected = {(l, target, time)
                    for l in range(depth+1)
                    for target in product(range(2**l), repeat=6)
                    for time in range(2**(depth-l))}
        observed = {(r["level"], tuple(r["target"]), r["time_index"]) for r in value["records"]}
        require(expected == observed and len(expected) == len(value["records"]), "Incomplete pair cover")
        for record in value["records"]:
            for kind in ["real_root", "complex_root"]:
                cells = record[kind]["analysis_cells"]
                volume = 0.
                for cell in cells:
                    bounds = np.asarray(cell["normalized_analysis_bounds"])
                    require(np.all(bounds[:, 0] < bounds[:, 1]) and np.all(abs(bounds) <= 1),
                            "Analysis cell outside declared box")
                    volume += float(np.prod(bounds[:, 1]-bounds[:, 0]))
                    require(cell["image_radius_upper"] < cell["anomaly_disk_radius"]
                            and cell["contraction_upper"] < 1
                            and cell["connected_to_known_reference_emission_root"]
                            and cell["positive_real_retarded_root_derivative"]
                            and cell["regularized_radius_real_part_positive"], "Unproved root/branch")
                require(volume == 64., "Analysis cover has incorrect volume")
                for i, a in enumerate(cells):
                    a = np.asarray(a["normalized_analysis_bounds"])
                    for b in cells[i+1:]:
                        b = np.asarray(b["normalized_analysis_bounds"])
                        require(np.any(np.maximum(a[:, 0], b[:, 0]) >= np.minimum(a[:, 1], b[:, 1])),
                                "Overlapping analysis interiors")
                analysis_cells += len(cells)
        bound = exact_error(value["sufficient_orders"], depth,
                            value["maximum_time_log_modulus_bound"], value["ellipse_parameter"],
                            value["maximum_frequency_phase_amplitudes"])
        require(bound <= 1e-8 and bound == value["exact_arithmetic_kernel_error_bound"],
                "Stored error arithmetic does not reproduce")
        if recompute:
            with tempfile.TemporaryDirectory() as temporary:
                rebuilt = certify(value["case"], depth, value["ellipse_parameter"],
                    Path(temporary)/"proof.json", partition_depth=value["maximum_analysis_partition_depth"])
            for name in ["all_hypotheses_certified", "sufficient_orders",
                         "maximum_time_log_modulus_bound", "maximum_frequency_phase_amplitudes",
                         "exact_arithmetic_kernel_error_bound"]:
                require(rebuilt[name] == value[name], f"Proof replay differs: {path.name}: {name}")
        proof_count += 1
    for path in sorted((directory/"executions").glob("*.json")):
        value = json.loads(path.read_text())
        proof = directory/"proofs"/value["proof_file"]
        require(hashlib.sha256(proof.read_bytes()).hexdigest() == value["proof_sha256"],
                "Execution proof binding differs")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            actual, reference = data["actual"][data["checked_indices"]], data["independent_reference"]
            maximum = float(max(abs(actual-reference)))
            relative = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
            require(np.isclose(maximum, value["maximum_independent_score_error"], rtol=1e-12, atol=1e-25),
                    "Saved independent error does not reproduce")
            require(np.isclose(relative, value["relative_independent_score_error"], rtol=1e-12, atol=1e-25),
                    "Saved relative error does not reproduce")
            require(maximum < value["exact_arithmetic_score_error_bound"],
                    "A sampled floating-point output violates the exact-arithmetic bound")
        require(not value["full_objective_complete"] and not value["fast_floating_point_output_certified"],
                "Unsupported completion/certification flag")
        execution_count += 1
    return {"verified_artifact_and_source_hashes": len(manifest["sha256"]),
            "proofs": proof_count, "analysis_root_cells": analysis_cells,
            "executions": execution_count, "all_interval_proofs_recomputed": recompute,
            "full_objective_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?",
                        default=Path(__file__).parent/"global_results")
    parser.add_argument("--recompute-proofs", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, recompute=args.recompute_proofs), indent=2))


if __name__ == "__main__":
    main()
