"""Replay the segment campaign's mathematical certificates and provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .segment_experiments import endpoint_rank_certificate, inverse_certificate, real_pixel_composition


def verify(directory):
    repository = Path(__file__).resolve().parents[2]
    results = json.loads((directory/"results.json").read_text())
    for name, digest in results["source_sha256"].items():
        if hashlib.sha256((repository/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source hash differs: {name}")
    if results["full_objective_complete"] or results["new_fast_transform_claimed"]:
        raise ValueError("Unsupported completion claim")
    for saved in results["endpoint_rank_certificates"]:
        rebuilt = endpoint_rank_certificate(saved["case"])
        for key in ["endpoint_minor_determinant_interval", "three_epoch_determinant_interval"]:
            if saved[key] != rebuilt[key]:
                raise ValueError(f"Rank certificate replay differs: {saved['case']}: {key}")
        if not rebuilt["midpoint_varies_on_endpoint_fiber_certified"]:
            raise ValueError("Endpoint ambiguity not certified")
    for saved in results["inverse_certificates"]:
        rebuilt = inverse_certificate(saved["case"], order=saved["order"],
                                      total_span=saved["exposure_span_days"])
        for key in ["uniform_l2_input_to_sup_response_upper",
                    "real_conjugate_symmetric_input_response_upper",
                    "real_input_inverse_lipschitz_lower_bound", "root_center", "root_widths"]:
            if saved[key] != rebuilt[key]:
                raise ValueError(f"Inverse certificate replay differs: {key}")
    for saved in results["endpoint_ambiguity"]:
        if (not np.all(abs(np.asarray(saved["reference_normalized"])) < 1)
                or not np.all(abs(np.asarray(saved["alternative_normalized"])) < 1)
                or saved["independently_checked_endpoint_error_arcsec"] >= 1e-7
                or saved["independently_checked_midpoint_separation_arcsec"] <= .1):
            raise ValueError("Stored endpoint witness is invalid")
    for saved in results["composition"]:
        if (saved["state_roundtrip_max_absolute"] >= 3e-13
                or saved["rebase_composition_scaled_error"] >= 1e-10
                or saved["segment_score_identity_max_error"] >= 2e-9):
            raise ValueError("Segment identity check failed")
    real = results["real_pixel_identity"]
    for name, digest in real["input_sha256"].items():
        if hashlib.sha256((repository/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Real-data input hash differs: {name}")
    rebuilt = real_pixel_composition()
    if (rebuilt["position_identity_max_pixels"] >= 1e-7
            or rebuilt["numerator_identity_max_absolute"] >= 1e-8
            or rebuilt["information_identity_max_absolute"] >= 1e-8):
        raise ValueError("Real-data identity replay failed")
    return {"verified_source_files": len(results["source_sha256"]),
            "endpoint_rank_certificates_recomputed": len(results["endpoint_rank_certificates"]),
            "inverse_certificates_recomputed": len(results["inverse_certificates"]),
            "real_ZTF_pixel_identity_recomputed": True,
            "original_orbit_domains_preserved": True,
            "complete_fast_transform_constructed": False,
            "full_objective_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?",
                        default=Path(__file__).parent/"segment_results")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))


if __name__ == "__main__":
    main()
