"""Replay stored continuous-geometry certificates and check artifact provenance.

Run from the repository root:
python -m research.kepler_transform.verify_construction RESULTS_DIRECTORY
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .atlas import GeometryAtlas
from .construction_experiments import setup
from .continuous import verify_partition
from .geometry_certificate import certify_geometry, score_error_bound
from .local_transform import LocalTransform


def verify(directory):
    complete = json.loads((directory/"complete.json").read_text())
    checks = {}
    for source, expected in complete["provenance"]["sources"].items():
        path = Path(source)
        if not path.exists() and "research/" in source:
            path = Path("research"+source.split("research", 1)[1])
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Recorded scientific source changed: {source}")
    for name, result in complete["cases"].items():
        separate = json.loads((directory/(name+".json")).read_text())
        if separate != result:
            raise ValueError(f"Case report differs from completed campaign: {name}")
        if name.startswith("local-") and result.get("certificate"):
            center, basis, geometry, data = setup(result["case"], frames=8)
            plan = LocalTransform(center, basis[3:]*result["inner_width_factor"], geometry)
            actual = certify_geometry(plan)
            for field in ["geometry_sha256", "polynomial_sha256", "position_error_radians",
                          "interpolation_remainder_radians", "coefficient_difference_radians"]:
                if actual[field] != result["certificate"][field]:
                    raise ValueError(f"Local certificate replay differs: {name}/{field}")
            if score_error_bound(plan, actual, data) != result["uniform_geometry_score_bound"]:
                raise ValueError(f"Local response error bound differs: {name}")
            checks[name] = "interval geometry and response-bound replay passed"
        elif name.startswith("atlas-"):
            center, basis, geometry, data = setup(result["case"], frames=result["frames"],
                                                 outer_scale=result["outer_width_factor"])
            atlas = GeometryAtlas(center, basis, geometry)
            actual = atlas.certify()
            for field in ["geometry_sha256", "atlas_sha256", "position_error_radians",
                          "interpolation_remainder_radians", "coefficient_difference_radians"]:
                if actual[field] != result["certificate"][field]:
                    raise ValueError(f"6D certificate replay differs: {name}/{field}")
            if atlas.phase_tail_bound(data) != result["uniform_phase_tail_bound_for_entire_6d_cell"]:
                raise ValueError(f"Uniform six-dimensional phase bound differs: {name}")
            checks[name] = "6D interval geometry and uniform phase-bound replay passed"
        print(f"Checked {name}", flush=True)
    verify_partition(complete["cases"]["constant-atlas-budget"])
    checks["constant-atlas-budget"] = "exact disjoint covering volume passed; unresolved leaves retained"
    with np.load(directory/"pixel-inputs.npz", allow_pickle=False) as data:
        if data["images"].shape != data["variance"].shape or data["images"].shape != data["masks"].shape:
            raise ValueError("Pixel input shape mismatch")
    return {"verified": True, "checks": checks,
            "scope": "replays interval certificates with the recorded implementation; independent mathematical checks are in the tests, not a second proof assistant",
            "full_objective_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError("Verification output must be new")
    result = verify(args.directory)
    text = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
