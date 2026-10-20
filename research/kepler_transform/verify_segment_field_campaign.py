"""Verify saved provenance and replay the segment field coverage/error checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from flint import ctx

from .chart_experiments import setup
from .factorization import ARCSEC
from .reception_flow import CircularObserver, box_balls, enclose, float_box, reception_cover
from .segment_field import SegmentTransform
from .segment_field_certificate import certify
from .segment_field_experiments import real_pixels


def verify(directory):
    root = Path(__file__).parents[2]
    ctx.prec = 100
    checked = set()
    for filename in ["forward.json", "real.json"]:
        artifact = json.loads((directory/filename).read_text())
        if artifact["full_objective_complete"]:
            raise ValueError("Artifact incorrectly marks the full objective complete")
        for name, digest in artifact["source_sha256"].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source hash mismatch: {name}")
            checked.add(name)
        for experiment in artifact["experiments"]:
            if experiment["status"] == "ran":
                result = experiment["result"]
                if result["full_objective_complete"]:
                    raise ValueError("An experiment incorrectly claims completion")
                for name, digest in result.get("input_sha256", {}).items():
                    if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Input hash mismatch: {name}")
                    checked.add(name)
    covers = []
    observer = CircularObserver()
    for case in ["near", "distant", "stress"]:
        old, center, widths, geometry, _ = setup(case)
        center, widths = reception_cover(old, center, widths)
        proof = {"case": case, "root_center": center.tolist(), "root_widths": widths.tolist(),
                 "endpoints": []}
        for time in [float(geometry.times[0])/2, float(geometry.times[-1])/2]:
            endpoint, _, work = enclose(box_balls(center, widths), 0., time, observer)
            c, w = float_box(endpoint)
            proof["endpoints"].append({"time": time, "center": c.tolist(),
                                       "widths": w.tolist(), "work": work})
        covers.append(proof)
    old, c, w, geometry, _ = setup("near")
    c, w = reception_cover(old, c, w)
    frequencies = np.tile([[.03/ARCSEC, .02/ARCSEC]], (8, 1, 1))
    transform = SegmentTransform(observer, c, w, np.linspace(-4., 4., 8), frequencies,
                                 order=3, exact_leaves=True)
    bound = certify(transform)
    # This intentionally records a failed target, rather than making a loose
    # enclosure look like a useful 1e-8 accuracy guarantee.
    bound["meets_1e_minus_8_target"] = bound["uniform_error_per_source_l1"] <= 1e-8
    real = real_pixels(14)
    if (real["numerator_relative_sup_error"] > 3e-7
            or real["information_relative_sup_error"] > 2e-7):
        raise ValueError("Real-image replay exceeded its recorded regression envelope")
    result = {"provenance_verified": True, "files_hashed": len(checked),
              "recomputed_whole_domain_covers": covers, "recomputed_tensor_error_bound": bound,
              "real_pixel_replay": real, "full_objective_complete": False,
              "cascade_uniform_interpolation_certified": False}
    (directory/"verification.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"files_hashed": len(checked),
                      "tensor_uniform_bound": bound["uniform_error_per_source_l1"],
                      "real_numerator_relative_error": real["numerator_relative_sup_error"],
                      "full_objective_complete": False}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)


if __name__ == "__main__":
    main()
