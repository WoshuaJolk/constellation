"""Verify provenance, atlas coverage and every retained cascade certificate."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter

from flint import ctx

from .cascade_domain import certify_domain
from .reception_flow import CircularObserver


def verify_rectangular_cover(center, widths, patches):
    """Exact rational check of the two-axis atlas cover, including boundaries."""
    def bounds(c, w):
        return [(Fraction(a)-Fraction(b), Fraction(a)+Fraction(b)) for a, b in zip(c, w)]
    root = bounds(center, widths)
    boxes = [bounds(p["center"], p["widths"]) for p in patches]
    if not boxes:
        raise ValueError("Empty atlas")
    for box in boxes:
        for axis in range(2, 6):
            if box[axis][0] > root[axis][0] or box[axis][1] < root[axis][1]:
                raise ValueError("Atlas unexpectedly trims an unsplit coordinate")
    cuts = []
    for axis in [0, 1]:
        a, b = root[axis]
        cuts.append(sorted({a, b, *(max(a, min(b, edge))
                                    for box in boxes for edge in box[axis])}))
    for left, right in zip(cuts[0][:-1], cuts[0][1:]):
        x = (left+right)/2
        for bottom, top in zip(cuts[1][:-1], cuts[1][1:]):
            y = (bottom+top)/2
            if not any(box[0][0] <= x <= box[0][1] and box[1][0] <= y <= box[1][1]
                       for box in boxes):
                raise ValueError("A retained atlas leaves a hole in the original box")
    # The finite union of closed rectangles contains the dense interior, hence
    # its closure too. No floating tolerances or random samples are used.


def verify(directory):
    ctx.prec = 100
    root = Path(__file__).parents[2]
    checked, certificates = set(), []
    artifacts = {}
    begin = perf_counter()
    for name in ["local", "hierarchy"]:
        artifact = json.loads((directory/(name+".json")).read_text())
        artifacts[name] = artifact
        if artifact["full_objective_complete"]:
            raise ValueError("Artifact incorrectly claims full completion")
        for filename, digest in artifact["source_sha256"].items():
            if hashlib.sha256((root/filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source hash mismatch: {filename}")
            checked.add(filename)
    for item in artifacts["local"]["certificates"]:
        certificates.append(item["transfer_certificate"])
    covers = 0
    for item in artifacts["hierarchy"]["hierarchies"]:
        if item["status"] != "ran":
            raise ValueError("Expected every original-domain hierarchy to be constructed")
        graph = item["result"]["graph"]
        verify_rectangular_cover(graph["original_center"], graph["original_widths"], graph["patch_domains"])
        covers += 1
        if not item["result"]["query_adjoint_relative_discrepancy"] < 1e-10:
            raise ValueError("Saved hierarchy adjoint check failed")
        for local in graph["graphs"]:
            for transfer in local["cascade_reports"]:
                certificates.append(transfer["certificate"])
    observer = CircularObserver()
    for index, saved in enumerate(certificates):
        current = certify_domain(observer, saved["parent_center"], saved["parent_widths"],
                                 saved["start"], saved["stop"],
                                 complex_ellipse=saved.get("complex_ellipse"))
        if not current["all_coordinate_hybrids_certified"] or not current["complex_inverse_certified"]:
            raise ValueError("A whole-domain inverse certificate did not reproduce")
        # A rebuilt certificate must cover the *stored* child grid, not merely
        # certify some newly selected (possibly smaller) domain.
        from flint import arb
        delta = list(map(arb, current["normalized_displacement_bounds"]))
        radii = list(map(arb, current["normalized_enclosing_radii"]))
        parent_widths = list(map(arb, saved["parent_widths"]))
        for i in range(6):
            center_shift = abs(arb(saved["child_center"][i])-arb(current["child_center"][i]))/parent_widths[i]
            target_radius = arb(saved["child_widths"][i])/parent_widths[i]+center_shift
            if not target_radius+delta[i] < radii[i]:
                raise ValueError("Replayed inverse box does not contain the stored child targets")
            # Forward image inclusion uses current delta and stored child width.
            # Exact deterministic replay should reproduce these endpoints.
            if current["child_center"][i] != saved["child_center"][i] or current["child_widths"][i] != saved["child_widths"][i]:
                raise ValueError("Child cover changed; regenerate this campaign on the current environment")
        if (index+1) % 20 == 0:
            print(json.dumps({"certificates_replayed": index+1}), flush=True)
    result = {"source_files_hashed": len(checked), "whole_atlases_verified_with_exact_rationals": covers,
              "real_and_complex_certificates_replayed": len(certificates),
              "seconds": perf_counter()-begin, "full_objective_complete": False,
              "floating_point_operator_certified": False}
    (directory/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)


if __name__ == "__main__":
    main()
