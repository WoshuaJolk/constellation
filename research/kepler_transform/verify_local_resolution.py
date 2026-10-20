"""Replay geometry, sensitivity, local ellipse inclusion and tree error/cost."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from flint import arb, ctx

from .cascade_domain import certify_domain
from .local_hierarchy import transfer_cost
from .local_resolution import axis_neighborhood, phase_bound, grid_plan
from .reception_flow import CircularObserver
from .verify_constructive_campaign import verify_rectangular_cover


def replay_proof(proof, observer, options=None):
    current = certify_domain(observer, proof["parent_center"], proof["parent_widths"],
                             proof["start"], proof["stop"], **(options or {}))
    for key in ("child_center", "child_widths", "normalized_enclosing_radii"):
        if current[key] != proof[key]:
            raise ValueError(f"Real domain certificate changed: {key}")
    return current


def replay_grid(grid, proof, observer, *, max_step=.03125):
    for i, (choice, saved) in enumerate(zip(grid["axes"], grid["certificates"])):
        old = saved["neighborhood"]
        current = axis_neighborhood(proof, observer, i, old["ellipse"], max_step=max_step)
        for key in ("enclosing_real_radii", "enclosing_imaginary_radii", "inverse_majorant",
                    "increment_majorant", "hybrid_halfwidth"):
            if current[key] != old[key]:
                raise ValueError(f"Analytic neighborhood changed: {key}")
        sensitivity = phase_bound(proof, current, observer, grid["time_range"],
                                  grid["frequency_bounds"], max_step=max_step)
        if sensitivity["phase_derivative_bound"] > choice["phase_derivative_bound"]:
            raise ValueError("Saved phase bound is too small")
        if choice["hybrid_halfwidth"] < current["hybrid_halfwidth"]:
            raise ValueError("Saved halfwidth is too small")
        if choice["neighborhood_ellipse"] != current["ellipse"]:
            raise ValueError("Grid uses the wrong analytic neighborhood")
        n, p = choice["cells"], choice["order"]
        if (type(n) is not int or n < 1 or type(p) is not int or not 1 <= p <= 64
                or n % choice["cell_multiple"] or choice["axis_nodes"] != n*p):
            raise ValueError("Invalid integer cell/order counts")
        r, r0 = arb(choice["ellipse"]), arb(current["ellipse"])
        if not (choice["ellipse"] <= current["ellipse"] or (
                arb(n)*((r0+1/r0)/2-1) >= (r+1/r)/2-1
                and arb(n)*(r0-1/r0) >= r-1/r)):
            raise ValueError("Local ellipse leaves certified neighborhood")
    current = grid_plan(grid["axes"], grid["tolerance"])
    for key in ("nodes", "cells", "orders", "shape", "error_bound", "input_error_amplification"):
        if current[key] != grid[key]:
            raise ValueError(f"Grid certificate/count changed: {key}")
    return current


def replay_tree(record, observer):
    times, K = record["times"], record["frequency_bounds"]
    options = record["certificate_options"]
    max_step = record["local_options"]["max_step"]
    identity = replay_proof(record["root_identity_certificate"], observer, options)
    root = record["root"]
    if identity["parent_center"] != root["center"] or identity["parent_widths"] != root["widths"]:
        raise ValueError("Root certificate describes a different box")
    counts = {"nodes": 0, "entries": 0, "edges": 0, "grids": 0}
    def visit(node, proof):
        counts["grids"] += 1
        grid = replay_grid(node["grid"], proof, observer, max_step=max_step)
        counts["nodes"] += grid["nodes"]
        if (node["grid"]["time_range"] != [times[node["first"]], times[node["last"]-1]]
                or node["grid"]["frequency_bounds"] != K):
            raise ValueError("Grid does not cover its exposure interval/frequencies")
        if not node["children"]:
            if node["last"]-node["first"] != 1 or node["epoch"] != times[node["first"]]:
                raise ValueError("Leaf is not an instantaneous exposure")
            error = arb(0)
        else:
            if len(node["children"]) != 2 or len(node["edges"]) != 2:
                raise ValueError("Incomplete binary subtree")
            left, right = node["children"]
            if (left["first"] != node["first"] or left["last"] != right["first"]
                    or right["last"] != node["last"]):
                raise ValueError("Exposure tree omits or overlaps exposures")
            errors = []
            for child, edge in zip(node["children"], node["edges"]):
                proof = replay_proof(edge["certificate"], observer, options)
                if (proof["parent_center"] != node["center"] or proof["parent_widths"] != node["widths"]
                        or proof["child_center"] != child["center"] or proof["child_widths"] != child["widths"]
                        or proof["start"] != node["epoch"] or proof["stop"] != child["epoch"]):
                    raise ValueError("Edge certificate does not cover its parent/child boxes")
                child_error = visit(child, proof)
                cg = child["grid"]
                errors.append(arb(cg["error_bound"])+arb(cg["input_error_amplification"])*child_error)
                cost = transfer_cost(grid["shape"], cg["shape"], cg["orders"])
                if cost != edge["cost"]:
                    raise ValueError("Intermediate grid cost changed")
                counts["entries"] += cost["stored_entries"]
                counts["edges"] += 1
            error = max(errors)
        if not error <= arb(node["nodal_budget"]):
            raise ValueError("Subtree budget exceeded")
        if not error <= arb(node["nodal_error_bound_per_source_l1"]):
            raise ValueError("Subtree error understated")
        # Round back to the saved outward bound, matching the declared recurrence.
        return arb(node["nodal_error_bound_per_source_l1"])
    error = visit(root, identity)
    bound = arb(root["grid"]["error_bound"])+arb(root["grid"]["input_error_amplification"])*error
    stats = record["stats"]
    if not bound <= arb(stats["uniform_error_per_source_l1"]) <= arb(stats["requested_tolerance"]):
        raise ValueError("Whole-tree continuous error budget failed")
    if (counts["nodes"] != stats["total_stored_field_nodes"]
            or counts["entries"] != stats["total_transfer_entries"]):
        raise ValueError("Whole-tree counts disagree")
    return counts


def verify(path):
    begin = perf_counter()
    artifact = json.loads(path.read_text())
    root = Path(__file__).parents[2]
    for name, digest in artifact["source_sha256"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source hash mismatch: {name}")
    ctx.prec = artifact["arb_precision_bits"]
    observer = CircularObserver()
    counts = {"grids": 0, "atlas_covers": 0, "entries": 0}
    if "local" in artifact:
        item = artifact["local"]
        proof = replay_proof(item["certificate"], observer)
        replay_grid(item["grid"], proof, observer)
        counts["grids"] = 1
    if "hierarchy" in artifact:
        item = artifact["hierarchy"]["atlas"]
        verify_rectangular_cover(item["original_center"], item["original_widths"], item["patch_domains"])
        if len(item["plans"]) != len(item["patch_domains"]):
            raise ValueError("An atlas patch has no complete hierarchy")
        counts["atlas_covers"] = 1
        for plan, patch in zip(item["plans"], item["patch_domains"]):
            if plan["root"]["center"] != patch["center"] or plan["root"]["widths"] != patch["widths"]:
                raise ValueError("Hierarchy does not cover declared patch")
            result = replay_tree(plan, observer)
            counts["grids"] += result["grids"]
            counts["entries"] += result["entries"]
        if counts["entries"] != item["stats"]["total_transfer_entries"]:
            raise ValueError("Atlas counts disagree")
    if "numerical" in artifact:
        result = replay_tree(artifact["numerical"]["plan"], observer)
        counts.update(result)
    return {**counts, "source_files_hashed": len(artifact["source_sha256"]),
            "seconds": perf_counter()-begin, "full_objective_complete": False,
            "floating_point_evaluation_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    result = verify(args.file)
    args.file.with_name("verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
