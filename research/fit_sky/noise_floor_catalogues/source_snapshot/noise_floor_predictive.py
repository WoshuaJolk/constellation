"""Held-out likelihood with simultaneous statistical evidence bounds."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from flint import arb, ctx

from .arithmetic import lower, upper
from .experiments import save
from .noise_floor import build, build_cover, _covariance
from .scene import Scene
from .statistical import infer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("training", type=Path)
    p.add_argument("joint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--samples", type=int, default=2_000_000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot/source.name)
    save(args.output/"contract.json", {
        "samples_per_evidence": args.samples, "seeds": [20261026, 20261027],
        "confidence_per_evidence": .995, "joint_predictive_confidence_lower": .99,
        "optional_stopping": False, "full_objective_complete": False})
    with ctx.workprec(100):
        covers = [json.loads(path.read_text()) for path in (args.training, args.joint)]
        scenes = [Scene(cover["manifest"]) for cover in covers]
        training, joint = scenes
        n = len(training.data)
        if (training.types != joint.types or training.nuisance != joint.nuisance
                or training.manifest["count_prior"] != joint.manifest["count_prior"]
                or training.exposures != joint.exposures[:len(training.exposures)]
                or not np.array_equal(_covariance(training), _covariance(joint)[:n, :n])
                or list(map(float, training.data)) != list(map(float, joint.data[:n]))):
            raise ValueError("Joint model does not retain the training model's marginal")
        logs = []
        for index, (name, scene, cover) in enumerate(zip(["training", "joint"], scenes, covers)):
            print("Constructing", name, flush=True)
            global_floor = build(scene, max_leaves=128, tolerance=1e-4)
            conditional = build_cover(scene, cover, max_leaves=8)
            save(args.output/f"{name}-global.json", global_floor)
            save(args.output/f"{name}-conditional.json", conditional)
            print("Validating and sampling", name, flush=True)
            result = infer(scene.manifest, cover_report=cover, samples=args.samples,
                           seed=20261026+index, confidence=.995, marginalize_flux=True,
                           noise_floor_certificate=global_floor, conditional_noise_certificate=conditional)
            result["manifest"] = scene.manifest
            save(args.output/f"{name}.json", result)
            lo, hi = result["scaled_evidence"]
            if lo <= 0:
                logs.append([None, None])
            else:
                logs.append([lower(arb(lo).log()+scene.log_normalization),
                             upper(arb(hi).log()+scene.log_normalization)])
            print(name, result["status"], result["relative_evidence_gap"], flush=True)
        bounds = ([lower(arb(logs[1][0])-arb(logs[0][1])),
                   upper(arb(logs[1][1])-arb(logs[0][0]))]
                  if all(x is not None for row in logs for x in row) else [None, None])
        save(args.output/"predictive.json", {
            "log_predictive_density_bounds": bounds, "simultaneous_confidence_lower": .99,
            "confidence_combination": "union bound for two fixed-budget 99.5% evidence intervals",
            "definition": "log Z(training + held-out pixels) - log Z(training pixels)",
            "heldout_used_to_choose_prior": False, "shared_background_marginalized": True,
            "conditional_on_stated_model_and_float_sampling_assumptions": True,
            "full_objective_complete": False})
        print("Predictive bounds", bounds, flush=True)
    save(args.output/"inventory.json", {
        str(f.relative_to(args.output)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(args.output.rglob("*")) if f.is_file()})


if __name__ == "__main__":
    main()
