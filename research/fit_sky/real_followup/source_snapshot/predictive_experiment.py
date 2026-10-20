"""Held-out image prediction through a ratio of joint marginal likelihoods."""

import argparse
import json
from pathlib import Path
import numpy as np

from .experiments import real_scene, save
from .integrate import infer, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("training_report", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    training = json.loads(args.training_report.read_text())
    scene, heldout, hashes = real_scene()
    if scene != training["manifest"]:
        raise ValueError("Training data/prior differs from the saved fit")
    scene["exposures"].append(heldout)
    design = []
    for i, exposure in enumerate(scene["exposures"]):
        n = int(np.count_nonzero(~np.asarray(exposure["mask"], bool)))
        design.extend([[1.]+[float(j == i) for j in range(3)]]*n)
    # The additional exposure deviation is independent under the same proper
    # prior. Integrating it out recovers the training model exactly.
    scene["linear_nuisance"] = {"design": design, "mean": [170., 0., 0., 0.],
                                "covariance": np.diag([900., 100., 100., 100.]).tolist()}
    report = infer(scene, method="auto", max_leaves=128, max_seconds=90,
                   queries=["count=0", "has:stationary", "has:moving"])
    save(args.output/"joint.json", report)
    save(args.output/"verified.json", verify(report))
    train_lo, train_hi = training["log_evidence"]
    joint_lo, joint_hi = report["log_evidence"]
    log_bounds = [joint_lo-train_hi if joint_lo is not None and train_hi is not None else None,
                  joint_hi-train_lo if joint_hi is not None and train_lo is not None else None]
    save(args.output/"predictive.json", {
        "log_predictive_density_bounds": log_bounds,
        "definition": "log Z(training + held-out pixels) - log Z(training pixels)",
        "heldout_used_to_choose_prior": False,
        "shared_background_marginalized": True,
        "raw_data_sha256": hashes,
        "training_status": training["status"], "joint_status": report["status"],
        "full_objective_complete": False,
        "scope": "Supplied Gaussian image/PSF model and declared source/background priors",
    })


if __name__ == "__main__":
    main()
