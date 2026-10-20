"""Fixed multi-exposure mixed-scene campaign; particle results are uncertified."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np

from .experiments import synthetic, save
from .reference import render
from .sequential import SequentialFit, BlockModel


def mixed_sequence():
    manifest, truth = synthetic("broad")
    rng = np.random.default_rng(20261103)
    exposures = []
    for i, time in enumerate(np.linspace(-2., 2., 27)):
        exposures.append({
            "id": f"additional-{i:02d}", "time": float(time),
            "instrument": "a" if i % 2 == 0 else "b",
            "data": np.zeros((3, 3)).tolist(),
            "psf_sigma": [.55, .65, .75][i % 3], "noise_sigma": .6,
            "parallax_x": float(np.sin(time)), "parallax_y": float(np.cos(time)),
        })
    images = render(exposures, truth["sources"], truth["nuisance"])
    for exposure, image in zip(exposures, images):
        exposure["data"] = (image+rng.normal(0., .6, image.shape)).tolist()
    manifest["exposures"].extend(exposures)
    return manifest, truth


def run(output, *, seed, particles, sweeps, exposures=30, pilot=False, gibbs_flux=False):
    output.mkdir(parents=True, exist_ok=False)
    source = output/"source_snapshot"
    source.mkdir()
    for name in ("sequential.py", "sequential_experiments.py", "scene.py", "arithmetic.py",
                 "reference.py", "experiments.py"):
        shutil.copy2(Path(__file__).parent/name, source/name)
    manifest, truth = mixed_sequence()
    manifest["exposures"] = manifest["exposures"][:exposures]
    queries = ["count=0", "count=1", "count=2", "count=3", "has:stationary", "has:moving",
               "has:galaxy", "any:moving:vx:0:1"]
    save(output/"input.json", {"scene": manifest, "queries": queries, "truth_for_validation_only": truth})
    save(output/"contract.json", {
        "pilot": pilot, "particles": particles, "sweeps": sweeps, "seed": seed,
        "exposures": exposures, "fixed_validation": not pilot, "gibbs_flux": gibbs_flux,
        "accuracy_target": {"posterior_interval_width": .01, "relative_evidence_gap": .01},
        "particle_agreement_is_not_a_certificate": True,
        "python": platform.python_version(), "numpy": np.__version__,
    })
    fit = SequentialFit(manifest, particles=particles, sweeps=sweeps, seed=seed,
                        queries=queries, gibbs_flux=gibbs_flux)
    started = perf_counter()
    try:
        for endpoint in range(1, exposures+1):
            fit.assimilate(endpoint)
            fit.save(output/"checkpoint.npz")
            save(output/"report.json", fit.report())
            if endpoint in (3, 10, 30):
                save(output/f"prefix-{endpoint}.json", fit.report(include_catalogues=True))
            print(json.dumps({"seed": seed, **fit.summary()}), flush=True)
    except Exception as exc:
        save(output/"failure.json", {"type": type(exc).__name__, "reason": str(exc),
                                     "last_complete_exposure": fit.assimilated,
                                     "seconds": perf_counter()-started, "accuracy_target_met": False})
        raise
    finally:
        save(output/"inventory.json", {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "inventory.json"})
    return fit.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--exposures", type=int, choices=range(1, 31), default=30)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--gibbs-flux", action="store_true")
    args = parser.parse_args()
    if not args.pilot and (args.seed not in range(20261110, 20261114)
                           or args.particles != 2048 or args.sweeps != 20 or args.exposures != 30
                           or args.gibbs_flux):
        parser.error("Changed budgets belong to a separate --pilot, not the fixed validation")
    run(args.output, seed=args.seed, particles=args.particles, sweeps=args.sweeps,
        exposures=args.exposures, pilot=args.pilot, gibbs_flux=args.gibbs_flux)


if __name__ == "__main__":
    main()
