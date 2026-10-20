"""Fixed-budget continuation of the original moving/real campaign."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from flint import ctx
from .scene import Scene
from .noise_floor import build, build_cover
from .statistical import infer
from .experiments import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--samples", type=int, default=8_000_000)
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--cases", nargs="+", choices=["moving", "real"], default=["moving", "real"])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot/source.name)
    save(args.output/"contract.json", {"samples_per_case": args.samples, "seed": args.seed,
         "confidence_per_case": .99, "tolerance": .01, "case_order": args.cases,
         "optional_stopping": False, "full_objective_complete": False})
    rows = {}
    with ctx.workprec(100):
        for case in args.cases:
            path = args.original/("real.json" if case == "real" else "moving-auto.json")
            cover = json.loads(path.read_text())
            m = cover["manifest"]
            scene = Scene(m)
            print("Constructing", case, flush=True)
            global_floor = build(scene, max_leaves=128, tolerance=1e-4, grid=9, iterations=1000)
            save(args.output/f"{case}-global.json", global_floor)
            conditional = build_cover(scene, cover, max_leaves=8, tolerance=1e-5, grid=3, iterations=100)
            save(args.output/f"{case}-conditional.json", conditional)
            queries = ["has:stationary", "has:moving", "any:moving:vx:0:0.6"] if case == "moving" else None
            print("Validating and sampling", case, flush=True)
            result = infer(m, cover_report=cover, noise_floor_certificate=global_floor,
                           conditional_noise_certificate=conditional, samples=args.samples,
                           seed=args.seed, queries=queries)
            result["manifest"] = m
            result["original_cover_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            save(args.output/f"{case}.json", result)
            rows[case] = {k: result[k] for k in ["status", "relative_evidence_gap", "queries", "seconds"]}
            save(args.output/"summary.json", rows)
            print(json.dumps(rows[case]), flush=True)
    save(args.output/"inventory.json", {
        str(f.relative_to(args.output)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(args.output.rglob("*")) if f.is_file()})


if __name__ == "__main__":
    main()
