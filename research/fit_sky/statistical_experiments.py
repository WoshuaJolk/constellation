"""Fixed-budget validation of the complementary global integration route."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .experiments import save, scalar, synthetic, real_scene
from .statistical import infer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deterministic_results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--samples", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--cases", nargs="+", choices=["scalar", "confused", "moving", "broad", "real"],
                   default=["scalar", "confused", "moving", "broad", "real"])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot/source.name)
    contract = {"samples_per_case": args.samples, "seed": args.seed,
                "confidence_per_case": .99, "joint_confidence_across_cases_claimed": False,
                "tolerance": .01, "selection": "fixed budget, no stopping on favorable bounds",
                "full_objective_complete": False}
    save(args.output/"contract.json", contract)
    rows = {}
    for case in args.cases:
        print("Starting", case, flush=True)
        if case == "scalar":
            m, cover_path = scalar(), args.deterministic_results/"scalar-hybrid.json"
        elif case == "real":
            m, _, _ = real_scene()
            cover_path = args.deterministic_results/"real.json"
        else:
            m, _ = synthetic(case)
            cover_path = args.deterministic_results/f"{case}-auto.json"
        # The broad experiment samples the unbounded Poisson prior, not the
        # deterministic campaign's computational count cutoff.
        cover = None if case == "broad" else json.loads(cover_path.read_text())
        queries = ["count=0", "count=1", "count=2"] if case == "confused" else None
        if case == "moving":
            queries = ["has:stationary", "has:moving", "any:moving:vx:0:0.6"]
        result = infer(m, cover_report=cover, samples=args.samples, queries=queries, seed=args.seed)
        result["manifest"] = m
        result["proposal_report_sha256"] = hashlib.sha256(cover_path.read_bytes()).hexdigest() if cover else None
        save(args.output/f"{case}.json", result)
        rows[case] = {k: result[k] for k in ["status", "queries", "relative_evidence_gap", "seconds",
                                            "maximum_sampled_source_count", "full_objective_complete"]}
        save(args.output/"summary.json", rows)
        print(json.dumps(rows[case]), flush=True)
    save(args.output/"inventory.json", {
        str(f.relative_to(args.output)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(args.output.rglob("*")) if f.is_file()})


if __name__ == "__main__":
    main()
