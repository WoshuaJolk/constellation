"""Executable FitSky research inference with explicit model/accuracy scope."""

import argparse
import hashlib
import json
from pathlib import Path

from .integrate import infer, verify
from .sampling import sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-leaves", type=int, default=4096)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--tolerance", type=float, default=.01)
    parser.add_argument("--method", choices=["range", "gauss", "hybrid", "auto"], default="auto")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--statistical-samples", type=int, default=0,
                        help="Fixed number of importance-sampling draws, with confidence bounds")
    parser.add_argument("--statistical-prior", action="store_true",
                        help="Sample the complete prior, including unbounded Poisson counts")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--noise-floor", choices=["global", "conditional", "both"],
                        help="Replay continuous-template likelihood ceilings for Gaussian statistical inference")
    parser.add_argument("--marginalize-flux", action="store_true",
                        help="Analytically integrate one continuous flux in each Gaussian statistical draw")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    if args.noise_floor and (not args.statistical_samples or args.statistical_prior):
        parser.error("--noise-floor requires --statistical-samples and a partition proposal")
    if args.marginalize_flux and not args.statistical_samples:
        parser.error("--marginalize-flux requires --statistical-samples")
    raw = args.manifest.read_bytes()
    value = json.loads(raw)
    args.output.mkdir(parents=True)
    report = infer(value["scene"], queries=value.get("queries"), max_leaves=args.max_leaves,
                   max_seconds=args.max_seconds, tolerance=args.tolerance, method=args.method)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if args.verify:
        (args.output/"verification.json").write_text(json.dumps(verify(report), indent=2, allow_nan=False)+"\n")
    if args.samples:
        (args.output/"samples.json").write_text(json.dumps(sample(report, args.samples), indent=2, allow_nan=False)+"\n")
    if args.statistical_samples:
        from .statistical import infer as integrate_statistically
        global_floor = conditional_floor = None
        if args.noise_floor:
            from flint import ctx
            from .scene import Scene
            from .noise_floor import build, build_cover
            with ctx.workprec(100):
                scene = Scene(value["scene"])
                if args.noise_floor in ("global", "both"):
                    global_floor = build(scene, max_leaves=128, tolerance=1e-4)
                    (args.output/"noise-floor.json").write_text(json.dumps(global_floor, indent=2, allow_nan=False)+"\n")
                if args.noise_floor in ("conditional", "both"):
                    conditional_floor = build_cover(scene, report, max_leaves=8)
                    (args.output/"conditional-noise-floor.json").write_text(json.dumps(conditional_floor, indent=2, allow_nan=False)+"\n")
        statistical = integrate_statistically(value["scene"], samples=args.statistical_samples,
                                              queries=value.get("queries"), seed=args.seed,
                                              tolerance=args.tolerance,
                                              cover_report=None if args.statistical_prior else report,
                                              noise_floor_certificate=global_floor,
                                              conditional_noise_certificate=conditional_floor,
                                              marginalize_flux=args.marginalize_flux)
        (args.output/"statistical.json").write_text(json.dumps(statistical, indent=2, allow_nan=False)+"\n")
    summary = {k: report[k] for k in ["status", "full_objective_complete", "queries", "work"]}
    if args.statistical_samples:
        summary["statistical_status"] = statistical["status"]
        summary["statistical_queries"] = statistical["queries"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
