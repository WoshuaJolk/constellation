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
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
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
        statistical = integrate_statistically(value["scene"], samples=args.statistical_samples,
                                              queries=value.get("queries"), seed=args.seed,
                                              cover_report=None if args.statistical_prior else report)
        (args.output/"statistical.json").write_text(json.dumps(statistical, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ["status", "full_objective_complete", "queries", "work"]}, indent=2))


if __name__ == "__main__":
    main()
