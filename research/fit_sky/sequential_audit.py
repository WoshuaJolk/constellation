"""Fixed global confidence audit of the 10/30-exposure mixed-scene inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .experiments import save
from .sequential_experiments import mixed_sequence
from .statistical import infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exposures", type=int, choices=[10, 30], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output/"source_snapshot"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source/path.name)
    manifest, _ = mixed_sequence()
    manifest["exposures"] = manifest["exposures"][:args.exposures]
    seed = 20261130 if args.exposures == 10 else 20261131
    queries = ["count=0", "count=1", "count=2", "count=3", "has:stationary",
               "has:moving", "has:galaxy", "any:moving:vx:0:1"]
    save(args.output/"input.json", {"scene": manifest, "queries": queries})
    save(args.output/"contract.json", {"samples": 1_000_000, "confidence": .99,
                                       "seed": seed, "marginalize_flux": True})
    try:
        result = infer(manifest, samples=1_000_000, confidence=.99, queries=queries,
                       seed=seed, marginalize_flux=True)
        save(args.output/"report.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
    except Exception as exc:
        save(args.output/"failure.json", {"type": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        save(args.output/"inventory.json", {
            str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "inventory.json"})


if __name__ == "__main__":
    main()
