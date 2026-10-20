"""Append observations to an approximate joint FitSky catalogue posterior."""

import argparse
import json
from pathlib import Path

from .sequential import SequentialFit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON with scene and optional queries")
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--resume", type=Path, help="Completed checkpoint; existing data and priors must match")
    parser.add_argument("--particles", type=int)
    parser.add_argument("--sweeps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pixel-block", type=int)
    parser.add_argument("--max-stages", type=int)
    parser.add_argument("--gibbs-flux", action="store_true", default=None,
                        help="Add conditional Gaussian flux draws during rejuvenation")
    parser.add_argument("--exposure-batch", type=int, default=1)
    parser.add_argument("--stop-after", type=int, help="Stop at this exposure prefix and save a checkpoint")
    parser.add_argument("--catalogues", action="store_true", help="Include weighted approximate catalogues")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    if args.exposure_batch < 1:
        parser.error("--exposure-batch must be positive")
    value = json.loads(args.manifest.read_text())
    settings = {k: getattr(args, k) for k in ("particles", "sweeps", "seed", "pixel_block", "max_stages", "gibbs_flux")
                if getattr(args, k) is not None}
    if args.resume:
        fit = SequentialFit.load(args.resume, value["scene"])
        if any(fit.settings[k] != v for k, v in settings.items()):
            parser.error("Resume uses checkpoint settings; conflicting settings require a fresh fit")
        if "queries" in value:
            fit.model.events(fit.population, value["queries"])
            fit.queries = value["queries"]
    else:
        fit = SequentialFit(value["scene"], queries=value.get("queries"), **settings)
    stop = len(value["scene"]["exposures"]) if args.stop_after is None else args.stop_after
    if not fit.assimilated < stop <= len(value["scene"]["exposures"]):
        parser.error("--stop-after must append at least one available exposure")
    args.output.mkdir(parents=True)
    (args.output/"input.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    try:
        while fit.assimilated < stop:
            fit.assimilate(min(stop, fit.assimilated+args.exposure_batch))
            fit.save(args.output/"checkpoint.npz")
            (args.output/"report.json").write_text(
                json.dumps(fit.report(include_catalogues=args.catalogues), indent=2, allow_nan=False)+"\n")
            print(json.dumps(fit.summary(), allow_nan=False), flush=True)
    except Exception as exc:
        (args.output/"failure.json").write_text(json.dumps({
            "status": "interrupted", "last_complete_exposure": fit.assimilated,
            "reason": str(exc), "full_objective_complete": False,
        }, indent=2, allow_nan=False)+"\n")
        raise


if __name__ == "__main__":
    main()
