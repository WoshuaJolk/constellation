"""Append observations to an approximate joint FitSky catalogue posterior."""

import argparse
import json
from pathlib import Path

from .workflow import prepare, advance


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
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
    parser.add_argument("--split-merge", action="store_true", default=None,
                        help="Add flux-conserving same-class split/merge proposals")
    parser.add_argument("--exposure-batch", type=int, default=1)
    parser.add_argument("--stop-after", type=int, help="Stop at this exposure prefix and save a checkpoint")
    parser.add_argument("--catalogues", action="store_true", help="Include weighted approximate catalogues")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be a new directory")
    if args.exposure_batch < 1:
        parser.error("--exposure-batch must be positive")
    value = json.loads(args.manifest.read_text())
    settings = {k: getattr(args, k) for k in ("particles", "sweeps", "seed", "pixel_block",
                                           "max_stages", "gibbs_flux", "split_merge")
                if getattr(args, k) is not None}
    try:
        fit, stop = prepare(value, settings, resume=args.resume, stop_after=args.stop_after,
                            exposure_batch=args.exposure_batch)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True)
    (args.output/"input.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    try:
        advance(fit, args.output, stop=stop, exposure_batch=args.exposure_batch,
                catalogues=args.catalogues,
                on_progress=lambda summary: print(json.dumps(summary, allow_nan=False), flush=True))
    except Exception as exc:
        (args.output/"failure.json").write_text(json.dumps({
            "status": "interrupted", "last_complete_exposure": fit.assimilated,
            "reason": str(exc), "full_objective_complete": False,
        }, indent=2, allow_nan=False)+"\n")
        raise


if __name__ == "__main__":
    main()
