"""Machine-readable CLI; all scientific assumptions live in the input manifests."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .io import run_nulls, run_photometry


def main(argv=None):
    parser = argparse.ArgumentParser(description="Moving-source astronomy tools for local FITS data")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["force_track", "dig_floor", "learn_null", "map_cover", "weigh_stamp"]:
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("-o", "--output", type=Path, help="Write new JSON file (default: stdout)")
        if name == "dig_floor":
            command.add_argument("--alpha", type=float, default=0.01, help="Family-wise Gaussian false-positive threshold")
        if name == "map_cover":
            command.add_argument("--photometry-output", type=Path, help="Export a ready-to-run force_track/dig_floor manifest")
            command.add_argument("--planning-output", type=Path, help="Export a learn_null manifest with future candidates, no assumed non-detections")
    demo = commands.add_parser("demo", help="Generate synthetic FITS data and run all three tools")
    demo.add_argument("directory", type=Path)
    demo.add_argument("--seed", type=int, default=2026)
    orbit_demo = commands.add_parser("orbit_demo", help="Generate an orbit-family coverage and artifact-model experiment")
    orbit_demo.add_argument("directory", type=Path)
    orbit_demo.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        if args.command == "orbit_demo":
            from .orbit_demo import create_orbit_demo
            result = create_orbit_demo(args.directory, args.seed)
        elif args.command == "demo":
            from .demo import create_demo
            result = create_demo(args.directory, args.seed)
        elif args.command == "learn_null":
            result = run_nulls(args.manifest)
        elif args.command == "map_cover":
            from .coverage import run_cover
            # Check all output destinations before writing any of them.
            destinations = [p.resolve() for p in [args.output, args.photometry_output, args.planning_output] if p]
            if len(set(destinations)) != len(destinations) or any(p.exists() for p in destinations):
                raise ValueError("Output paths must be distinct new files")
            result = run_cover(args.manifest)
            for destination, key in [(args.photometry_output, "photometry_manifest"), (args.planning_output, "planning_manifest")]:
                if destination:
                    with destination.open("x") as stream:
                        stream.write(json.dumps(result[key], indent=2, allow_nan=False) + "\n")
        elif args.command == "weigh_stamp":
            from .stamps import run_stamp
            result = run_stamp(args.manifest)
        else:
            result = run_photometry(args.manifest, joint=args.command == "dig_floor", alpha=getattr(args, "alpha", 0.01))
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if getattr(args, "output", None):
            with args.output.open("x") as stream:
                stream.write(text)
        else:
            sys.stdout.write(text)
    except (ValueError, OSError, KeyError, TypeError, IndexError, ImportError) as error:
        parser.exit(2, f"magictools: {error}\n")


if __name__ == "__main__":
    main()
