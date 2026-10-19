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
    for name in ["force_track", "dig_floor", "learn_null", "map_cover", "weigh_stamp", "find_ejector", "ejector_prepare", "ejector_fit", "prove_floor", "ztf_search", "ztf_fetch", "judge_line", "inject_tracks", "map_complete", "inject_all", "match_all", "verify_match", "fold_kepler"]:
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
    fold_demo = commands.add_parser("fold_demo", help="Benchmark a shared-subpath search on synthetic Kepler orbits")
    fold_demo.add_argument("directory", type=Path)
    fold_demo.add_argument("--seed", type=int, default=2026)
    fold_demo.add_argument("--grid-size", type=int, default=24)
    fold_demo.add_argument("--frames", type=int, default=32)
    for name in ["floor_demo", "ejector_demo"]:
        command = commands.add_parser(name, help="Generate and run a reproducible research experiment")
        command.add_argument("directory", type=Path)
        command.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        if getattr(args, "output", None) and args.output.exists():
            raise ValueError("Output file already exists")
        if args.command == "fold_kepler":
            from .fold_kepler import run_fold
            result = run_fold(args.manifest)
        elif args.command == "fold_demo":
            from .fold_demo import create_fold_demo
            result = create_fold_demo(args.directory, args.seed, args.grid_size, args.frames)
        elif args.command in {"match_all", "verify_match"}:
            from .matching import run_match, run_verify_match
            result = (run_match if args.command == "match_all" else run_verify_match)(args.manifest)
        elif args.command in {"map_complete", "inject_all"}:
            from .selection import run_selection
            result = run_selection(args.manifest, monte_carlo=args.command == "inject_all")
        elif args.command == "judge_line":
            from .vetting import run_judge
            result = run_judge(args.manifest)
        elif args.command == "inject_tracks":
            from .injections import run_injections
            result = run_injections(args.manifest)
        elif args.command in {"ztf_search", "ztf_fetch"}:
            from .ztf import run_ztf_search, run_ztf_fetch
            result = (run_ztf_search if args.command == "ztf_search" else run_ztf_fetch)(args.manifest)
        elif args.command in {"floor_demo", "ejector_demo"}:
            from .research_demo import create_floor_demo, create_ejector_demo
            result = (create_floor_demo if args.command == "floor_demo" else create_ejector_demo)(args.directory, args.seed)
        elif args.command == "ejector_fit":
            from .ejector_fit import run_fit
            result = run_fit(args.manifest)
        elif args.command == "ejector_prepare":
            from .ejector_gaia import run_prepare
            result = run_prepare(args.manifest)
        elif args.command == "find_ejector":
            from .ejector import run_ejector
            result = run_ejector(args.manifest)
        elif args.command == "prove_floor":
            from .prove_floor import run_floor
            result = run_floor(args.manifest)
        elif args.command == "orbit_demo":
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
