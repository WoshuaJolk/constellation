"""One public FitSky tool with explicit, scientifically distinct engines."""
import argparse
from importlib import import_module


def main(argv=None, *, prog="constellation fit_sky"):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog=prog, add_help=False, allow_abbrev=False)
    parser.add_argument("--engine", choices=["sequential", "bounded"], default="sequential")
    args, remaining = parser.parse_known_args(arguments)
    if "--help" in remaining or "-h" in remaining:
        print("fit_sky: joint source inference. --engine {sequential,bounded} (default: sequential).")
        print("Sequential: approximate, resumable exposure assimilation. Bounded: small-scene research inference.")
        print("Each engine retains its own input model, options and accuracy limits.\n")
    module = "research.fit_sky.sequential_cli" if args.engine == "sequential" else "research.fit_sky.cli"
    return import_module(module).main(remaining, prog=prog) or 0
