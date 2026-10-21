"""Agent-facing JSON CLI. Scientific engines remain unchanged."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
from importlib import import_module
import json
import math
from pathlib import Path
import sys

from .agent_contract import TOOLS, TOOL_ALIASES, schema
from .cli import MANIFEST_COMMANDS, DEMO_COMMANDS
from . import __version__
from .release import READY, EXPERIMENTAL, catalogue as release_catalogue


ARCHIVE_COMMANDS = {
    "capabilities": "Describe archive providers, query modes and credentials",
    "coverage": "Map imaging datasets to archive adapters and gaps",
    "find_frames": "Find candidate surveys and frames using a JSON manifest",
    "status": "Inspect an archive search run",
    "check_frames": "Check archive coverage evidence and receipt integrity",
    "discover": "Discover candidate archive services in the VO registry",
    "links": "Resolve public archive DataLink product references",
    "credentials": "Manage private archive credential profiles",
    "merge": "Deduplicate archive frame manifests",
    "serve": "Run the optional local archive search API",
}
ARCHIVE_ALIASES = {"search": "find_frames", "audit": "check_frames"}
RESEARCH_ALIASES = {
    "fit-sky": "research.fit_sky.cli",
    "fit-sky-stream": "research.fit_sky.sequential_cli",
}
RESEARCH_COMMANDS = {
    "fold-chart": "research.kepler_transform.chart_search",
    "fit_sky": "magictools.fit_sky_cli",
}


class InputError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise InputError(message)


def output(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def write(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise InputError(f"Nonfinite JSON number: {value}")
    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            return constant(value)
        return number
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant, parse_float=finite)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dependencies(tool):
    required = ["jsonschema"]+TOOLS[tool]["dependencies"]
    return {name: importlib.util.find_spec(name) is not None for name in required}


def require(tool):
    missing = [name for name, available in dependencies(tool).items() if not available]
    if missing:
        raise ImportError("Missing modules: "+", ".join(missing)+". Install this checkout with pip install '.[research,agent]'.")


def admission(tool, value, base):
    """Schema + model/resource admission. Never claim a global orbit proof."""
    from jsonschema import Draft202012Validator
    errors = sorted(Draft202012Validator(schema(tool)).iter_errors(value), key=lambda e: str(list(e.path)))
    if errors:
        e = errors[0]
        raise InputError("/"+"/".join(map(str, e.path))+": "+e.message)
    value = deepcopy(value)
    resources = []
    def resource(name):
        p = (base/name).resolve()
        if not p.is_file():
            raise InputError(f"Input resource does not exist: {p}")
        resources.append(p)
        return str(p)
    def unique(rows):
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise InputError("IDs must be unique within each input list")
    if tool == "fit_sky":
        from research.fit_sky.sequential import BlockModel
        import numpy as np
        model = BlockModel(value["scene"])
        model.events(model.prior(2, np.random.default_rng(0)), value.get("queries", []))
        scope = "Schema and model construction; no assimilation or global orbital-domain/accuracy proof"
    elif tool == "kepler-propagate":
        import numpy as np
        unique(value["states"])
        if "observers_heliocentric_au" in value and len(value["times_days"]) != len(value["observers_heliocentric_au"]):
            raise InputError("Exactly one reception observer per time required")
        if np.any(np.linalg.norm(np.array([r["state"][:3] for r in value["states"]]), axis=1) < 1e-5):
            raise InputError("Initial heliocentric radius must be at least 1e-5 AU")
        scope = "Schema and initial geometry; propagation convergence checked during run"
    elif tool == "kepler-fold":
        from astropy.io import fits
        from astropy.wcs import WCS
        unique(value["exposures"])
        rows = value.get("orbits", value.get("trajectories"))
        unique(rows)
        ntime = len(value["exposures"])
        if len(rows) > 100000 or ntime > 256 or len(rows)*ntime > 4_000_000:
            raise InputError("Fold bank exceeds engine limits: 100000 tracks, 256 exposures or 4 million pairs")
        if value.get("top_k", min(10, len(rows))) > len(rows):
            raise InputError("top_k exceeds number of supplied candidates")
        if "trajectories" in value and any(len(r["xy"]) != ntime for r in rows):
            raise InputError("Each trajectory needs one xy pair per exposure")
        seen = set()
        for e in value["exposures"]:
            for key in ["image", "variance", "mask", "psf_image"]:
                if isinstance(e.get(key), str):
                    e[key] = resource(e[key])
            with fits.open(e["image"], memmap=True) as hdus:
                index = hdus.index_of(e.get("image_hdu", 0))
                if (e["image"], index) in seen:
                    raise InputError("Duplicate image HDU")
                seen.add((e["image"], index))
                if hdus[index].header.get("NAXIS") != 2:
                    raise InputError("Science image must be two-dimensional")
                if "orbits" in value and not WCS(hdus[index].header).has_celestial:
                    raise InputError("Orbital fold exposures require celestial WCS")
        scope = "Schema, bank dimensions, resource existence and science FITS headers; pixel/noise/PSF checks occur during run"
    else:
        import numpy as np
        from research.kepler_transform.observer_chart import ObserverChart
        value["arrays"] = resource(value["arrays"])
        chart = ObserverChart(**value["chart"])
        center, widths = np.asarray(value["center"]), np.asarray(value["widths"])
        if center[0]-widths[0] <= 0:
            raise InputError("Chart box must have positive inverse distance")
        chart.states(center[None])
        with np.load(value["arrays"], allow_pickle=False) as a:
            required = {"times", "observers", "visits", "frequencies", "data", "parameters"}
            if not required <= set(a.files):
                raise InputError("Missing NPZ arrays: "+", ".join(sorted(required-set(a.files))))
            t, v, q = a["times"], a["visits"], a["parameters"]
            if (t.ndim != 1 or not len(t) or a["observers"].shape != (len(t), 3)
                    or v.ndim != 1 or not len(v) or not np.issubdtype(v.dtype, np.integer)
                    or np.any(v < 0) or np.any(v >= len(t))
                    or a["frequencies"].shape != (len(v), 2) or a["data"].shape != (len(v),)
                    or q.ndim != 2 or q.shape[1] != 6 or not len(q)
                    or np.any(np.abs(q-center) > widths)):
                raise InputError("Incompatible chart arrays or query outside declared domain")
            for key in required:
                if not np.all(np.isfinite(a[key])):
                    raise InputError(f"Nonfinite NPZ array: {key}")
                if key != "data" and not np.isrealobj(a[key]):
                    raise InputError(f"Real-valued geometry array required: {key}")
            axes = [k for k in ["normal", "east", "north"] if k in a.files]
            if axes:
                if len(axes) != 3 or any(a[k].shape != (len(t), 3) or not np.isrealobj(a[k]) for k in axes):
                    raise InputError("Supply all three per-exposure tangent axes")
                matrices = np.stack([a[k] for k in axes], axis=1)
                if not np.allclose(matrices@matrices.transpose(0, 2, 1), np.eye(3), atol=2e-12, rtol=0):
                    raise InputError("Invalid per-exposure tangent axes")
        scope = "Schema, resources, array dimensions and domain membership; chart certificates constructed during run"
    return value, sorted(set(resources)), scope


def execute(tool, value, directory, args):
    if tool == "fit_sky":
        from research.fit_sky.workflow import prepare, advance
        settings = {key: getattr(args, key) for key in FIT_OPTIONS if getattr(args, key) is not None}
        try:
            fit, stop = prepare(value, settings, resume=args.resume, stop_after=args.stop_after,
                                exposure_batch=args.exposure_batch or 1)
        except ValueError as error:
            raise InputError(str(error)) from error
        def progress(summary):
            print(json.dumps({"event": "exposure_complete", "summary": summary}, allow_nan=False),
                  file=sys.stderr, flush=True)
        return advance(fit, directory, stop=stop, exposure_batch=args.exposure_batch or 1,
                       catalogues=bool(args.catalogues), on_progress=progress)
    if tool == "kepler-fold":
        from .fold_kepler import run_fold
        report = run_fold(directory/"input.json")
        summary = {"trials": report["trials"], "detection_hypotheses": len(report["detections"]),
                   "top": report["top"][:5], "screening": report["screening"]}
    elif tool == "kepler-chart":
        from research.kepler_transform.chart_search import run
        report = run(directory/"input.json", directory/"engine")
        summary = {k: report[k] for k in ["status", "all_geometry_cells_resolved", "full_score_error_certified"]}
        summary["application"] = report.get("application")
        if report["status"] != "evaluated_all_queries":
            write(directory/"report.json", report)
            raise RuntimeError(report.get("reason", "Chart evaluation refused"))
    else:
        import numpy as np
        from research.kepler_transform.universal import propagate, apparent
        from .orbits import GM, C_AU_DAY
        states = np.array([row["state"] for row in value["states"]])
        times = np.array(value["times_days"])
        if "observers_heliocentric_au" in value:
            observers = np.array(value["observers_heliocentric_au"])
            result, emitted = apparent(states, times, observers)
            delta = result[..., :3]-observers[None]
            ra = np.degrees(np.arctan2(delta[..., 1], delta[..., 0])) % 360
            dec = np.degrees(np.arctan2(delta[..., 2], np.hypot(delta[..., 0], delta[..., 1])))
            directions = np.stack([ra, dec], axis=-1).tolist()
        else:
            result, emitted, directions = propagate(states, times), np.broadcast_to(times, (len(states), len(times))), None
        report = {"status": "propagated", "ids": [r["id"] for r in value["states"]],
                  "times_days": times.tolist(), "emission_times_days": emitted.tolist(),
                  "states": result.tolist(), "ra_dec_deg": directions,
                  "units": {"position": "AU", "velocity": "AU/day", "time": "TDB days from input epoch"},
                  "constants": {"mu_au3_day2": GM["sun"], "c_au_day": C_AU_DAY},
                  "full_objective_complete": False}
        summary = {"status": "propagated", "orbits": len(states), "times": len(times), "light_time": directions is not None}
    write(directory/"report.json", report)
    return report, summary


FIT_OPTIONS = ("particles", "sweeps", "seed", "pixel_block", "max_stages", "gibbs_flux", "split_merge")


def parser():
    p = Parser(prog="constellation", description="Archive discovery, Kepler and FitSky for agents. See each command's help for its output contract.")
    p.add_argument("--version", action="version", version=__version__)
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("frames", help="All archive commands, including archive schema inspection")
    for name, help_text in ARCHIVE_COMMANDS.items():
        subs.add_parser(name, help=help_text)
    for name in MANIFEST_COMMANDS:
        subs.add_parser(name, help=("Experimental/specialist: " if name in EXPERIMENTAL else "") + "Run the " + name + " manifest tool")
    for name in DEMO_COMMANDS:
        subs.add_parser(name, help="Generate and run the " + name + " experiment")
    subs.add_parser("kepler-propagate", help="Propagate supplied orbits; shortcut for run kepler-propagate")
    for name in RESEARCH_COMMANDS:
        subs.add_parser(name, help="Experimental: run the " + name + " research interface")
    listing = subs.add_parser("list", help="Discover tools, release status, and scientific limitations")
    listing.add_argument("--ready", action="store_true", help="Show only tools with release acceptance evidence")
    doctor = subs.add_parser("doctor", help="Check dependencies for release-ready tools")
    doctor.add_argument("--all", action="store_true", help="Also inspect optional research dependencies")
    s = subs.add_parser("schema", help="Return a tool's JSON Schema in result")
    s.add_argument("tool", choices=TOOLS, type=lambda name: TOOL_ALIASES.get(name, name))
    e = subs.add_parser("example", help="Create a runnable synthetic manifest and its resources")
    e.add_argument("tool", choices=TOOLS, type=lambda name: TOOL_ALIASES.get(name, name))
    e.add_argument("--output", required=True, type=Path)
    for name in ["validate", "run"]:
        s = subs.add_parser(name)
        s.add_argument("tool", choices=TOOLS, type=lambda name: TOOL_ALIASES.get(name, name))
        s.add_argument("manifest", type=Path)
        if name == "run":
            s.add_argument("-o", "--output", type=Path, help="New run directory (default: unique directory under runs/)")
            s.add_argument("--resume", type=Path, help="FitSky checkpoint, including v1 non-orbital checkpoints")
            for key in ["particles", "sweeps", "seed", "pixel_block", "max_stages", "stop_after", "exposure_batch"]:
                minimum = 2 if key == "particles" else 0 if key == "seed" else 1
                def integer(value, minimum=minimum):
                    number = int(value)
                    if number < minimum:
                        raise argparse.ArgumentTypeError(f"Must be at least {minimum}")
                    return number
                s.add_argument("--"+key.replace("_", "-"), type=integer)
            for key in ["gibbs_flux", "split_merge", "catalogues"]:
                s.add_argument("--"+key.replace("_", "-"), action="store_true", default=None)
    s = subs.add_parser("inspect", help="Read a run directory or NPZ checkpoint without resuming")
    s.add_argument("path", type=Path)
    return p


def artifacts(directory):
    return {str(p.relative_to(directory)): str(p.resolve()) for p in sorted(directory.rglob("*")) if p.is_file() and not p.name.endswith(".tmp")}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    envelope = {"schema_version": 1, "ok": False}
    owned = None
    executing = False
    try:
        if argv and argv[0] == "kepler-propagate":
            argv.insert(0, "run")
        if argv and argv[0] in ARCHIVE_ALIASES:
            argv[0] = ARCHIVE_ALIASES[argv[0]]
        if argv and (argv[0] == "frames" or argv[0] in ARCHIVE_COMMANDS):
            from .frames.cli import main as frames_main
            if argv[0] == "frames":
                return frames_main(argv[1:], prog="constellation frames")
            return frames_main(argv)
        if argv and argv[0] in (*MANIFEST_COMMANDS, *DEMO_COMMANDS):
            from .cli import main as manifest_main
            return manifest_main(argv, prog="constellation") or 0
        if argv and argv[0] in (RESEARCH_COMMANDS.keys() | RESEARCH_ALIASES.keys()):
            envelope.update(command=argv[0])
            entry = import_module((RESEARCH_COMMANDS | RESEARCH_ALIASES)[argv[0]]).main
            return entry(argv[1:], prog="constellation " + argv[0]) or 0
        args = parser().parse_args(argv)
        if args.command == "run" and args.output is None:
            from .run_paths import new_run
            args.output = new_run(args.tool, args.manifest)
        envelope.update(command=args.command)
        if hasattr(args, "tool"):
            envelope.update(tool=args.tool, scientific_scope=TOOLS[args.tool]["accuracy"], full_objective_complete=False)
        if args.command == "list":
            result = {"release": release_catalogue(), "tools": TOOLS, "aliases": {
                "archive": ARCHIVE_ALIASES, "science": TOOL_ALIASES,
                "research": {"fit-sky": "fit_sky --engine bounded",
                             "fit-sky-stream": "fit_sky --engine sequential"},
                "interfaces": {"run kepler-fold": "fold_kepler", "run kepler-chart": "fold-chart",
                               "run fit_sky": "fit_sky --engine sequential (agent JSON contract)"},
            }, "commands": {
                "archive": dict(ARCHIVE_COMMANDS, **{"frames inspect": "Inspect an archive service schema"}),
                "manifests": list(MANIFEST_COMMANDS), "demos": list(DEMO_COMMANDS),
                "direct": {"kepler-propagate": "run kepler-propagate"},
                "research": list(RESEARCH_COMMANDS),
                "science": ["list", "doctor", "schema", "example", "validate", "run", "inspect"],
                "help": "constellation <command> --help; each tool retains its output and exit-code contract",
            }, "exit_codes": {"0": "execution completed; inspect scientific status", "2": "input or usage error",
                       "3": "missing dependency", "4": "execution failed/refused", "130": "interrupted"}}
            if args.ready:
                result = {"tools": READY, "evidence": "docs/release-readiness.md",
                          "meaning": release_catalogue()["meaning"]}
        elif args.command == "doctor":
            core = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "astropy", "jsonschema")}
            checked = {name: dict(core) for name in READY}
            if args.all:
                checked.update({name: dependencies(name) for name in TOOLS})
            result = {"ready": all(core.values()), "tools": checked,
                      "scope": "Installed dependencies; no archive-uptime or scientific-validation check",
                      "install": "pip install ." if not args.all else "pip install '.[research]'"}
        elif args.command == "schema":
            result = schema(args.tool)
        elif args.command == "inspect":
            if args.path.is_dir():
                result = read(args.path/"run.json")
            else:
                import numpy as np
                with np.load(args.path, allow_pickle=False) as saved:
                    m = json.loads(str(saved["metadata"]))
                    if m.get("schema") not in ("fit-sky-sequential-v1", "fit-sky-sequential-v2"):
                        raise InputError("Unsupported checkpoint schema")
                    result = {k: m[k] for k in ["schema", "assimilated", "log_evidence", "settings", "queries"]}
                    result.update(particles=int(saved["counts"].size), inspection="Metadata only; use resume for full compatibility checks")
        else:
            require(args.tool)
            if hasattr(args, "output") and args.output.exists():
                raise InputError("Output directory already exists; choose a new directory")
            if args.command == "example":
                from .agent_examples import create
                args.output.mkdir(parents=True, exist_ok=False)
                owned = args.output.resolve()
                value = create(args.tool, owned)
                write(owned/"input.json", value)
                admission(args.tool, value, owned)
                result = {"manifest": str(owned/"input.json"), "synthetic": True}
            else:
                raw = read(args.manifest)
                value, resources, scope = admission(args.tool, raw, args.manifest.resolve().parent)
                if args.command == "validate":
                    result = {"valid": True, "validation_scope": scope, "resources": list(map(str, resources))}
                else:
                    if args.tool != "fit_sky" and any(getattr(args, k) is not None for k in
                            (*FIT_OPTIONS, "resume", "stop_after", "exposure_batch", "catalogues")):
                        raise InputError("FitSky execution options cannot be used with a Kepler tool")
                    args.output.mkdir(parents=True, exist_ok=False)
                    owned = args.output.resolve()
                    write(owned/"input.json", value)
                    write(owned/"provenance.json", {"manifest": str(args.manifest.resolve()),
                          "manifest_sha256": digest(args.manifest), "resolved_input_sha256": digest(owned/"input.json"),
                          "resources": {str(p): digest(p) for p in resources},
                          "resume_sha256": digest(args.resume) if args.resume else None})
                    write(owned/"run.json", dict(envelope, status="running", output=str(owned)))
                    executing = True
                    with redirect_stdout(sys.stderr):
                        report, result = execute(args.tool, value, owned, args)
                    envelope["scientific_status"] = report.get("status", TOOLS[args.tool]["accuracy"])
        envelope.update(ok=True, result=result)
        if owned:
            envelope.update(output=str(owned), artifacts=artifacts(owned))
            envelope["artifacts"]["run.json"] = str(owned/"run.json")
            write(owned/"run.json", envelope)
        output(envelope)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            code, kind = 130, "interrupted"
        elif isinstance(exc, ImportError):
            code, kind = 3, "missing_dependency"
        elif isinstance(exc, InputError) or (not executing and isinstance(exc, (ValueError, KeyError, TypeError, IndexError, OSError))):
            code, kind = 2, "invalid_input"
        else:
            code, kind = 4, "execution_failed"
        envelope.update(ok=False, error={"code": kind, "message": str(exc) or kind, "exception": type(exc).__name__})
        if owned:
            envelope.update(output=str(owned), artifacts=artifacts(owned), resume_available=(owned/"checkpoint.npz").is_file())
            try:
                write(owned/"failure.json", envelope)
                write(owned/"run.json", envelope)
            except OSError:
                pass
        output(envelope)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
