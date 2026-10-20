"""Agent-facing JSON CLI. Scientific engines remain unchanged."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from .agent_contract import TOOLS, schema


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
    if tool == "fitsky":
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
    if tool == "fitsky":
        from research.fit_sky.sequential import SequentialFit
        settings = {key: getattr(args, key) for key in FIT_OPTIONS if getattr(args, key) is not None}
        if args.resume:
            fit = SequentialFit.load(args.resume, value["scene"])
            if any(fit.settings[k] != v for k, v in settings.items()):
                raise InputError("Resume settings conflict with checkpoint; omit overrides or start a new fit")
            if "queries" in value:
                fit.model.events(fit.population, value["queries"])
                fit.queries = value["queries"]
        else:
            fit = SequentialFit(value["scene"], queries=value.get("queries"), **settings)
        stop = len(value["scene"]["exposures"]) if args.stop_after is None else args.stop_after
        if not fit.assimilated < stop <= len(value["scene"]["exposures"]):
            raise InputError("stop-after must append at least one available exposure")
        batch = args.exposure_batch or 1
        if batch < 1:
            raise InputError("exposure-batch must be positive")
        # A starting checkpoint also survives interruption during first assimilation.
        fit.save(directory/"checkpoint.npz")
        while fit.assimilated < stop:
            fit.assimilate(min(stop, fit.assimilated+batch))
            fit.save(directory/"checkpoint.npz")
            write(directory/"report.json", fit.report(include_catalogues=bool(args.catalogues)))
            print(json.dumps({"event": "exposure_complete", "summary": fit.summary()}, allow_nan=False), file=sys.stderr, flush=True)
        return fit.report(include_catalogues=bool(args.catalogues)), fit.summary()
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
    p = Parser(description="JSON contracts for Kepler and FitSky. Help is text; operational output is one JSON object.")
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("list", help="Discover tools and scientific limitations")
    subs.add_parser("doctor", help="Check optional dependencies without running science")
    s = subs.add_parser("schema", help="Return a tool's JSON Schema in result")
    s.add_argument("tool", choices=TOOLS)
    e = subs.add_parser("example", help="Create a runnable synthetic manifest and its resources")
    e.add_argument("tool", choices=TOOLS)
    e.add_argument("--output", required=True, type=Path)
    for name in ["validate", "run"]:
        s = subs.add_parser(name)
        s.add_argument("tool", choices=TOOLS)
        s.add_argument("manifest", type=Path)
        if name == "run":
            s.add_argument("--output", required=True, type=Path)
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
    envelope = {"schema_version": 1, "ok": False}
    owned = None
    executing = False
    try:
        args = parser().parse_args(argv)
        envelope.update(command=args.command)
        if hasattr(args, "tool"):
            envelope.update(tool=args.tool, scientific_scope=TOOLS[args.tool]["accuracy"], full_objective_complete=False)
        if args.command == "list":
            result = {"tools": TOOLS, "exit_codes": {"0": "execution completed; inspect scientific status", "2": "input or usage error",
                       "3": "missing dependency", "4": "execution failed/refused", "130": "interrupted"}}
        elif args.command == "doctor":
            result = {"tools": {name: dependencies(name) for name in TOOLS}, "install": "pip install '.[research,agent]'"}
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
                    if args.tool != "fitsky" and any(getattr(args, k) is not None for k in
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
