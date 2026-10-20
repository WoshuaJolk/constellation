"""Agent-facing command line: JSON in, receipts and durable artifacts out."""
import argparse
import json
from pathlib import Path
import sys

from .credentials import Credentials, initialize
from .engine import export_csv, merge_manifests, plan, run, status
from .inventory import coverage
from .model import LIMITATIONS, MOST_CATALOGS, PROVIDERS, validate
from .transport import ArchiveError, Transport


def capabilities():
    return {"version": 2, "providers": PROVIDERS, "mostCatalogs": MOST_CATALOGS,
            "modes": ["field", "object", "orbits", "arc", "ephemeris"],
            "defaultTimeoutSeconds": 600, "clientRowLimit": None, "dateSpanLimit": None,
            "orbitCountLimit": None, "credentials": {"init": "magictools frames credentials init", "status": "magictools frames credentials status"},
            "limitations": LIMITATIONS, "datasetCoverage": coverage(),
            "customServices": {"protocols": ["obscore", "sia1", "sia2"], "idPrefix": "vo:",
                               "configuration": "services object in the search manifest", "credentials": "Private credential profile with explicit HTTPS allowedOrigins"},
            "discoveryCommands": ["discover", "inspect", "links", "audit"],
            "example": {"mode": "field", "ra": 180, "dec": 30, "radius": 0.05, "start": "2019-01-28", "end": "2019-01-29", "providers": ["ztf"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="magictools frames", description="Resumable archive image discovery for agents")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities", help="Print adapters, schemas, limits, and credential instructions as JSON")
    commands.add_parser("coverage", help="Map every constellation imaging dataset to adapters and explicit gaps")
    audit = commands.add_parser("audit", help="Assess coverage evidence separately from query success")
    audit.add_argument("run_dir", type=Path)
    audit.add_argument("--checksums", action="store_true", help="Verify saved raw-response SHA-256 receipts")
    discover = commands.add_parser("discover", help="Find candidate SIA/TAP endpoints in the IVOA registry")
    discover.add_argument("--text", default="")
    discover.add_argument("--protocol", choices=["all", "obscore", "sia1", "sia2"], default="all")
    inspect = commands.add_parser("inspect", help="Inspect a built-in VO provider or a service JSON file")
    inspect.add_argument("service")
    links = commands.add_parser("links", help="Resolve a public DataLink URL to native product references; no image downloads")
    links.add_argument("url")
    for command in (discover, inspect, links):
        command.add_argument("--output-dir", type=Path, required=True, help="New directory for metadata and request receipts")
        command.add_argument("--timeout", type=float, default=600)
        command.add_argument("--credentials", type=Path)
    search = commands.add_parser("search", help="Execute a JSON manifest and save complete local query receipts")
    search.add_argument("manifest", type=Path)
    search.add_argument("--run-dir", type=Path, required=True)
    search.add_argument("--resume", action="store_true")
    search.add_argument("--retry-partial", action="store_true")
    search.add_argument("--timeout", type=float, help="Override per-request timeout, seconds (default 600)")
    search.add_argument("--credentials", type=Path)
    search.add_argument("--format", choices=["json", "ndjson", "csv"], default="json")
    search.add_argument("--plan", action="store_true", help="Validate and show requests without contacting any archive")
    check = commands.add_parser("status", help="Inspect a run from another agent/process, including pending queries")
    check.add_argument("run_dir", type=Path)
    merge = commands.add_parser("merge", help="Deduplicate field/orbit runs, preserving every product and provenance record")
    merge.add_argument("manifests", nargs="+", type=Path)
    merge.add_argument("-o", "--output", type=Path, required=True)
    credential = commands.add_parser("credentials")
    credential.add_argument("action", choices=["init", "status"])
    credential.add_argument("--config", type=Path)
    serve = commands.add_parser("serve", help="Optional local POST /api/frames jobs with poll URLs; no website")
    serve.add_argument("--directory", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--credentials", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            result = capabilities()
        elif args.command == "coverage":
            result = coverage()
        elif args.command == "audit":
            from .assessment import audit_run
            result = audit_run(args.run_dir, args.checksums)
            json.dump(result, sys.stdout, indent=2, allow_nan=False)
            sys.stdout.write("\n")
            return 4 if result.get("artifactIntegrity", {}).get("failures") else 0
        elif args.command in {"discover", "inspect", "links"}:
            from .model import finite
            from .profiles import VO_PROFILES
            from .vo import discover as discover_services, inspect_service, datalinks
            timeout = finite(args.timeout, "timeout", 1, 86400)
            args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            transport = Transport(args.output_dir, timeout=timeout, credentials=Credentials(args.credentials))
            try:
                if args.command == "discover":
                    result = discover_services(transport, args.text, args.protocol)
                elif args.command == "inspect":
                    config = VO_PROFILES.get(args.service) or json.loads(Path(args.service).read_text())
                    result = inspect_service(transport, config)
                else:
                    result = datalinks(transport, args.url)
                (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            finally:
                (args.output_dir / "requests.json").write_text(json.dumps(transport.calls, indent=2, allow_nan=False)+"\n")
        elif args.command == "credentials":
            result = initialize(args.config) if args.action == "init" else Credentials(args.config).status()
        elif args.command == "status":
            result = status(args.run_dir)
        elif args.command == "merge":
            if args.output.exists() or args.output.with_suffix(".csv").exists():
                raise ValueError("Output file exists; choose a new path")
            if args.output.suffix.lower() != ".json":
                raise ValueError("Merge output must use .json (a companion .csv is also written)")
            result = merge_manifests([json.loads(p.read_text()) for p in args.manifests])
            with args.output.open("x") as out:
                json.dump(result, out, indent=2, allow_nan=False)
            export_csv(result, args.output.with_suffix(".csv"))
        elif args.command == "serve":
            from .server import serve
            serve(args.directory, args.port, Credentials(args.credentials))
            return 0
        else:
            data = json.loads(args.manifest.read_text())
            if args.timeout is not None:
                data["timeoutSeconds"] = args.timeout
            x = validate(data, args.manifest.parent)
            if args.plan:
                from dataclasses import asdict
                jobs, unavailable = plan(x)
                result = {"input": x, "queries": [asdict(j) for j in jobs], "unqueried": unavailable}
            else:
                def emit(event):
                    stream = sys.stdout if args.format == "ndjson" else sys.stderr
                    stream.write(json.dumps(event, allow_nan=False)+"\n")
                    stream.flush()
                result = run(x, args.run_dir, resume=args.resume, credentials=Credentials(args.credentials),
                             emit=emit, retry_partial=args.retry_partial)
                if args.format == "csv":
                    with (args.run_dir / "frames.csv").open() as f:
                        for chunk in iter(lambda: f.read(65536), ""):
                            sys.stdout.write(chunk)
                elif args.format == "json":
                    json.dump(result, sys.stdout, indent=2, allow_nan=False)
                    sys.stdout.write("\n")
                return 0 if result["status"] == "complete" else 3
        json.dump(result, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
        return 0
    except ArchiveError as error:
        sys.stderr.write(json.dumps({"status": "archive-error", "error": str(error)})+"\n")
        return 3
    except (ValueError, OSError, TypeError, KeyError) as error:
        sys.stderr.write(json.dumps({"status": "invalid", "error": str(error)})+"\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write('{"status":"interrupted","message":"Completed queries are checkpointed; resume with the same input and run directory."}\n')
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
