"""Reproducible local-grid sizes and complete original-cadence cost plans."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from flint import ctx

from .cascade_domain import certify_domain
from .constructive_experiments import domain
from .factorization import ARCSEC
from .local_hierarchy import LocalAtlasPlan, LocalHierarchyPlan
from .local_resolution import local_grid
from .reception_flow import CircularObserver


def hashes():
    root = Path(__file__).parents[2]
    files = list(Path(__file__).parent.glob("*.py"))+[
        root/"src/magictools/orbits.py", root/"tests/test_kepler_local_resolution.py"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def local(case):
    *_, c, w = domain(case)
    observer = CircularObserver()
    proof = certify_domain(observer, c, w, 0., .125)
    plan = local_grid(proof, observer, [.125, .125], [1/ARCSEC]*2,
                      neighborhood_ellipses=(1.2, 2., 4., 8., 16., 32.))
    return {"case": case, "certificate": proof, "grid": plan}


def hierarchy(case, cache_directory=None):
    *_, geom, c, w = domain(case)
    def progress(record):
        print(json.dumps({"case": case, **record}), flush=True)
    atlas = LocalAtlasPlan(CircularObserver(), c, w, geom.times, [1/ARCSEC]*2,
                           max_patches=32, neighborhood_ellipses=(1.05, 2., 4., 8., 32.),
                           progress=progress, cache_directory=cache_directory)
    return {"case": case, "original_times": geom.times.tolist(), "atlas": atlas.record()}


def numerical():
    *_, c, w = domain("near")
    observer = CircularObserver()
    plan = LocalHierarchyPlan(observer, c, w, [-1e-7, 1e-7], [1/ARCSEC]*2,
                              tolerance=1e-8, neighborhood_ellipses=(2., 8., 32.))
    frequencies = [np.array([[1/ARCSEC, -1/ARCSEC]])]*2
    graph = plan.materialize(frequencies, max_nodes=2_000_000, max_entries=30_000_000)
    data = np.array([1+2j, -.3+.7j])
    begin = perf_counter()
    values = graph.transform(data)
    application_seconds = perf_counter()-begin
    rng = np.random.default_rng(134)
    queries = c+rng.uniform(-.999, .999, (64, 6))*w
    actual = graph.evaluate(values, queries)
    # Independent Cartesian projection, not the transfer's rebase method.
    from .compiled_reference import projected
    from .segment_experiments import geometry
    states = observer.chart(0.).states(queries)
    positions = projected(states, geometry(plan.times, observer.axes))
    exact = sum(np.exp(1j*positions[:,i] @ frequencies[i][0])*data[i] for i in range(2))
    y = rng.normal(size=len(values))+1j*rng.normal(size=len(values))
    adjoint = abs(np.vdot(y, values)-np.vdot(graph.adjoint(y), data))
    return {"plan": plan.record(), "materialized": graph.stats,
            "sampled_error_per_source_l1": float(max(abs(actual-exact))/sum(abs(data))),
            "query_count": len(queries), "adjoint_absolute_discrepancy": float(adjoint),
            "application_seconds": application_seconds,
            "not_an_archive_benchmark": True, "time_span_days": 2e-7,
            "numerical_regression_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mode", choices=["local", "hierarchy", "numerical"], required=True)
    parser.add_argument("--case", choices=["near", "distant", "stress"], default="near")
    parser.add_argument("--cache-directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    ctx.prec = 100
    source = hashes()
    begin = perf_counter()
    result = {"source_sha256": source, "arb_precision_bits": ctx.prec,
              "python": platform.python_version(), "full_objective_complete": False}
    result[args.mode] = (numerical() if args.mode == "numerical" else
                         local(args.case) if args.mode == "local" else hierarchy(args.case, args.cache_directory))
    result["total_seconds"] = perf_counter()-begin
    if hashes() != source:
        raise RuntimeError("Source changed during campaign; refusing provenance")
    (args.directory/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"saved": str(args.directory), "seconds": result["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
