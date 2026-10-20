"""Reproduce constructive cascade certificates and full-domain hierarchy checks."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from flint import ctx

from .cascade_domain import certify_domain
from .certified_cascade import (
    CertifiedCascadeTransfer, cadence_count, fourier_modulus_bound, sufficient_uniform_order,
)
from .certified_segment_field import CertifiedAtlasTransform
from .chart_experiments import setup
from .compiled_reference import projected
from .factorization import ARCSEC
from .reception_flow import CircularObserver, reception_cover
from .segment_experiments import geometry
from .segment_field import TensorGrid


def domain(case):
    old, oc, ow, geom, _ = setup(case)
    c, w = reception_cover(old, oc, ow)
    return old, oc, ow, geom, c, w


def certificate(case):
    _, _, _, geom, c, w = domain(case)
    observer = CircularObserver()
    proof = certify_domain(observer, c, w, 0., .125, complex_ellipse=1.2)
    modulus = fourier_modulus_bound(proof, observer, [1/ARCSEC, 1/ARCSEC])
    order = sufficient_uniform_order(1.2, modulus["log_modulus_per_source_l1"])
    return {"case": case, "original_stack_endpoints": [float(geom.times[0]), float(geom.times[-1])],
            "transfer_certificate": proof, "physical_frequency_modulus": modulus,
            "sufficient_single_cell_order_not_allocated": order,
            "is_complete_original_stack_transform": False}


def refinement():
    _, _, _, _, c, w = domain("near")
    observer, results = CircularObserver(), []
    for p in [3, 5]:
        parent = TensorGrid(c, w, (1,)*6, p)
        for refine in [False, True]:
            transfer = CertifiedCascadeTransfer(
                observer, parent, 0., .5, child_order=2, refine=refine)
            def polynomial(q):
                z = (q-transfer.child.center)/transfer.child.widths
                return z[:, 3]+.3*z[:, 4]+.2*z[:, 0]*z[:, 1]+.1*z[:, 2]*z[:, 5]
            source = polynomial(transfer.child.points())
            begin = perf_counter()
            actual = transfer @ source
            apply_seconds = perf_counter()-begin
            mapped = observer.chart(0.).rebase(parent.points(), .5, observer.chart(.5))
            expected = polynomial(mapped)
            results.append({
                "parent_order": p, "child_order": 2, "refined": refine,
                "parent_nodes": parent.size,
                "max_error_on_all_parent_nodes": float(np.max(abs(actual-expected))),
                "apply_seconds": apply_seconds, "transfer": transfer.report,
                "uniform_error_certified": False,
                "child_polynomial_exactly_represented": True,
            })
    return results


def hierarchy(case):
    old, oc, ow, original, c, w = domain(case)
    observer = CircularObserver()
    # Retain the original cadence, duration and state box, including 32 distant frames.
    times = original.times.copy()
    frames = len(times)
    rng = np.random.default_rng(20261031)
    directions = rng.uniform(-1, 1, (frames, 3, 2))
    bands = [.03, 1.]
    frequencies = np.concatenate([directions*b/ARCSEC for b in bands], axis=1)
    coefficients = rng.normal(size=(frames, 3))+1j*rng.normal(size=(frames, 3))
    parameters = oc+rng.uniform(-.999, .999, (128, 6))*ow
    states = old.states(parameters)
    queries = observer.chart(0.).parameters(states)
    geom = geometry(times, observer.axes)
    begin = perf_counter()
    projected(states[:1], geom)
    warmup = perf_counter()-begin
    transform = CertifiedAtlasTransform(
        observer, c, w, times, frequencies, order=3, exact_leaves=True,
        max_patches=32, max_nodes=500_000, max_entries=20_000_000,
        certificate_options={"complex_ellipse": 1.05, "max_iterations": 8})
    positions = projected(states, geom)
    results = []
    for index, band in enumerate(bands):
        data = np.zeros((frames, 6), complex)
        data[:, index*3:(index+1)*3] = coefficients
        begin = perf_counter()
        field = transform.transform(data.ravel())
        apply_seconds = perf_counter()-begin
        begin = perf_counter()
        actual = field.evaluate(queries)
        query_seconds = perf_counter()-begin
        reference = np.einsum("ntm,tm->n", np.exp(
            1j*np.einsum("nti,tmi->ntm", positions, directions*band/ARCSEC)), coefficients)
        results.append({
            "frequency_bound_inverse_arcsec": band,
            "sampled_error_per_source_l1": float(
                np.max(abs(actual-reference))/np.sum(abs(coefficients))),
            "apply_seconds": apply_seconds, "query_seconds": query_seconds,
            "queries": len(queries), "uniform_error_certified": False,
        })
    data = np.c_[coefficients, coefficients].ravel()
    field = transform.transform(data)
    y = rng.normal(size=len(queries))+1j*rng.normal(size=len(queries))
    lhs = np.vdot(y, field.evaluate(queries))
    rhs = np.vdot(field.evaluate_adjoint(queries, y), data)
    return {
        "case": case, "original_center": oc.tolist(), "original_widths": ow.tolist(),
        "original_times": times.tolist(), "frames": frames, "order": 3,
        "synthetic_fourier_data": True, "reference_compile_warmup_seconds": warmup,
        "bands": results, "query_adjoint_relative_discrepancy": float(
            abs(lhs-rhs)/max(1., abs(lhs), abs(rhs))),
        "graph": transform.stats, "full_objective_complete": False,
    }


def source_hashes():
    root = Path(__file__).parents[2]
    paths = sorted(Path(__file__).parent.glob("*.py"))+[
        root/"src/magictools/orbits.py",
        root/"tests/test_kepler_cascade_domain.py",
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(directory, mode):
    directory.mkdir(parents=True, exist_ok=False)
    ctx.prec = 100
    hashes = source_hashes()
    result = {"source_sha256": hashes, "arb_precision_bits": ctx.prec,
              "full_objective_complete": False}
    if mode in ["all", "local"]:
        result["certificates"] = [certificate(case) for case in ["near", "distant", "stress"]]
        result["refinement"] = refinement()
        rng = np.random.default_rng(5)
        result["cadence_count_model"] = []
        for n in [64, 1024, 16384]:
            for name, times in [
                ("regular", np.linspace(0, 1, n)),
                ("clustered", np.sort(np.r_[rng.uniform(0, 1e-6, n-2), [.5, 1.]])),
            ]:
                result["cadence_count_model"].append({"cadence": name, **cadence_count(times, 32)})
    if mode in ["all", "hierarchy"]:
        result["hierarchies"] = []
        for case in ["near", "distant", "stress"]:
            begin = perf_counter()
            try:
                value = hierarchy(case)
                result["hierarchies"].append({"status": "ran", "result": value})
                print(json.dumps({"case": case, "patches": value["graph"]["patches"],
                                  "seconds": perf_counter()-begin}), flush=True)
            except (ValueError, MemoryError) as error:
                result["hierarchies"].append({
                    "case": case, "status": "refused", "reason": str(error),
                    "seconds": perf_counter()-begin, "full_objective_complete": False})
                print(json.dumps(result["hierarchies"][-1]), flush=True)
    if hashes != source_hashes():
        raise RuntimeError("Research source changed during the campaign; refusing provenance")
    (directory/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mode", choices=["all", "local", "hierarchy"], default="all")
    args = parser.parse_args()
    run(args.directory, args.mode)


if __name__ == "__main__":
    main()
