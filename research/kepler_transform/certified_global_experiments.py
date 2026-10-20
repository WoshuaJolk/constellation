"""Execute orders chosen by whole-cell proofs; keep measured and proved errors separate."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from .chart_experiments import save, setup
from .factorization import ARCSEC
from .global_kepler import ContinuousKeplerPhase
from .source_butterfly import SourceButterfly


def run(proof_path, output, *, frames=None, random_queries=128):
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(output)
    proof = json.loads(proof_path.read_text())
    if not proof["all_hypotheses_certified"]:
        raise ValueError("An incomplete proof cannot select certified orders")
    case = proof["case"]
    chart, center, widths, geometry, _ = setup(case)
    # Bind the proof to the actual model/domain instead of reusing its label.
    if (chart.to_dict() != proof["chart"] or not np.array_equal(center, proof["root_center"])
            or not np.array_equal(widths, proof["root_widths"])
            or max(abs(geometry.times)) != proof["time_half_span_days"]
            or 1/ARCSEC != proof["frequency_scale"]):
        raise ValueError("Proof and executed domain differ")
    rng = np.random.default_rng(20261031)
    frames = len(geometry.times) if frames is None else frames
    if frames < 2 or random_queries < 0:
        raise ValueError("At least two exposure times and nonnegative random query count required")
    times = np.linspace(-1, 1, frames)
    sources = np.column_stack((np.repeat(times, 16), rng.uniform(-1, 1, (frames*16, 2))))
    data = rng.normal(size=len(sources))+1j*rng.normal(size=len(sources))
    data /= np.linalg.norm(data)
    phase = ContinuousKeplerPhase(chart, center, widths,
        time_half_span=proof["time_half_span_days"], frequency_scale=proof["frequency_scale"])
    before = perf_counter()
    plan = SourceButterfly(phase, sources, output_dimension=6, depth=proof["depth"],
                           order=tuple(proof["sufficient_orders"]),
                           max_gate_bytes=8_000_000_000,
                           moment_backend="dense" if frames <= 32 else "nufft")
    construction = perf_counter()-before
    print("Constructed", case, "orders", plan.orders, "seconds", construction, flush=True)
    before = perf_counter()
    field = plan.transform(data)
    application = perf_counter()-before
    print("Applied", case, "seconds", application, flush=True)
    construction_geometry = phase.geometry_pairs
    boundaries = rng.uniform(-.9, .9, (6, 6))
    np.fill_diagonal(boundaries, 0)
    queries = np.concatenate((np.array(list(product([-1., 1.], repeat=6))),
        rng.uniform(-1, 1, (random_queries, 6)), boundaries, np.zeros((1, 6))))
    before = perf_counter()
    actual = field.evaluate(queries, accelerated=True)
    evaluation = perf_counter()-before
    total_geometry = phase.geometry_pairs
    # Independent Python universal propagation at all small-run queries, or a
    # deterministic boundary/random subset of large runs.
    checked = np.unique(np.r_[np.arange(64), np.linspace(64, len(queries)-1, 135, dtype=int)])
    before = perf_counter()
    reference = np.exp(1j*phase.independent(queries[checked], sources)) @ data
    independent_seconds = perf_counter()-before
    error = actual[checked]-reference
    # Compiled direct baseline includes geometry and all 16 Fourier modes at
    # every supplied exposure. Chunking bounds memory; no Python orbit loop.
    before = perf_counter()
    direct = np.empty(len(queries), complex)
    for start in range(0, len(queries), 256):
        selected = queries[start:start+256]
        direct[start:start+len(selected)] = np.exp(1j*phase(selected, sources)) @ data
    direct_seconds = perf_counter()-before
    minus, plus = boundaries.copy(), boundaries.copy()
    np.fill_diagonal(minus, -1e-10)
    np.fill_diagonal(plus, 1e-10)
    jumps = field.evaluate(plus, accelerated=True)-field.evaluate(minus, accelerated=True)
    expected = (np.exp(1j*phase.independent(plus, sources))
                -np.exp(1j*phase.independent(minus, sources))) @ data
    from magictools.orbits import GM
    states = chart.states(center+queries[checked]*widths)
    energy = np.sum(states[:, 3:]**2, axis=1)/2-GM["sun"]/np.linalg.norm(states[:, :3], axis=1)
    proof_seconds = sum(json.loads(p.read_text())["certificate_construction_seconds"]
                        for p in proof_path.parent.glob(f"{case}-rho*.json"))
    result = {
        "case": case, "proof_file": proof_path.name,
        "proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        "orders": plan.orders, "depth": plan.depth, "plan": plan.report,
        "frames": frames, "sources": len(sources), "queries": len(queries),
        "independently_checked_queries": len(checked),
        "construction_seconds": construction, "application_seconds": application,
        "evaluation_seconds": evaluation, "total_fast_seconds": construction+application+evaluation,
        "compiled_direct_seconds": direct_seconds,
        "direct_over_total_fast": direct_seconds/(construction+application+evaluation),
        "proof_search_seconds_all_saved_ellipses": proof_seconds,
        "total_including_proof_search_seconds": construction+application+evaluation+proof_seconds,
        "direct_over_total_including_proof": direct_seconds/(construction+application+evaluation+proof_seconds),
        "independent_validation_seconds": independent_seconds,
        "construction_and_application_geometry_pairs": construction_geometry,
        "total_fast_geometry_pairs": total_geometry, "direct_geometry_pairs": len(queries)*frames,
        "maximum_independent_score_error": float(max(abs(error))),
        "relative_independent_score_error": float(np.linalg.norm(error)/np.linalg.norm(reference)),
        "maximum_compiled_direct_score_error": float(max(abs(actual-direct))),
        "sampled_l1_normalized_error": float(max(abs(error))/sum(abs(data))),
        "maximum_excess_boundary_jump": float(max(abs(jumps-expected))),
        "exact_arithmetic_kernel_error_bound": proof["exact_arithmetic_kernel_error_bound"],
        "exact_arithmetic_score_error_bound": proof["exact_arithmetic_kernel_error_bound"]*float(sum(abs(data))),
        "peak_process_rss_bytes": 1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sampled_bound_states": int(sum(energy < 0)), "sampled_unbound_states": int(sum(energy > 0)),
        "all_six_coordinates_free": True, "entire_domain_constructed_before_queries": True,
        "physical_orbit_domain_shrunk": False, "physical_frequency_scale": 1/ARCSEC,
        "input_kind": "synthetic complex Fourier response coefficients, 16 modes per exposure",
        "fast_floating_point_output_certified": False, "full_objective_complete": False,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    np.savez_compressed(output.with_suffix(".npz"), sources=sources, data=data, queries=queries,
                        actual=actual, direct=direct, checked_indices=checked,
                        independent_reference=reference, excess_boundary_jumps=jumps-expected)
    save(output, result)
    print(json.dumps({k: result[k] for k in ["case", "orders", "total_fast_seconds",
        "compiled_direct_seconds", "relative_independent_score_error", "sampled_l1_normalized_error"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proof", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--random-queries", type=int, default=128)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.proof, args.output, frames=args.frames, random_queries=args.random_queries)


if __name__ == "__main__":
    main()
