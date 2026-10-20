"""Challenge the global construction at unchanged physical orbit domains.

The measurements are falsification checks of a mathematical route, not a
replacement for proving its quantitative Kepler hypotheses.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .chart_experiments import save, setup
from .factorization import ARCSEC
from .global_kepler import ContinuousKeplerPhase
from .source_butterfly import SourceButterfly, theoretical_counts, conditional_error_bound
from .sundman import regularized
from .universal import propagate


def trial(case, fraction, depth, order, output, *, budget):
    chart, center, widths, geometry, _ = setup(case)
    rng = np.random.default_rng(20261030)
    times = np.linspace(-1, 1, len(geometry.times))
    sources = np.column_stack((np.repeat(times, 16), rng.uniform(-1, 1, (len(times)*16, 2))))
    data = rng.normal(size=len(sources))+1j*rng.normal(size=len(sources))
    data /= np.linalg.norm(data)
    phase = ContinuousKeplerPhase(chart, center, widths,
        time_half_span=max(abs(geometry.times)), frequency_scale=fraction/ARCSEC)
    before = perf_counter()
    result = {"case": case, "frequency_fraction": fraction, "depth": depth, "order": order,
              "source_samples": len(sources), "independent_exposure_times": len(times),
              "physical_bandwidth": fraction == 1, "all_six_coordinates_free": True,
              "full_objective_complete": False, "uniform_error_certified": False}
    try:
        plan = SourceButterfly(phase, sources, output_dimension=6, depth=depth, order=order,
                               max_gate_bytes=budget, max_pairs=200_000)
    except ValueError as error:
        result.update(status="construction_refused", reason=str(error), seconds=perf_counter()-before,
                      construction_geometry_pairs=phase.geometry_pairs)
        save(output, result)
        return result
    result.update(status="constructed_entire_domain", plan=plan.report,
                  construction_geometry_pairs=phase.geometry_pairs)
    start = perf_counter()
    field = plan.transform(data)
    result["application_seconds"] = perf_counter()-start
    # Every output coordinate is selected after the field has been constructed.
    corners = np.array(list(__import__("itertools").product([-1., 1.], repeat=6)))
    random = rng.uniform(-1, 1, (128, 6))
    boundaries = rng.uniform(-.9, .9, (6, 6))
    for i in range(6):
        boundaries[i, i] = 0.
    queries = np.concatenate((corners, random, boundaries, np.zeros((1, 6))))
    start = perf_counter()
    actual = field.evaluate(queries)
    result["evaluation_seconds"] = perf_counter()-start
    start = perf_counter()
    independent_phase = phase.independent(queries, sources)
    reference = np.exp(1j*independent_phase) @ data
    result["independent_validation_seconds"] = perf_counter()-start
    error = abs(actual-reference)
    result.update(queries=len(queries), maximum_score_error=float(max(error)),
        relative_score_error=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
        sampled_l1_normalized_error=float(max(error)/sum(abs(data))),
        sampled_relative_accuracy_target_met=bool(np.linalg.norm(actual-reference)/np.linalg.norm(reference) <= 1e-8),
        sampled_accuracy_is_not_a_uniform_bound=True)
    # An algebraic adjoint check uses the whole six-dimensional finite query set.
    values = rng.normal(size=len(queries))+1j*rng.normal(size=len(queries))
    adjoint = plan.adjoint(queries, values)
    result["adjoint_relative_defect"] = float(abs(np.vdot(actual, values)-np.vdot(data, adjoint))/
        max(1., abs(np.vdot(actual, values)), abs(np.vdot(data, adjoint))))
    minus, plus = boundaries.copy(), boundaries.copy()
    for i in range(6):
        minus[i, i], plus[i, i] = -1e-10, 1e-10
    jumps = field.evaluate(plus)-field.evaluate(minus)
    expected = (np.exp(1j*phase.independent(plus, sources))-np.exp(1j*phase.independent(minus, sources))) @ data
    result["maximum_excess_jump_across_leaf_boundary"] = float(max(abs(jumps-expected)))
    # Inspect all conic signs without filtering them out of the construction.
    from magictools.orbits import GM
    states = chart.states(center+queries*widths)
    energies = np.sum(states[:, 3:]**2, axis=1)/2-GM["sun"]/np.linalg.norm(states[:, :3], axis=1)
    result["sampled_bound_states"], result["sampled_unbound_states"] = int(sum(energies < 0)), int(sum(energies > 0))
    np.savez_compressed(output.with_suffix(".npz"), sources=sources, data=data, queries=queries,
                        approximate=actual, independent_reference=reference,
                        center=center, widths=widths, excess_boundary_jumps=jumps-expected)
    save(output, result)
    return result


def identities(output):
    from magictools.orbits import GM
    mu = GM["sun"]
    states = np.array([[r, 0, 0, radial, scale*np.sqrt(2*mu/r), .0002]
                       for r in [1.5, 3., 20.] for radial in [-.001, .002]
                       for scale in [.7, 1., 1.4]])
    transformed, times, info = regularized(states, np.linspace(-1, 1, 31))
    independent = propagate(states, times)
    counts = [theoretical_counts(6, 3, depth, 8) for depth in range(1, 9)]
    result = {
        "sundman": {"states": len(states), "sample_times": times.shape[1],
                    "maximum_position_difference_au": float(np.max(abs(transformed[..., :3]-independent[..., :3]))),
                    "maximum_velocity_difference_au_day": float(np.max(abs(transformed[..., 3:]-independent[..., 3:]))),
                    "physical_time_spread_at_s_one_days": float(np.ptp(times[:, -1])),
                    "initial_clock_derivatives": info["initial_clock_derivatives"].tolist(),
                    "shared_clock_claim_refuted": True},
        "full_tree_counts": counts,
        "fixed_correction_degree_lower_bounds": [
            {"phase_radius": radius, "minimum_polynomial_degree_for_error_less_than_one": 2*int(np.floor(radius/np.pi))}
            for radius in [1., 10., 100., 1000.]],
        "conditional_error_bounds": [
            {"depth": depth, "order": order, "rho": 2., "D": 1.,
             **conditional_error_bound(depth, order, 3, 2., 1.)}
            for depth in [1, 2, 4] for order in [8, 16, 32, 64]],
        "full_objective_complete": False,
    }
    save(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="+", default=["near", "distant", "stress"],
                        choices=["near", "distant", "stress"])
    parser.add_argument("--budget-bytes", type=int, default=1_600_000_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identities(args.output/"identities-and-costs.json")
    results = []
    for case in args.cases:
        # Low-bandwidth control is labeled and never substitutes for physical resolution.
        for fraction, depth, order in [(1e-3, 1, 8), (1., 1, 8), (1., 1, 16),
                                       (1., 1, 32), (1., 2, 8), (1., 2, 12)]:
            print("Attempt", case, fraction, depth, order, flush=True)
            path = args.output/f"{case}-f{fraction:g}-L{depth}-p{order}.json"
            result = trial(case, fraction, depth, order, path, budget=args.budget_bytes)
            results.append(result)
            print({key: result.get(key) for key in ["status", "relative_score_error",
                  "maximum_excess_jump_across_leaf_boundary", "construction_geometry_pairs"]}, flush=True)
            save(args.output/"progress.json", {"attempts": results})
            del result
            gc.collect()
    save(args.output/"complete.json", {"attempts": results,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
        "full_objective_complete": False})


if __name__ == "__main__":
    main()
