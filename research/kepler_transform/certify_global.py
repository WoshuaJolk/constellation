"""Certify global source interpolation hypotheses on unchanged orbit boxes.

This is a proof-construction experiment. It records sufficient orders and
their resource costs even when they are too large to execute.
"""

import argparse
import hashlib
from itertools import product
import json
from math import lgamma, log, exp
from pathlib import Path
from time import perf_counter

import numpy as np
from flint import acb, arb, ctx

from research.detection_limit.interval_orbits import upper_float
from .chart_cover import enclosing_cell
from .chart_experiments import save, setup
from .complex_source_certificate import root_and_gradients, global_speed_certificate
from .factorization import ARCSEC
from .source_butterfly import box_center


def log_local_errors(orders, time_D, rho, frequency_amplitudes):
    lambdas = np.sqrt(2*np.asarray(orders)-1)
    p = orders[0]
    errors = [log(1+lambdas[0])+log(2)+time_D-(p-1)*log(rho)-log(rho-1)]
    for p, leb, amplitude in zip(orders[1:], lambdas[1:], frequency_amplitudes):
        if amplitude == 0:
            errors.append(-np.inf)
        elif amplitude >= 2*(p+1):
            errors.append(np.inf)
        else:
            errors.append(log(1+leb)+log(2)+p*log(amplitude/2)-lgamma(p+1)
                          -log(1-amplitude/(2*(p+1))))
    return np.array(errors), np.log(lambdas)


def log_error(orders, depth, time_D, rho, frequency_amplitudes):
    errors, lebesgue = log_local_errors(orders, time_D, rho, frequency_amplitudes)
    local = np.logaddexp.reduce(errors+np.r_[0., np.cumsum(lebesgue)[:-1]])
    amplification = np.logaddexp.reduce(np.arange(depth+1)*sum(lebesgue))
    return float(local+amplification)


def exact_error(orders, depth, time_D, rho, frequency_amplitudes):
    """Outward Arb evaluation, after a floating-point degree proposal."""
    with ctx.workprec(128):
        lebesgue = [(arb(2*p-1)).sqrt() for p in orders]
        p = orders[0]
        ellipse = arb(float(rho))
        errors = [(1+lebesgue[0])*2*arb(float(time_D)).exp()*ellipse**(-(p-1))/(ellipse-1)]
        for p, leb, amplitude in zip(orders[1:], lebesgue[1:], frequency_amplitudes):
            half = arb(float(amplitude))/2
            if not half < p+1:
                raise ValueError("Bessel tail geometric ratio is not below one")
            errors.append((1+leb)*2*half**p/arb.fac_ui(p)/(1-half/(p+1)))
        local, prefix = arb(0), arb(1)
        for error, leb in zip(errors, lebesgue):
            local += prefix*error
            prefix *= leb
        amplification, power = arb(0), arb(1)
        for _ in range(depth+1):
            amplification += power
            power *= prefix
        return upper_float(local*amplification)


def choose_orders(depth, time_D, rho, frequency_amplitudes, tolerance=1e-8):
    orders = [2, max(2, int(frequency_amplitudes[0]/2)),
              max(2, int(frequency_amplitudes[1]/2))]
    for _ in range(20_000):
        if max(orders) > 4096:
            raise ValueError("Sufficient interpolation degree exceeds proof-search budget")
        error = log_error(orders, depth, time_D, rho, frequency_amplitudes)
        if error <= log(tolerance)-1e-10:
            exact = exact_error(orders, depth, time_D, rho, frequency_amplitudes)
            if exact <= tolerance:
                return orders, exact
        # Increasing an initially tiny degree can first increase the Lebesgue
        # amplification. Follow the dominant error contribution through that
        # transient instead of greedily minimizing one-step total error.
        local, lebesgue = log_local_errors(orders, time_D, rho, frequency_amplitudes)
        axis = int(np.argmax(local+np.r_[0., np.cumsum(lebesgue)[:-1]]))
        orders[axis] += 1
    raise ValueError("Proof-order selection did not converge")


def partitioned_gradients(chart, center, widths, time_center, time_half_width, rho,
                          *, partition_depth=0):
    """Analysis-only dyadic cover; derivatives keep the ORIGINAL cell scale."""
    attempts = []
    original_center, original_widths = np.asarray(center), np.asarray(widths)

    def visit(bounds, depth):
        subcenter, subwidths = enclosing_cell(original_center, original_widths, bounds)
        try:
            gradients, record = root_and_gradients(
                chart, subcenter, subwidths, time_center, time_half_width, rho,
                derivative_scales=original_widths)
            if not (record["connected_to_known_reference_emission_root"]
                    and record["positive_real_retarded_root_derivative"]):
                raise ValueError("Root branch connection to physical reference is not proved")
            record.update(normalized_analysis_bounds=bounds.tolist(),
                          physical_center=subcenter.tolist(), physical_widths=subwidths.tolist())
            return gradients, [record]
        except ValueError as error:
            attempts.append({"bounds": bounds.tolist(), "reason": str(error)})
            if depth >= partition_depth:
                raise ValueError(f"Analysis cover failed at depth {depth}: {error}") from error
        axis = depth % 2  # inverse distance and radial rate; never drop a cell
        midpoint = (bounds[axis, 0]+bounds[axis, 1])/2
        left, right = bounds.copy(), bounds.copy()
        left[axis, 1], right[axis, 0] = midpoint, midpoint
        a, a_records = visit(left, depth+1)
        b, b_records = visit(right, depth+1)
        union = [[acb(x.real.union(y.real), x.imag.union(y.imag))
                  for x, y in zip(aa, bb)] for aa, bb in zip(a, b)]
        return union, a_records+b_records

    gradients, records = visit(np.tile([-1., 1.], (6, 1)), 0)
    return gradients, {"certified": True, "analysis_cells": records,
                       "failed_parent_enclosures": attempts,
                       "connected_to_known_reference_emission_root": True,
                       "positive_real_retarded_root_derivative": True}


def certify(case, depth, rho, output, *, partition_depth=0):
    started = perf_counter()
    if output.exists():
        raise FileExistsError(output)
    if (not isinstance(depth, int) or depth < 0 or depth > 20
            or sum(2**(5*l+depth) for l in range(depth+1)) > 10_000):
        raise ValueError("Certificate cell budget exceeded or invalid depth; domain not reduced")
    if not isinstance(partition_depth, int) or not 0 <= partition_depth <= 10:
        raise ValueError("Analysis partition depth must be in 0..10")
    chart, root_center, root_widths, geometry, _ = setup(case)
    try:
        global_physical = global_speed_certificate(chart, root_center, root_widths)
    except ValueError as error:
        global_physical = {"global_all_time_speed_proof": False, "reason": str(error),
                           "not_required_for_reference_connected_observation_branch": True}
    span = float(max(abs(geometry.times)))
    records, failures = [], []
    maximum_time_D = arb(0)
    maximum_frequency = [arb(0), arb(0)]
    with ctx.workprec(128):
        for level in range(depth+1):
            source_depth = depth-level
            source_half = 2.**(-source_depth)
            for target in product(range(2**level), repeat=6):
                normalized_center = box_center(target, level)
                bounds = np.column_stack((normalized_center-2.**(-level), normalized_center+2.**(-level)))
                center, widths = enclosing_cell(root_center, root_widths, bounds)
                for time_index in range(2**source_depth):
                    time_center = float(box_center([time_index], source_depth)[0])*span
                    try:
                        real_gradients, real_record = partitioned_gradients(
                            chart, center, widths, time_center, source_half*span, rho=1.,
                            partition_depth=partition_depth)
                        complex_gradients, complex_record = partitioned_gradients(
                            chart, center, widths, time_center, source_half*span, rho=rho,
                            partition_depth=partition_depth)
                        if not all(r["connected_to_known_reference_emission_root"]
                                   and r["positive_real_retarded_root_derivative"]
                                   for r in [real_record, complex_record]):
                            raise ValueError("Root branch connection to the physical reference is not proved")
                    except ValueError as error:
                        failures.append({"level": level, "target": target, "time_index": time_index,
                                         "reason": str(error)})
                        continue
                    # Frequencies remain real when bounding time interpolation.
                    # Their full root range [-1,1] covers every source-frequency box.
                    time_D = sum((abs(value.imag) for axis in complex_gradients for value in axis), arb(0))/arb(ARCSEC)
                    frequency = [source_half*sum((abs(value.real) for value in axis), arb(0))/arb(ARCSEC)
                                 for axis in real_gradients]
                    maximum_time_D = maximum_time_D.max(time_D)
                    maximum_frequency = [old.max(new) for old, new in zip(maximum_frequency, frequency)]
                    records.append({"level": level, "target": target, "time_index": time_index,
                                    "physical_cell_center": center.tolist(), "physical_cell_widths": widths.tolist(),
                                    "time_log_modulus_bound": upper_float(time_D),
                                    "frequency_phase_amplitudes": [upper_float(v) for v in frequency],
                                    "real_root": real_record, "complex_root": complex_record})
            print("Proof", case, "rho", rho, "level", level, "records", len(records), "failures", len(failures), flush=True)
    result = {"case": case, "depth": depth, "ellipse_parameter": rho,
              "maximum_analysis_partition_depth": partition_depth,
              "chart": chart.to_dict(), "root_center": root_center.tolist(), "root_widths": root_widths.tolist(),
              "time_half_span_days": span, "frequency_scale": 1/ARCSEC,
              "global_physical_root_certificate": global_physical,
              "records": records, "failures": failures,
              "all_hypotheses_certified": not failures,
              "maximum_time_log_modulus_bound": upper_float(maximum_time_D),
              "maximum_frequency_phase_amplitudes": [upper_float(v) for v in maximum_frequency],
              "physical_orbit_domain_shrunk": False,
              "fast_floating_point_output_certified": False, "full_objective_complete": False}
    if not failures:
        orders, bound = choose_orders(depth, result["maximum_time_log_modulus_bound"], rho,
                                       result["maximum_frequency_phase_amplitudes"])
        rank = int(np.prod(orders))
        pairs = [2**(6*l+3*(depth-l)) for l in range(depth+1)]
        edges = sum(pairs[1:])*8
        result.update(sufficient_orders=orders, exact_arithmetic_kernel_error_bound=bound,
                      rank=rank, final_field_bytes=16*rank*2**(6*depth),
                      full_tree_gate_bytes=16*rank*edges)
    result["certificate_construction_seconds"] = perf_counter()-started
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob("*.py"))}
    save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--cases", nargs="+", default=["near", "distant", "stress"])
    parser.add_argument("--partition-depth", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in args.cases:
        for rho in [1.2, 1.5, 2.]:
            result = certify(case, args.depth, rho, args.output/f"{case}-rho{rho:g}.json",
                             partition_depth=args.partition_depth)
            summary = {k: result.get(k) for k in ["case", "depth", "ellipse_parameter",
                       "all_hypotheses_certified", "sufficient_orders", "exact_arithmetic_kernel_error_bound",
                       "rank", "final_field_bytes", "full_tree_gate_bytes"]}
            results.append(summary)
            print(json.dumps(summary), flush=True)
            save(args.output/"summary.json", results)


if __name__ == "__main__":
    main()
