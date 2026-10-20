"""Measured input-to-output comparison, including reusable direct evaluators."""

import argparse
import gc
import hashlib
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import finufft  # Import/library startup is excluded for every method.
import numpy as np

from ..chart_experiments import setup
from ..compiled_reference import projected_chart
from ..timing.benchmark import measure
from .transform import TimePlan, image_coefficients


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "research/kepler_transform").rglob("*.py"))
            if "executed_sources" not in p.parts}


def run(directory, case, frames, side, counts, repeats, cache_bytes):
    directory.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    started = perf_counter()
    chart, center, _, geom, _ = setup(case)
    direct_model_seconds = perf_counter() - started
    started = perf_counter()
    projected_chart(chart, center[None], geom)
    jit_seconds = perf_counter() - started
    rng = np.random.default_rng(20261103)
    span = float(max(abs(geom.times)))
    times = np.r_[-span, np.sort(rng.uniform(-span, span, frames - 2)), span]
    corners = np.array(list(product([-1., 1.], repeat=6)))
    queries = np.r_[corners, rng.uniform(-1, 1, (max(0, max(counts) - 64), 6))][:max(counts)]
    images = rng.normal(size=(frames, side, side))
    coefficients = image_coefficients(images)
    fft_timing = measure(lambda: image_coefficients(images), repeats)
    rho = 1.5 if case == "stress" else 2.
    plans, fields, setup_times, compression_times = {}, {}, {}, {}
    for backend in ("dense", "nufft"):
        started = perf_counter()
        plans[backend] = TimePlan(case, times, (side, side), rho=rho, compression_backend=backend)
        setup_times[backend] = perf_counter() - started
        compression_times[backend] = measure(lambda: plans[backend].transform(coefficients), repeats)
        fields[backend] = plans[backend].transform(coefficients)
    plan = plans["dense"]
    field_error = float(np.max(abs(fields["dense"] - fields["nufft"])) / np.sum(abs(coefficients)))
    if not field_error < 1e-10:
        raise ValueError("Fast cosine moments disagree with dense compression")
    save(directory / "certificate.json", plan.proof)
    result = {
        "source_sha256": sources,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {n: version(n) for n in ["numpy", "scipy", "numba", "finufft", "python-flint"]},
            "threads": {n: os.environ.get(n) for n in
                        ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]},
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        },
        "workload": {
            "case": case, "span_days": 2 * span, "exposures": frames,
            "image_side": side, "fourier_modes_per_image": side * side,
            "time_order": plan.order, "frequency_bound": plan.proof["frequency_bound"],
            "exact_arithmetic_kernel_error_bound": plan.proof["exact_arithmetic_kernel_error_bound"],
            "input": "synthetic real periodic common-grid images; all Fourier modes retained",
            "cadence": "irregular, including both endpoints", "seed": 20261103,
            "query_counts": counts, "dense_cache_budget_bytes": cache_bytes,
            "whole_original_six_dimensional_box": True,
            "real_survey_data": False, "floating_point_error_certified": False,
            "full_fkt_objective_complete": False,
        },
        "setup": {"direct_model_seconds": direct_model_seconds,
                  "jit_startup_seconds_excluded_for_both": jit_seconds,
                  "time_plan_seconds_including_proof": setup_times},
        "pixel_fft": fft_timing, "compression": compression_times,
        "dense_vs_fast_compression_l1_error": field_error,
        "measurements": [],
        "timing_convention": "Setup once; application median of independent batches. No output cache.",
        "native_nufft_workspace_bytes_in_array_counts": False,
    }
    # Independent solver on eight box corners and actual exposure times.
    small = queries[:min(8, len(queries))]
    k = rng.choice(side * side, size=min(4, side * side), replace=False)
    sparse_sources = np.column_stack((np.repeat(times / span, len(k)),
                                     np.tile(plan.frequencies[k] / plan.phase.frequency_scale, (frames, 1))))
    independent = np.exp(1j * plan.phase.independent(small, sparse_sources)) @ coefficients.reshape(frames, -1)[:, k].ravel()
    sparse = np.zeros_like(coefficients).reshape(frames, -1)
    sparse[:, k] = coefficients.reshape(frames, -1)[:, k]
    sparse = sparse.reshape(coefficients.shape)
    sparse_actual = plan.evaluate(plan.transform(sparse), small, backend="stream")
    independent_error = float(max(abs(sparse_actual - independent)) / np.sum(abs(sparse)))
    if independent_error > plan.proof["exact_arithmetic_kernel_error_bound"] + 1e-11:
        raise ValueError("Independent orbit solver comparison failed")
    result["independent_sparse_score_error_per_l1"] = independent_error
    np.savez_compressed(directory / "inputs.npz", images=images, times=times, queries=queries)
    retained = {}
    for count in counts:
        q = queries[:count]
        row = {"queries": count, "counts": plan.counts(count), "direct": {}, "compressed": {}}
        reference = None
        for family in ("direct", "compressed"):
            for backend in ("stream", "dense", "nufft"):
                builder = plan.direct_plan if family == "direct" else plan.query_plan
                data = coefficients if family == "direct" else fields["dense"]
                started = perf_counter()
                try:
                    evaluator = builder(q, backend=backend, max_cache_bytes=cache_bytes)
                except ValueError as error:
                    if backend != "dense" or "cache exceeds" not in str(error):
                        raise
                    row[family][backend] = {"skipped": str(error)}
                    continue
                construction = perf_counter() - started
                actual = evaluator.apply(data)
                if reference is None:
                    reference = actual.copy()
                error = float(max(abs(actual - reference)) / np.sum(abs(coefficients)))
                if error > plan.proof["exact_arithmetic_kernel_error_bound"] + 1e-11:
                    raise ValueError(f"Score comparison failed: {family}, {backend}: {error}")
                # A second, changed coefficient vector tests actual reuse, not output memoization.
                changed = coefficients * np.exp(1j * np.arange(frames)[:, None, None])
                if family == "compressed":
                    changed_actual = evaluator.apply(plan.transform(changed))
                    changed_direct = plan.direct_plan(q[:8], backend="stream").apply(changed)
                    if max(abs(changed_actual[:8] - changed_direct)) / np.sum(abs(changed)) > 1e-8:
                        raise ValueError("Changed-input reuse failed")
                row[family][backend] = {
                    "setup_seconds": construction,
                    "apply": measure(lambda: evaluator.apply(data), repeats),
                    "sampled_score_error_per_coefficient_l1": error,
                    "persistent_numpy_array_bytes_excluding_native_workspaces": evaluator.persistent_array_bytes,
                }
                if family == "compressed" and backend == "nufft":
                    fast_actual = evaluator.apply(fields["nufft"])
                    fast_error = float(max(abs(fast_actual - reference)) / np.sum(abs(coefficients)))
                    if fast_error > 1e-8:
                        raise ValueError("Fast moment plus output NUFFT comparison failed")
                    row["fast_compression_and_nufft_error_per_l1"] = fast_error
                    retained[f"compressed_{count}"] = fast_actual
                print(json.dumps({"case": case, "frames": frames, "side": side, "queries": count,
                                  "family": family, "backend": backend, **row[family][backend]}), flush=True)
                del evaluator
                gc.collect()
        retained[f"direct_{count}"] = reference
        fft = fft_timing["median_seconds"]
        doptions = {name: value for name, value in row["direct"].items() if "skipped" not in value}
        poptions = {name: value for name, value in row["compressed"].items() if "skipped" not in value}
        first_direct = {name: fft + direct_model_seconds + v["setup_seconds"] + v["apply"]["median_seconds"]
                        for name, v in doptions.items()}
        reuse_direct = {name: fft + v["apply"]["median_seconds"] for name, v in doptions.items()}
        first_partial, reuse_partial, fresh_queries = {}, {}, {}
        for compression, timing in compression_times.items():
            for backend, v in poptions.items():
                key = f"{compression}_compression+{backend}_output"
                application = fft + timing["median_seconds"] + v["apply"]["median_seconds"]
                reuse_partial[key] = application
                first_partial[key] = application + setup_times[compression] + v["setup_seconds"]
                fresh_queries[key] = v["setup_seconds"] + v["apply"]["median_seconds"]
        def best(options):
            method = min(options, key=options.get)
            return {"method": method, "seconds": options[method]}
        row["comparison"] = {
            "first_use_direct": best(first_direct), "first_use_compressed": best(first_partial),
            "changed_input_fixed_queries_direct": best(reuse_direct),
            "changed_input_fixed_queries_compressed": best(reuse_partial),
            "new_queries_existing_compressed_field": best(fresh_queries),
            "new_queries_existing_input_fields_direct": best({
                n: v["setup_seconds"] + v["apply"]["median_seconds"] for n, v in doptions.items()}),
        }
        for prefix, a, b in [
            ("first_use", "first_use_direct", "first_use_compressed"),
            ("changed_input_fixed_queries", "changed_input_fixed_queries_direct", "changed_input_fixed_queries_compressed"),
            ("new_queries", "new_queries_existing_input_fields_direct", "new_queries_existing_compressed_field"),
        ]:
            row["comparison"][prefix + "_direct_over_compressed"] = (
                row["comparison"][a]["seconds"] / row["comparison"][b]["seconds"])
        result["measurements"].append(row)
        result["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        save(directory / "results.json", result)
        print(json.dumps({"comparison": row["comparison"], "queries": count}), flush=True)
    np.savez_compressed(directory / "scores.npz", **retained)
    if source_hashes() != sources:
        raise RuntimeError("Source files changed during benchmark")
    result["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in directory.iterdir() if p.name != "results.json"}
    save(directory / "results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", choices=["near", "distant", "stress"], default="near")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--counts", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-bytes", type=int, default=600_000_000)
    args = parser.parse_args()
    if args.frames < 2 or args.side < 2 or min(args.counts + [args.repeats, args.cache_bytes]) < 1:
        parser.error("Positive counts/repeats/budget and at least two frames and pixels per side required")
    run(args.directory, args.case, args.frames, args.side, sorted(set(args.counts)),
        args.repeats, args.cache_bytes)


if __name__ == "__main__":
    main()
