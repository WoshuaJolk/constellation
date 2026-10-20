"""End-to-end finite-detector likelihood experiment with strong pixel baselines."""

import argparse
import gc
import hashlib
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from ..factorization import ARCSEC
from ..temporal.benchmark import source_hashes
from ..temporal.transform import TimePlan
from ..timing.benchmark import measure
from .fields import CommonDetector, l1_upper, trim_fields, sum_bounds
from .pixel_cache import PixelCache
from .plan import DetectorTimePlan, domain_source_box, statistic_intervals


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(directory, frames, side, counts, repeats, backend, cache_bytes):
    directory.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    rng = np.random.default_rng(20261105)
    times = np.r_[-4., np.sort(rng.uniform(-4, 4, frames - 2)), 4.]
    sigma = rng.uniform(1.2, 1.6, (frames, 2))
    angles = rng.uniform(-.03, .03, frames)
    matrices = np.array([[[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] for a in angles])
    matrices *= rng.uniform(.97, 1.03, (frames, 1, 1))
    offsets = rng.uniform(-.3, .3, (frames, 2))
    flux_scales = rng.uniform(.8, 1.2, frames)
    variance = rng.uniform(.8, 1.2, (frames, side, side))
    masks = rng.uniform(size=variance.shape) < .03
    images = rng.normal(size=variance.shape) * np.sqrt(variance)
    images += rng.normal(size=(frames, 1, 1)) * .3
    axis = np.arange(side) - (side - 1) / 2
    for j in range(frames):
        position = -np.linalg.solve(matrices[j], offsets[j])
        px = ndtr((axis + .5 - position[0]) / sigma[j, 0]) - ndtr((axis - .5 - position[0]) / sigma[j, 0])
        py = ndtr((axis + .5 - position[1]) / sigma[j, 1]) - ndtr((axis - .5 - position[1]) / sigma[j, 1])
        images[j] += 10. * flux_scales[j] * py[:, None] * px[None, :]
    queries = np.r_[np.zeros((1, 6)), np.array(list(product([-1., 1.], repeat=6))),
                    rng.uniform(-1, 1, (max(0, max(counts) - 65), 6))][:max(counts)]
    started = perf_counter()
    cover = domain_source_box("near")
    cover_time = perf_counter() - started
    half = float(np.ceil(max(cover["relative_reference_position_bound_arcsec"])))
    started = perf_counter()
    detector = CommonDetector(variance, sigma, matrices, offsets=offsets, masks=masks,
                              flux_scales=flux_scales, source_half_width=half,
                              max_bytes=4_000_000_000, spectral_sigma=8.)
    detector_setup = perf_counter() - started
    print(json.dumps({"phase": "detector_plan", "seconds": detector_setup, "full_side": detector.side}), flush=True)
    started = perf_counter()
    full_n, full_i, detector_report = detector.prepare(images)
    preparation = perf_counter() - started
    started = perf_counter()
    temporary_n, i, trim = trim_fields(full_n, full_i, [1e-5, 1e-7])
    ncut, numerator_report = detector.choose_numerator_cutoff(images, 1e-5)
    mid = detector.side // 2
    n = np.ascontiguousarray(full_n[:, mid - ncut:mid + ncut + 1, mid - ncut:mid + ncut + 1])
    trim_time = perf_counter() - started
    del full_n, full_i, temporary_n
    gc.collect()
    print(json.dumps({"phase": "prepared", "seconds": preparation, "trim_seconds": trim_time, **trim}), flush=True)
    epsilon_n = min(1e-8, 1e-6 / max(1., l1_upper(n)))
    epsilon_i = min(1e-8, 1e-8 / max(1., l1_upper(i)))
    # Warm the shared compiled orbit solver before timing either route.
    warm = TimePlan("near", [-1e-7, 1e-7])
    warm.positions(np.zeros((1, 6)), [-4., 4.])
    del warm
    started = perf_counter()
    plan = DetectorTimePlan("near", times, n.shape[1:], detector.spacing,
                            tolerance=epsilon_n, compression_backend=backend)
    info_plan = DetectorTimePlan("near", times, i.shape[1:], detector.spacing,
                                 tolerance=epsilon_i, compression_backend=backend)
    plan_time = perf_counter() - started
    started = perf_counter()
    gn, gi = plan.transform(n), info_plan.transform(i)
    compression = perf_counter() - started
    print(json.dumps({"phase": "compressed", "orders": [plan.order, info_plan.order],
                      "field_sides": [n.shape[1], i.shape[1]], "seconds": compression}), flush=True)
    nbound = sum_bounds([numerator_report["numerator_model_error_bound"], plan.score_error_bound(n)])
    ibound = sum_bounds([x["information_model_error_bound"] for x in detector_report["exposures"]] +
                       [trim["discarded_information_l1_bound"], info_plan.score_error_bound(i)])

    def new_image_field(data):
        coefficients, budget = detector.prepare_numerator(data, ncut)
        if budget["numerator_model_error_bound"] + plan.score_error_bound(coefficients) > 2e-5:
            raise ValueError("Changed image requires a wider frequency or time plan")
        return plan.transform(coefficients)

    # Fixed detector model lets both routes reuse information. New pixel data
    # still undergo the complete numerator preparation and time compression.
    preprocessing = measure(lambda: new_image_field(images), repeats, minimum_seconds=.01)
    result = {
        "source_sha256": sources,
        "environment": {"versions": {x: version(x) for x in ["numpy", "scipy", "finufft", "python-flint", "numba"]},
                        "thread_limits": {x: os.environ.get(x) for x in
                                          ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"]}},
        "workload": {"frames": frames, "pixels_per_image": side**2, "image_side": side,
                     "span_days": 8., "case": "original full near box", "seed": 20261105,
                     "synthetic_independent_exposures": True, "real_survey_stack": False,
                     "injected_flux": 10., "injected_query_index": 0,
                     "pointing": "tracks the reference orbit in the certified circular-observer model",
                     "full_detector_side": detector.side, "retained_fourier_side": n.shape[1],
                     "information_fourier_side": i.shape[1], "information_time_order": info_plan.order,
                     "time_order": plan.order, "compression_backend": backend, "cache_budget_bytes": cache_bytes},
        "source_domain_proof": cover, "time_proof": plan.proof,
        "information_time_proof": info_plan.proof,
        "detector_report": detector_report, "trim": trim, "numerator_report": numerator_report,
        "error_budgets": {"numerator": nbound, "information": ibound,
                         "all_error_components_exact_arithmetic_model_only": True,
                         "coefficient_or_orbit_roundoff_certified": False},
        "setup_seconds": {"source_domain_proof": cover_time, "detector_plan": detector_setup,
                          "full_field_preparation": preparation, "trim": trim_time,
                          "time_plan": plan_time, "compression_both_fields": compression},
        "changed_image_preparation_and_compression": preprocessing,
        "measurements": [], "full_objective_complete": False,
    }
    np.savez_compressed(directory / "inputs.npz", images=images, variances=variance, masks=masks,
                        sigma=sigma, matrices=matrices, offsets=offsets, flux_scales=flux_scales,
                        times=times, queries=queries)
    outputs = {}
    for count in counts:
        q = queries[:count]
        started = perf_counter()
        positions = (plan.positions(q, times) - plan.center_input) / ARCSEC
        direct_geometry = perf_counter() - started
        started = perf_counter()
        streaming = PixelCache(detector, positions, backend="stream", max_bytes=cache_bytes)
        stream_setup = perf_counter() - started
        direct_timing = measure(lambda: streaming.apply(images), repeats, minimum_seconds=.01)
        dn, di = detector.direct(images, positions)
        sn, si = streaming.apply(images)
        if max(np.max(abs(sn - dn)), np.max(abs(si - di))) > 1e-8:
            raise ValueError("Separable direct baseline disagrees with independent pixel integrals")
        started = perf_counter()
        fast_query = plan.query_plan(q, backend="nufft")
        info_query = info_plan.query_plan(q, backend="nufft")
        fast_setup = perf_counter() - started
        an, ai = fast_query.apply(gn), info_query.apply(gi)
        fast_both = measure(lambda: np.array([fast_query.apply(gn), info_query.apply(gi)]),
                            repeats, minimum_seconds=.01)
        fast_n = measure(lambda: fast_query.apply(gn), repeats, minimum_seconds=.01)
        errors = [float(np.max(abs(an - dn))), float(np.max(abs(ai - di)))]
        if errors[0] > nbound + 1e-8 or errors[1] > ibound + 1e-9:
            raise ValueError(f"End-to-end pixel likelihood mismatch: {errors}, budgets {nbound, ibound}")
        intervals = statistic_intervals(an.real, ai.real, nbound, ibound)
        resolved = intervals["positive_information_resolved"]
        if not np.all((dn[resolved] / np.sqrt(di[resolved]) >= intervals["snr_interval"][resolved, 0] - 1e-8)
                      & (dn[resolved] / np.sqrt(di[resolved]) <= intervals["snr_interval"][resolved, 1] + 1e-8)):
            raise ValueError("Independent direct S/N leaves propagated model-error interval")
        row = {"queries": count, "direct_geometry_seconds": direct_geometry,
               "direct_stream_setup_seconds": stream_setup,
               "direct_pixels": direct_timing, "compressed_query_setup_seconds": fast_setup,
               "compressed_both_statistics": fast_both, "compressed_numerator_only": fast_n,
               "maximum_numerator_error": errors[0], "maximum_information_error": errors[1],
               "unresolved_information_queries": int(np.sum(~resolved)),
               "maximum_snr_error_resolved": float(np.max(abs(an.real[resolved] / np.sqrt(ai.real[resolved])
                                                             - dn[resolved] / np.sqrt(di[resolved])))),
               "injected_flux_estimate": float(an.real[0] / ai.real[0]),
               "injected_snr": float(an.real[0] / np.sqrt(ai.real[0])),
               "best_resolved_query_is_injected": bool(np.flatnonzero(resolved)[
                   np.argmax(an.real[resolved] / np.sqrt(ai.real[resolved]))] == 0),
               "cached_pixels": {}}
        for cache_backend in ("dense", "factor"):
            started = perf_counter()
            try:
                cache = PixelCache(detector, positions, backend=cache_backend, max_bytes=cache_bytes)
            except ValueError as error:
                if "budget" not in str(error):
                    raise
                row["cached_pixels"][cache_backend] = {"skipped": str(error)}
                continue
            seconds = perf_counter() - started
            cn, ci = cache.apply(images)
            if max(np.max(abs(cn - dn)), np.max(abs(ci - di))) > 1e-8:
                raise ValueError("Cached pixel baseline disagrees with independent pixel integrals")
            row["cached_pixels"][cache_backend] = {
                "setup_seconds": seconds, "bytes": cache.persistent_bytes,
                "application": measure(lambda: cache.apply(images), repeats, minimum_seconds=.01)}
            del cache
            gc.collect()
        setup = sum(result["setup_seconds"].values())
        direct_options = {"stream": direct_geometry + stream_setup + direct_timing["median_seconds"]}
        reuse_options = {"stream": direct_timing["median_seconds"]}
        for name, v in row["cached_pixels"].items():
            if "skipped" not in v:
                direct_options[name] = direct_geometry + v["setup_seconds"] + v["application"]["median_seconds"]
                reuse_options[name] = v["application"]["median_seconds"]
        first_fast = setup + fast_setup + fast_both["median_seconds"]
        reuse_fast = preprocessing["median_seconds"] + fast_n["median_seconds"]
        row["comparison"] = {
            "first_direct_seconds": min(direct_options.values()), "first_compressed_seconds": first_fast,
            "first_direct_method": min(direct_options, key=direct_options.get),
            "new_image_direct_seconds": min(reuse_options.values()), "new_image_compressed_seconds": reuse_fast,
            "new_image_direct_method": min(reuse_options, key=reuse_options.get),
            "first_direct_over_compressed": min(direct_options.values()) / first_fast,
            "new_image_direct_over_compressed": min(reuse_options.values()) / reuse_fast,
            "information_is_reused_for_both_on_changed_images": True,
        }
        result["measurements"].append(row)
        outputs[f"fast_n_{count}"], outputs[f"fast_i_{count}"] = an, ai
        outputs[f"direct_n_{count}"], outputs[f"direct_i_{count}"] = dn, di
        print(json.dumps({"queries": count, **row["comparison"], "errors": errors}), flush=True)
        save(directory / "results.json", result)
        del fast_query, info_query, streaming
        gc.collect()
    # A different input validates the reusable pipeline; all original pixel
    # masks/noise/PSF geometry stay fixed, and no output vector is substituted.
    changed = .7 * images + rng.normal(size=images.shape) * .01
    field = new_image_field(changed)
    q = queries[:8]
    actual = plan.evaluate(field, q, backend="nufft")
    reference = detector.direct(changed, (plan.positions(q, times) - plan.center_input) / ARCSEC)[0]
    if max(abs(actual - reference)) > 1e-4:
        raise ValueError("Changed-input end-to-end comparison failed")
    result["changed_input_maximum_numerator_error"] = float(max(abs(actual - reference)))
    np.savez_compressed(directory / "scores.npz", **outputs)
    if source_hashes() != sources:
        raise RuntimeError("Source changed during benchmark")
    result["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in directory.iterdir() if p.name != "results.json"}
    save(directory / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--side", type=int, default=33)
    parser.add_argument("--counts", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=["dense", "nufft"], default="dense")
    parser.add_argument("--cache-bytes", type=int, default=1_200_000_000)
    args = parser.parse_args()
    if args.frames < 2 or args.side < 3 or args.side % 2 == 0 or min(args.counts + [args.repeats]) < 1:
        parser.error("At least two frames, odd image side >=3, and positive query/repeat counts required")
    run(args.directory, args.frames, args.side, sorted(set(args.counts)), args.repeats,
        args.backend, args.cache_bytes)
