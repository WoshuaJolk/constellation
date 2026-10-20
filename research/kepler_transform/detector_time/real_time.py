"""Empirical time representation of the retained, real-ephemeris ZTF stack.

This runs the PFKT time construction, not just its Fourier image adapter.
No whole-domain interpolation certificate is claimed. A quadratic observer
extension passes through the three recorded positions; only scores at those
actual exposures define the target operator.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BarycentricInterpolator

from ..compiled_reference import projected_chart
from ..factorization import ARCSEC
from ..source_butterfly import cardinal
from ..temporal.benchmark import source_hashes
from ..temporal.transform import FourierEvaluator
from .benchmark import save
from .pixel_cache import PixelCache
from .scaling import real, streamed, timed_values


class EmpiricalObserverTimePlan:
    """Chebyshev time fields using recorded observer samples.

    The interpolating extension defines auxiliary query nodes; it is not a
    physical ephemeris between observations. Accuracy must be measured against
    direct evaluation at the recorded observations. This three-sample research
    implementation intentionally refuses other observer sample counts.
    """

    def __init__(self, context, order):
        if isinstance(order, bool) or not isinstance(order, int) or not 2 <= order <= 64:
            raise ValueError("Integer order in 2..64 required")
        self.times = np.array(context.times, dtype=float, copy=True)
        observers = np.array(context.observers, dtype=float, copy=True)
        if (self.times.shape != (3,) or observers.shape != (3, 3)
                or not np.all(np.isfinite(self.times)) or not np.all(np.isfinite(observers))
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Three finite, strictly ordered recorded observer samples required")
        self.origin = (self.times[0] + self.times[-1]) / 2
        self.span = (self.times[-1] - self.times[0]) / 2
        self.chart, self.axes = context.chart, np.array(context.axes, copy=True)
        self.lower = context.queries.min(axis=0)
        self.upper = context.queries.max(axis=0)
        self.center = (self.lower + self.upper) / 2
        self.shape = tuple(context.gn.shape[1:])
        if context.gn.shape != context.gi.shape:
            raise ValueError("Common numerator/information Fourier shapes required")
        self.spacing = context.detector.spacing
        scaled = (self.times - self.origin) / self.span
        # Explicit weights avoid random permutation in scipy's weight builder.
        differences = scaled[:, None] - scaled[None, :]
        np.fill_diagonal(differences, 1.)
        self.observer = BarycentricInterpolator(
            scaled, observers, axis=0, wi=1 / np.prod(differences, axis=1))
        self.order = order
        alpha = (np.arange(order) + .5) * np.pi / order
        nodes = np.cos(alpha)
        self.node_times = self.origin + self.span * nodes
        self.interpolation = cardinal(scaled, nodes, (-1.)**np.arange(order) * np.sin(alpha))
        modes = [np.arange(-(n // 2), n // 2 + 1) for n in self.shape]
        xx, yy = np.meshgrid(*modes, indexing="ij")
        self.frequencies = np.column_stack([xx.ravel(), yy.ravel()]) * self.spacing
        self.center_input = self.positions(self.center[None], self.times)[0]
        self.center_nodes = self.positions(self.center[None], self.node_times)[0]
        self.modulation = np.exp(1j * self.center_input @ self.frequencies.T)

    def positions(self, queries, times):
        q, t = np.asarray(queries, float), np.asarray(times, float)
        if (q.ndim != 2 or q.shape[1] != 6 or not len(q) or not np.all(np.isfinite(q))
                or np.any(q < self.lower) or np.any(q > self.upper)):
            raise ValueError("Finite queries inside the declared real-data box required")
        if (t.ndim != 1 or not len(t) or not np.all(np.isfinite(t))
                or np.any(t < self.times[0]) or np.any(t > self.times[-1])):
            raise ValueError("Times must lie within the observed interval")
        geometry = SimpleNamespace(
            times=t, observers=self.observer((t - self.origin) / self.span),
            normal=np.tile(self.axes[0], (len(t), 1)),
            east=np.tile(self.axes[1], (len(t), 1)),
            north=np.tile(self.axes[2], (len(t), 1)))
        return projected_chart(self.chart, q, geometry) / ARCSEC

    def transform(self, fields):
        data = np.asarray(fields, complex)
        if data.shape != (3, *self.shape) or not np.all(np.isfinite(data)):
            raise ValueError("Finite fields at the three original exposures required")
        return (self.interpolation.T @
                (data.reshape(3, -1) * self.modulation)).reshape(self.order, *self.shape)

    def query_plan(self, queries):
        positions = self.positions(queries, self.node_times) - self.center_nodes
        return FourierEvaluator(positions, self.shape, [self.spacing]*2, backend="nufft")


def errors(n, i, dn, di):
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(i)) or np.any(i.real <= 0):
        raise ValueError("Finite scores and positive information required")
    return {
        "max_numerator_error": float(np.max(abs(n - dn))),
        "max_information_error": float(np.max(abs(i - di))),
        "max_snr_error": float(np.max(abs(n.real / np.sqrt(i.real) - dn / np.sqrt(di)))),
        "max_imaginary_residual": float(max(np.max(abs(n.imag)), np.max(abs(i.imag)))),
    }


def run(directory, count=65536, repeats=3):
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parents[1] / "chart_results/real_inputs"
    ctx = real(count, source)
    q = ctx.queries
    pos = ctx.positions(q)
    dn, di = streamed(ctx.detector, ctx.images, pos)
    threshold = 1e-8
    report = {
        **ctx.metadata, "kind": "real_empirical_time", "time_compression_applied": True,
        "time_interpolation_certified": False, "queries": count,
        "orders": [], "measurements": [], "threshold_max_absolute_snr_error": threshold,
        "detector_setup_seconds": ctx.setup, "source_sha256": source_hashes(),
        "notes": [
            "Actual ZTF pixels and recorded observer positions, with affine WCS and Gaussian PSFs.",
            "Quadratic observer extension matches all three records; no between-exposure physical accuracy claim.",
            "Order selected using all saved queries; an additional independently seeded holdout is checked.",
            "Sampled time errors are not whole-domain guarantees; no circular-observer certificate is reused.",
            "First-use timing excludes research order selection, imports/JIT, data loading, validation and query generation.",
            "Reuse retains geometry and information for both methods; changed image values are processed anew.",
            "Three application samples per timing; setup measured once; local single numerical thread.",
        ],
    }
    arrays = {"queries": q, "direct_n": dn, "direct_i": di}
    selected = None
    for order in [2, 3, 4, 6, 8, 12, 16, 24, 32]:
        plan = EmpiricalObserverTimePlan(ctx, order)
        gn, gi = plan.transform(ctx.gn), plan.transform(ctx.gi)
        evaluator = plan.query_plan(q)
        n, i = evaluator.apply(gn), evaluator.apply(gi)
        row = {"order": order, **errors(n, i, dn, di),
               "stored_field_bytes": gn.nbytes + gi.nbytes}
        row["passes_sampled_threshold"] = row["max_snr_error"] <= threshold
        report["orders"].append(row)
        arrays[f"n_{order}"], arrays[f"i_{order}"] = n, i
        if selected is None and row["passes_sampled_threshold"]:
            selected = order
        print(json.dumps(row), flush=True)
        del evaluator
    if selected is None:
        save(directory / "failed-orders.json", report)
        raise ValueError("No tested order meets the sampled SNR threshold")

    started = perf_counter()
    plan = EmpiricalObserverTimePlan(ctx, selected)
    gn, gi = plan.transform(ctx.gn), plan.transform(ctx.gi)
    plan_setup = perf_counter() - started
    report.update(selected_order=selected, plan_setup_seconds=plan_setup,
                  input_exposures=3, output_time_fields=selected,
                  actually_reduces_time_fields=selected < 3,
                  original_field_bytes=ctx.gn.nbytes + ctx.gi.nbytes)
    compact_threshold = 1e-4
    compact_order = next(r["order"] for r in report["orders"]
                         if r["max_snr_error"] <= compact_threshold)
    started = perf_counter()
    compact = EmpiricalObserverTimePlan(ctx, compact_order)
    compact_n, compact_i = compact.transform(ctx.gn), compact.transform(ctx.gi)
    compact_setup = perf_counter() - started
    report["compact"] = {
        "order": compact_order, "threshold_max_absolute_snr_error": compact_threshold,
        "plan_setup_seconds": compact_setup,
        "actually_reduces_time_fields": compact_order < 3,
        "note": "Explicit alternative accuracy target, not a pass at the tighter 1e-8 threshold.",
    }
    report["observer_record_max_error_au"] = float(np.max(abs(
        plan.observer((ctx.times-plan.origin)/plan.span) - ctx.observers)))
    report["recorded_position_max_error_arcsec"] = float(np.max(abs(plan.positions(q, ctx.times)-pos)))
    assert report["observer_record_max_error_au"] < 1e-14
    assert report["recorded_position_max_error_arcsec"] < 1e-8

    # Independent full pixel templates at corners and distributed interiors.
    indices = np.unique(np.r_[np.arange(65), np.linspace(0, count-1, 32, dtype=int)])
    pn, pi = ctx.detector.direct(ctx.images, pos[indices])
    assert np.max(abs(pn-dn[indices])) < 1e-8 and np.max(abs(pi-di[indices])) < 1e-8
    report["independent_pixel_queries"] = len(indices)

    # Fresh interior queries are not used to choose the time order.
    holdout = np.random.default_rng(20261029).uniform(plan.lower, plan.upper, (8192, 6))
    hn, hi = streamed(ctx.detector, ctx.images, ctx.positions(holdout))
    evaluator = plan.query_plan(holdout)
    an, ai = evaluator.apply(gn), evaluator.apply(gi)
    report["holdout"] = {"queries": len(holdout), **errors(an, ai, hn, hi)}
    assert report["holdout"]["max_snr_error"] <= threshold
    arrays.update(holdout_queries=holdout, holdout_direct_n=hn, holdout_direct_i=hi,
                  holdout_fast_n=an, holdout_fast_i=ai)
    del evaluator
    evaluator = compact.query_plan(holdout)
    an, ai = evaluator.apply(compact_n), evaluator.apply(compact_i)
    report["compact"]["holdout"] = errors(an, ai, hn, hi)
    assert report["compact"]["holdout"]["max_snr_error"] <= compact_threshold
    arrays.update(holdout_compact_n=an, holdout_compact_i=ai)
    del evaluator

    changed = .7*ctx.images + np.random.default_rng(118).normal(size=ctx.images.shape)*.01
    changed_n, changed_prep = timed_values(lambda: plan.transform(ctx.prepare(changed)), repeats)
    report["changed_image_preparation"] = changed_prep
    cn, ci = streamed(ctx.detector, changed, pos)
    evaluator = plan.query_plan(q)
    an = evaluator.apply(changed_n)
    report["changed_image"] = errors(an, arrays[f"i_{selected}"], cn, ci)
    assert report["changed_image"]["max_snr_error"] <= threshold
    arrays.update(changed_direct_n=cn, changed_fast_n=an)
    del evaluator
    compact_changed, compact_prep = timed_values(
        lambda: compact.transform(ctx.prepare(changed)), repeats)
    evaluator = compact.query_plan(q)
    an = evaluator.apply(compact_changed)
    report["compact"]["changed_image"] = errors(an, arrays[f"i_{compact_order}"], cn, ci)
    assert report["compact"]["changed_image"]["max_snr_error"] <= compact_threshold
    arrays["changed_compact_n"] = an
    del evaluator

    # Repeat all three routes in one campaign, including cached direct pixels.
    for size in [v for v in [256, 1024, 4096, 8192, 16384, 32768, 65536] if v <= count]:
        subset = q[:size]
        started = perf_counter()
        geometry = ctx.positions(subset)
        geometry_seconds = perf_counter() - started
        _, direct = timed_values(lambda: np.array(streamed(ctx.detector, ctx.images, geometry)), repeats)
        started = perf_counter()
        cache = PixelCache(ctx.detector, geometry, backend="dense", max_bytes=3_000_000_000)
        cache_seconds = perf_counter() - started
        _, cached = timed_values(lambda: np.array(cache.apply(changed)), repeats)
        del cache
        row = {"queries": size, "geometry_seconds": geometry_seconds,
               "direct_stream": direct, "direct_cache_setup_seconds": cache_seconds,
               "direct_cached": cached}
        row["first_direct_seconds"] = min(
            geometry_seconds + direct["median_seconds"],
            geometry_seconds + cache_seconds + cached["median_seconds"])
        row["reuse_direct_seconds"] = cached["median_seconds"]
        for method in ["adapter", "pfkt", "compact"]:
            started = perf_counter()
            if method == "adapter":
                evaluator = ctx.query(subset, geometry)[0]
            else:
                evaluator = (plan if method == "pfkt" else compact).query_plan(subset)
            query_seconds = perf_counter() - started
            if method == "adapter":
                query_seconds += geometry_seconds
            fields_n, fields_i = {"adapter": (ctx.gn, ctx.gi), "pfkt": (gn, gi),
                                  "compact": (compact_n, compact_i)}[method]
            _, both = timed_values(lambda: np.array([evaluator.apply(fields_n), evaluator.apply(fields_i)]), repeats)
            if method == "adapter":
                new_field, prep = timed_values(lambda: ctx.prepare(changed), repeats)
            elif method == "pfkt":
                new_field, prep = changed_n, changed_prep
            else:
                new_field, prep = compact_changed, compact_prep
            _, numerator = timed_values(lambda: evaluator.apply(new_field), repeats)
            row[method] = {"query_setup_seconds": query_seconds, "both": both,
                           "changed_preparation": prep, "numerator": numerator}
            row[f"first_{method}_seconds"] = (ctx.setup + query_seconds + both["median_seconds"]
                                            + {"adapter": 0, "pfkt": plan_setup,
                                               "compact": compact_setup}[method])
            row[f"reuse_{method}_seconds"] = prep["median_seconds"] + numerator["median_seconds"]
            del evaluator
        report["measurements"].append(row)
        print(json.dumps({k:v for k,v in row.items() if k.endswith("_seconds") or k=="queries"}), flush=True)

    np.savez_compressed(directory / "scores.npz", **arrays)
    report["artifact_sha256"] = {"scores.npz": hashlib.sha256((directory/"scores.npz").read_bytes()).hexdigest()}
    save(directory / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--queries", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.queries < 256 or args.repeats < 1:
        parser.error("At least 256 queries and one repeat required")
    run(args.directory, args.queries, args.repeats)
