"""Reproducible forward/cascade and real-pixel segment field experiments."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from flint import ctx

from .chart_experiments import setup
from .compiled_reference import projected
from .factorization import ARCSEC
from .reception_flow import CircularObserver, PolynomialObserver, reception_cover
from .segment_experiments import geometry
from .segment_field import SegmentTransform
from .segment_field_certificate import certify


def forward(case="near", frames=8, orders=(3, 3, 3, 3, 3, 3),
            backend="tensor", frequency=.03, adaptive=False):
    old, oc, ow, original_geometry, _ = setup(case)
    observer = CircularObserver()
    center, widths = reception_cover(old, oc, ow)
    times = np.linspace(original_geometry.times[0], original_geometry.times[-1], frames)
    rng = np.random.default_rng(20261029)
    frequencies = rng.uniform(-frequency, frequency, (frames, 3, 2))/ARCSEC
    data = rng.normal(size=frames*3)+1j*rng.normal(size=frames*3)
    parameters = oc+rng.uniform(-.999, .999, (256, 6))*ow
    states = old.states(parameters)
    queries = observer.chart(0.).parameters(states)
    geom = geometry(times, observer.axes)
    half_span = max(abs(times))

    def order_policy(c, w, h, leaf):
        if leaf:
            return orders[3], orders[4]
        if not adaptive:
            return orders
        # Explicit heuristic, not a derivative/accuracy certificate.
        levels = max(0., np.log2(half_span/max(h, np.finfo(float).tiny)))
        return tuple(max(2, int(np.ceil(p-decay*levels)))
                     for p, decay in zip(orders, [2, 2, 1, 0, 0, 1]))
    warm = perf_counter()
    projected(states[:1], geom)
    warm = perf_counter()-warm
    start = perf_counter()
    operator = SegmentTransform(observer, center, widths, times, frequencies,
        order=order_policy,
        max_nodes=1_500_000, max_entries=60_000_000, transfer_backend=backend,
        cascade_padding_rate=.5, exact_leaves=True, root_epoch=0.)
    construction = perf_counter()-start
    start = perf_counter()
    field = operator.transform(data)
    apply = perf_counter()-start
    start = perf_counter()
    actual = field.evaluate(queries)
    query_seconds = perf_counter()-start

    def direct(q):
        s = observer.chart(0.).states(q)
        positions = projected(s, geom)
        return np.einsum("ntm,tm->n",
            np.exp(1j*np.einsum("nti,tmi->ntm", positions, frequencies)), data.reshape(frames, 3))

    start = perf_counter()
    expected = direct(queries)
    direct_queries = perf_counter()-start
    # Diagnostic only: these direct node sums are NEVER used by construction.
    start = perf_counter()
    nodal_expected = direct(operator.root.grid.points())
    direct_root = perf_counter()-start
    exact_node_interpolant = operator.root.grid.matrix(queries) @ nodal_expected
    norm = float(sum(abs(data)))
    return {"case": case, "frames": frames, "orders": list(orders), "backend": backend,
            "adaptive_order_heuristic": adaptive,
            "frequency_bound_inverse_arcsec": frequency, "synthetic_fourier_data": True,
            "original_domain_center": oc.tolist(), "original_domain_widths": ow.tolist(),
            "original_time_endpoints": [float(times[0]), float(times[-1])],
            "all_six_coordinates_vary": True,
            "queries": len(queries),
            "error_per_source_l1": float(np.max(abs(actual-expected))/norm),
            "root_nodal_error_per_source_l1": float(np.max(abs(field.values-nodal_expected))/norm),
            "root_interpolation_only_error_per_source_l1":
                float(np.max(abs(exact_node_interpolant-expected))/norm),
            "construction_seconds": construction, "apply_seconds": apply,
            "query_seconds": query_seconds, "total_seconds": construction+apply+query_seconds,
            "compiled_reference_warmup_seconds": warm,
            "direct_query_seconds": direct_queries, "direct_root_node_seconds": direct_root,
            "total_speedup_against_direct_root_plus_queries":
                (direct_root+direct_queries)/(construction+apply+query_seconds),
            "graph": operator.stats, "full_objective_complete": False}


def real_pixels(angular_order=10, backend="cascade", hidden_order=2, rate_order=3):
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.special import ndtr
    from .detector_fields import direct_pixel_statistics
    from .observer_chart import ObserverChart
    from .universal import apparent

    directory = Path(__file__).parent/"chart_results"
    manifest = directory/"real_inputs/observations.json"
    prior = directory/"real/result.json"
    metadata = json.loads(manifest.read_text())
    previous = json.loads(prior.read_text())["cover"]
    old = ObserverChart(**previous["chart"])
    images, variances, masks, wcs_objects = [], [], [], []
    files = [manifest, prior]
    for filename in metadata["files"]:
        path = manifest.parent/filename
        files.append(path)
        with fits.open(path) as hdus:
            images.append(hdus["SCI"].data.astype(float))
            variances.append(hdus["VAR"].data.astype(float))
            masks.append(hdus["MASK"].data != 0)
            wcs_objects.append(WCS(hdus["SCI"].header))
    images, variances, masks = map(np.asarray, [images, variances, masks])
    times = np.asarray(metadata["mjd_tdb"])-metadata["mjd_tdb"][0]
    O, V = np.asarray(metadata["observers"]), np.asarray(metadata["observer_velocities"])
    span = times[-1]
    u = times/span
    powers = np.arange(2, 6)
    matrix = np.vstack([u[1]**powers, powers*u[1]**(powers-1),
                        u[2]**powers, powers*u[2]**(powers-1)])
    target = np.array([O[1]-O[0]-times[1]*V[0], span*(V[1]-V[0]),
                       O[2]-O[0]-times[2]*V[0], span*(V[2]-V[0])])
    coefficients = np.vstack([O[0], V[0], np.linalg.solve(matrix, target)/span**powers[:, None]])
    observer = PolynomialObserver(coefficients, axes=old.axes)
    center, widths = reception_cover(old, previous["center"], previous["widths"])
    weight = np.where(masks, 0., 1/variances)
    sigma = np.asarray(metadata["sigma_pixels"])
    scales = np.asarray(metadata["flux_scales"])
    size = images.shape[1]
    axis = np.arange(size)-(size-1)/2

    def pixels(direction, visit):
        ra = np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0])) % 360
        dec = np.rad2deg(np.arctan2(direction[:, 2], np.hypot(direction[:, 0], direction[:, 1])))
        x, y = wcs_objects[visit].all_world2pix(ra, dec, 0)
        return np.column_stack((x-(size-1)/2, y-(size-1)/2))

    def kernel(visit, positions):
        direction = old.axes[0]+positions[:, :1]*old.axes[1]+positions[:, 1:]*old.axes[2]
        xy = pixels(direction, visit)
        x, y = xy[:, 0, None], xy[:, 1, None]
        px = ndtr((axis+.5-x)/sigma[visit, 0])-ndtr((axis-.5-x)/sigma[visit, 0])
        py = ndtr((axis+.5-y)/sigma[visit, 1])-ndtr((axis-.5-y)/sigma[visit, 1])
        template = (py[:, :, None]*px[:, None, :]*scales[visit]).reshape(len(positions), -1)
        weights = weight[visit].ravel()
        template -= (template @ weights/sum(weights))[:, None]
        return np.column_stack((template, template**2))

    start = perf_counter()
    operator = SegmentTransform(observer, center, widths, times,
        source_counts=[2*size*size]*len(times), leaf_kernel_builder=kernel,
        order=lambda c, w, h, leaf: ((angular_order,)*2 if leaf else
                                    (hidden_order, hidden_order, rate_order,
                                     angular_order, angular_order, rate_order)),
        transfer_backend=backend, exact_leaves=True, root_epoch=0.,
        max_nodes=100_000, max_entries=40_000_000)
    construction = perf_counter()-start
    numerator_input = np.concatenate([np.r_[(weight[j]*images[j]).ravel(), np.zeros(size*size)]
                                      for j in range(len(times))])
    information_input = np.concatenate([np.r_[np.zeros(size*size), weight[j].ravel()]
                                        for j in range(len(times))])
    start = perf_counter()
    numerator, information = operator.transform(numerator_input), operator.transform(information_input)
    apply = perf_counter()-start
    rng = np.random.default_rng(20261030)
    qold = np.asarray(previous["center"])+rng.uniform(-.99, .99, (128, 6))*previous["widths"]
    states = old.states(qold)
    queries = observer.chart(0.).parameters(states)
    start = perf_counter()
    actual = numerator.evaluate(queries).real, information.evaluate(queries).real
    query_time = perf_counter()-start
    emitted, _ = apparent(states, times, O)
    positions = np.stack([pixels(emitted[:, j, :3]-O[j], j) for j in range(len(times))], axis=1)
    expected = direct_pixel_statistics(images, variances, sigma, positions, masks=masks,
                                      profile_background=True, flux_scales=scales)
    model_observers = np.array([observer.chart(t).observer for t in times])
    return {"real_survey_data": True, "uses_full_recorded_TPV_WCS": True,
            "exposures": len(times), "pixels_per_exposure": size*size, "queries": len(queries),
            "backend": backend, "angular_order": angular_order,
            "hidden_order": hidden_order, "rate_order": rate_order,
            "numerator_max_absolute_error": float(np.max(abs(actual[0]-expected[0]))),
            "information_max_absolute_error": float(np.max(abs(actual[1]-expected[1]))),
            "numerator_relative_sup_error": float(np.max(abs(actual[0]-expected[0]))/max(abs(expected[0]))),
            "information_relative_sup_error": float(np.max(abs(actual[1]-expected[1]))/max(abs(expected[1]))),
            "minimum_interpolated_information": float(np.min(actual[1])),
            "observer_model": "quintic Hermite through recorded positions and velocities",
            "observer_at_exposure_max_absolute_error_au": float(np.max(abs(model_observers-O))),
            "between_exposure_ephemeris_accuracy_certified": False,
            "uniform_detector_interpolation_error_certified": False,
            "same_declared_Gaussian_PSF_and_variance_approximations_as_prior_real_experiment": True,
            "construction_seconds": construction, "two_fields_apply_seconds": apply,
            "two_fields_query_seconds": query_time, "graph": operator.stats,
            "input_sha256": {str(p.relative_to(Path(__file__).parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in files},
            "full_objective_complete": False}


def run(output, mode):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    ctx.prec = 100
    sources = sorted(Path(__file__).parent.glob("*.py"))+[Path(__file__).parents[2]/"src/magictools/orbits.py"]
    source_hashes = {str(p.relative_to(Path(__file__).parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sources}
    results = {"schema_version": 1, "full_objective_complete": False, "experiments": [],
               "python": platform.python_version(), "numpy": np.__version__, "arb_precision_bits": ctx.prec}
    jobs = []
    if mode in ["all", "forward"]:
        jobs += [(f"{case}-tensor", lambda case=case: forward(case, 8, (3,)*6))
                 for case in ["near", "distant", "stress"]]
        jobs += [(f"near-cascade-{p}", lambda p=p: forward("near", 8, (4, 4, p, 10, 10, p), "cascade"))
                 for p in [4, 6]]
        jobs += [("near-physical-frequency", lambda: forward("near", frequency=1.)),
                 ("near-cascade-adaptive", lambda: forward("near", 32, (4, 4, 6, 10, 10, 6),
                                                           "cascade", adaptive=True))]
    if mode in ["all", "real"]:
        jobs += [(f"real-{p}", lambda p=p: real_pixels(p)) for p in [6, 10, 14]]
        jobs += [("real-18-refined", lambda: real_pixels(18, hidden_order=3, rate_order=4))]
    for name, function in jobs:
        started = perf_counter()
        try:
            result = {"name": name, "status": "ran", "result": function()}
        except (ValueError, MemoryError) as exc:
            result = {"name": name, "status": "refused", "reason": str(exc),
                      "elapsed_seconds": perf_counter()-started}
        results["experiments"].append(result)
        (output/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
        print(name, result["status"], "seconds", perf_counter()-started, flush=True)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != source_hashes[str(p.relative_to(Path(__file__).parents[2]))]
           for p in sources):
        raise RuntimeError("Research source changed during the campaign; rerun with stable sources")
    results["source_sha256"] = source_hashes
    (output/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["all", "real", "forward"], default="all")
    args = parser.parse_args()
    run(args.output, args.mode)


if __name__ == "__main__":
    main()
