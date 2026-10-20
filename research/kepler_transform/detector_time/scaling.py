"""Larger-query sweep on retained pixels, with a budgeted direct factor cache.

The synthetic run uses the certified time transform. The real run uses the
detector Fourier adapter at actual reception times, without time compression.
"""

import argparse
import gc
import hashlib
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

from ..compiled_reference import projected_chart
from ..factorization import ARCSEC
from ..temporal.benchmark import source_hashes
from ..temporal.transform import FourierEvaluator, TimePlan
from ..timing.benchmark import measure
from .benchmark import save
from .fields import CommonDetector, l1_upper, sum_bounds, trim_fields
from .pixel_cache import PixelCache
from .plan import DetectorTimePlan, domain_source_box, statistic_intervals


def timed_values(function, repeats):
    """Return the last output and independent timings without extra warmup calls.

    Shared imports/JIT are warmed before the campaign. Retain the first measured
    application too; especially small real-data timings can remain noisy.
    """
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        values = function()
        samples.append(perf_counter() - start)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite measured output")
    return values, {"samples_seconds":samples, "median_seconds":float(np.median(samples)),
                    "min_seconds":min(samples), "max_seconds":max(samples), "loops_per_sample":1}


class BudgetPixelCache:
    """Cache as many complete query PSF factors as fit, recompute the remainder.

    Both information and positions are retained. New image applications compute
    numerator only, including for queries whose factors do not fit. The budget
    includes retained arrays, not transient matrix products or detector inputs.
    """

    def __init__(self, detector, positions, information, max_bytes, batch=4096):
        self.detector = detector
        self.positions = detector.pixel_positions(positions)
        self.information = np.array(information, float, copy=True)
        if (self.information.shape != (len(self.positions),)
                or not np.all(np.isfinite(self.information)) or batch < 1):
            raise ValueError("Finite information per query and positive batch required")
        self.batch = batch
        base = self.positions.nbytes + self.information.nbytes
        if base > max_bytes:
            raise ValueError("Position and information cache exceed budget")
        per_query = detector.frames * detector.n * 2 * 8
        self.cached_queries = min(len(positions), (max_bytes - base) // per_query)
        self.px = np.empty((detector.frames, self.cached_queries, detector.n))
        self.py = np.empty_like(self.px)
        for j in range(detector.frames):
            self.px[j], self.py[j] = self.factors(j, 0, self.cached_queries)
        self.persistent_bytes = base + self.px.nbytes + self.py.nbytes
        assert self.persistent_bytes <= max_bytes

    def factors(self, j, start, stop):
        d = self.detector
        axis = np.arange(d.n) - (d.n - 1) / 2
        x, y = self.positions[start:stop, j, 0, None], self.positions[start:stop, j, 1, None]
        sx, sy = d.sigma[j]
        return (ndtr((axis + .5 - x) / sx) - ndtr((axis - .5 - x) / sx),
                ndtr((axis + .5 - y) / sy) - ndtr((axis - .5 - y) / sy))

    def apply(self, images):
        d = self.detector
        images = np.asarray(images, float)
        if images.shape != d.shape_pixels or not np.all(np.isfinite(images)):
            raise ValueError("Finite images matching detector required")
        n = np.zeros(len(self.positions))
        for j in range(d.frames):
            w, values = d.weight[j], images[j].copy()
            if d.profile and np.sum(w) > 0:
                values -= np.sum(w * values) / np.sum(w)
            weighted = w * values * d.scales[j]
            for start in range(0, self.cached_queries, self.batch):
                stop = min(start + self.batch, self.cached_queries)
                n[start:stop] += np.sum((self.py[j, start:stop] @ weighted)
                                       * self.px[j, start:stop], axis=1)
            for start in range(self.cached_queries, len(n), self.batch):
                stop = min(start + self.batch, len(n))
                px, py = self.factors(j, start, stop)
                n[start:stop] += np.sum((py @ weighted) * px, axis=1)
        return n, self.information.copy()


def streamed(detector, images, positions, batch=4096):
    values = [PixelCache(detector, positions[k:k + batch], backend="stream",
                         max_bytes=3_000_000_000).apply(images)
              for k in range(0, len(positions), batch)]
    return np.concatenate([v[0] for v in values]), np.concatenate([v[1] for v in values])


def synthetic(count, source):
    old = json.loads((source / "results.json").read_text())
    with np.load(source / "inputs.npz") as archive:
        a = {k: archive[k] for k in archive.files}
    rng = np.random.default_rng(20261106)
    q = np.concatenate([a["queries"], rng.uniform(-1, 1, (max(0, count - len(a["queries"])), 6))])[:count]
    # The same original complete near box; old query prefix is unchanged.
    warm = TimePlan("near", [-1e-7, 1e-7])
    warm.positions(q[:1], [-4., 4.])
    started = perf_counter()
    cover = domain_source_box("near")
    d = CommonDetector(a["variances"], a["sigma"], a["matrices"], offsets=a["offsets"],
                       masks=a["masks"], flux_scales=a["flux_scales"],
                       source_half_width=float(np.ceil(max(cover["relative_reference_position_bound_arcsec"]))),
                       max_bytes=4_000_000_000)
    full_n, full_i, report = d.prepare(a["images"])
    tn, i, trim = trim_fields(full_n, full_i, [1e-5, 1e-7])
    ncut, nr = d.choose_numerator_cutoff(a["images"], 1e-5)
    mid = d.side // 2
    n = np.ascontiguousarray(full_n[:, mid-ncut:mid+ncut+1, mid-ncut:mid+ncut+1])
    del full_n, full_i, tn
    gc.collect()
    pn = DetectorTimePlan("near", a["times"], n.shape[1:], d.spacing,
                          tolerance=min(1e-8, 1e-6 / max(1., l1_upper(n))))
    pi = DetectorTimePlan("near", a["times"], i.shape[1:], d.spacing,
                          tolerance=min(1e-8, 1e-8 / max(1., l1_upper(i))))
    gn, gi = pn.transform(n), pi.transform(i)
    setup = perf_counter() - started
    nbound = sum_bounds([nr["numerator_model_error_bound"], pn.score_error_bound(n)])
    ibound = sum_bounds([r["information_model_error_bound"] for r in report["exposures"]] +
                       [trim["discarded_information_l1_bound"], pi.score_error_bound(i)])
    del n, i
    gc.collect()

    def prepare(data):
        n, report = d.prepare_numerator(data, ncut)
        if report["numerator_model_error_bound"] + pn.score_error_bound(n) > 2e-5:
            raise ValueError("New input exceeds existing numerator plan budget")
        return pn.transform(n)

    return SimpleNamespace(
        detector=d, images=a["images"], queries=q, gn=gn, gi=gi, setup=setup,
        prepare=prepare, bounds=[nbound, ibound],
        positions=lambda q: (pn.positions(q, a["times"]) - pn.center_input) / ARCSEC,
        query=lambda q, pos: (pn.query_plan(q, backend="nufft"), pi.query_plan(q, backend="nufft")),
        query_uses_positions=False,
        metadata={"kind": "synthetic", "workload": old["workload"], "source_domain_proof": cover,
                  "time_proof": pn.proof, "information_time_proof": pi.proof,
                  "time_compression_applied": True,
                  "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [source/"inputs.npz", source/"results.json"]}},
    )


def real(count, source):
    from astropy.io import fits
    from astropy.wcs import WCS
    from ..observer_chart import ObserverChart
    from ..real_exposure_experiment import directions, tangent_wcs
    root = Path(__file__).resolve().parents[3]
    meta = json.loads((source / "observations.json").read_text())
    images, variances, masks, worlds = [], [], [], []
    for name in meta["files"]:
        with fits.open(source / name) as hdus:
            images.append(hdus["SCI"].data.astype(float))
            variances.append(hdus["VAR"].data.astype(float))
            masks.append(hdus["MASK"].data != 0)
            worlds.append(WCS(hdus["SCI"].header))
    images, variances, masks = map(np.asarray, (images, variances, masks))
    normal, east, north, _ = tangent_wcs(worlds[0], 4., 4.)
    matrices, offsets = [], []
    for wcs in worlds:
        sky = wcs.all_pix2world([[4,4],[4.5,4],[3.5,4],[4,4.5],[4,3.5]], 0)
        v = directions(sky[:, 0], sky[:, 1])
        xy = np.column_stack([v @ east, v @ north]) / (v @ normal)[:, None] / ARCSEC
        offsets.append(xy[0])
        matrices.append(np.column_stack([xy[1]-xy[2], xy[3]-xy[4]]))
    chart = ObserverChart(meta["observers"][0], meta["observer_velocities"][0],
                          normal, east, north, retarded_epoch=True)
    previous = root / "research/kepler_transform/chart_results/real/queries-and-scores.npz"
    with np.load(previous) as a:
        old_q = a["parameters"]
    # Extend inside the bounding rectangle of the existing real-query domain.
    # No claim of a certified real-ephemeris time-interpolation domain.
    lo, hi = old_q.min(axis=0), old_q.max(axis=0)
    rng = np.random.default_rng(20261106)
    corners = np.array(list(product([0., 1.], repeat=6))) * (hi-lo) + lo
    q = np.r_[(lo+hi)[None]/2, corners, rng.uniform(lo, hi, (max(0, count-65), 6))][:count]
    times = np.array(meta["mjd_tdb"]) - meta["mjd_tdb"][0]
    geometry = SimpleNamespace(times=times, observers=np.array(meta["observers"]),
                               normal=np.tile(normal, (len(times),1)),
                               east=np.tile(east, (len(times),1)),
                               north=np.tile(north, (len(times),1)))
    positions = lambda q: projected_chart(chart, q, geometry) / ARCSEC
    positions(q[:1])  # Shared JIT startup excluded for both routes.
    started = perf_counter()
    d = CommonDetector(variances, meta["sigma_pixels"], matrices, offsets=offsets, masks=masks,
                       flux_scales=meta["flux_scales"], source_half_width=4.)
    fn, fi, report = d.prepare(images)
    n, i, trim = trim_fields(fn, fi, [1e-8, 1e-10])
    def prepare(data):
        numerator, _, new_report = d.prepare(data, include_information=False)
        mid, half = d.side//2, n.shape[1]//2
        cut = np.ascontiguousarray(numerator[:, mid-half:mid+half+1, mid-half:mid+half+1])
        numerator[:, mid-half:mid+half+1, mid-half:mid+half+1] = 0
        budget = sum_bounds([r["numerator_model_error_bound"] for r in new_report["exposures"]]
                            + [l1_upper(numerator)])
        if budget > 1e-4:
            raise ValueError("Changed real image exceeds the retained mode budget")
        return cut

    nbound = sum_bounds([r["numerator_model_error_bound"] for r in report["exposures"]] +
                        [trim["discarded_numerator_l1_bound"]])
    ibound = sum_bounds([r["information_model_error_bound"] for r in report["exposures"]] +
                        [trim["discarded_information_l1_bound"]])
    setup = perf_counter() - started

    def query(q, pos):
        evaluator = FourierEvaluator(pos, n.shape[1:], [d.spacing]*2, backend="nufft")
        return evaluator, evaluator

    return SimpleNamespace(
        detector=d, images=images, queries=q, gn=n, gi=i, setup=setup, prepare=prepare,
        bounds=[nbound, ibound], positions=positions, query_uses_positions=True,
        query=query,
        worlds=worlds, axes=np.array([normal,east,north]),
        chart=chart, times=times, observers=np.array(meta["observers"]),
        metadata={"kind": "real", "time_compression_applied": False,
                  "workload": {"frames":len(images), "image_side":images.shape[1],
                               "span_days":float(np.ptp(times)), "real_survey_stack":True,
                               "query_lower":lo.tolist(), "query_upper":hi.tolist()},
                  "model": "Original real ZTF pixels, actual stored ephemerides, affine WCS, Gaussian PSFs",
                  "input_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [source/"observations.json", previous]+[source/f for f in meta["files"]]}},
    )


def run(directory, kind, source, counts, repeats, cache_bytes):
    directory.mkdir(parents=True, exist_ok=False)
    ctx = (synthetic if kind == "synthetic" else real)(max(counts), source)
    result = {**ctx.metadata, "source_sha256": source_hashes(),
              "setup_seconds":ctx.setup, "bounds":ctx.bounds,
              "cache_budget_bytes":cache_bytes, "batch":4096, "repeats":repeats,
              "thread_limits": {k:os.environ.get(k) for k in
                                ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"]},
              "measurements":[], "notes":[
                  "First-use totals include preparation, query geometry and both output statistics.",
                  "Reuse keeps geometry and information fixed for both methods.",
                  "Direct partial cache includes positions, information and retained PSF factors.",
                  "Cache budget excludes temporary arrays and shared detector inputs; it is not total RSS.",
                  "Full independent pixel template checks use a recorded subset; all queries use separable direct checks.",
                  "Timing medians are measured locally; imports, JIT, input loading and validation excluded.",
              ]}
    prep = measure(lambda:ctx.prepare(ctx.images), repeats)
    result["changed_image_preparation"] = prep
    np.savez_compressed(directory/"queries.npz", queries=ctx.queries)
    outputs = {}
    for count in counts:
        q = ctx.queries[:count]
        print(json.dumps({"phase":"query_start","kind":kind,"queries":count}), flush=True)
        start = perf_counter()
        pos = ctx.positions(q)
        geometry_seconds = perf_counter()-start
        (dn, di), direct = timed_values(lambda: np.array(streamed(ctx.detector,ctx.images,pos)), repeats)
        # Full-template independent comparison at corners and distributed interiors.
        indices = np.unique(np.r_[np.arange(min(65,count)), np.linspace(0,count-1,32,dtype=int)])
        check_n, check_i = ctx.detector.direct(ctx.images, pos[indices])
        assert max(np.max(abs(check_n-dn[indices])),np.max(abs(check_i-di[indices]))) < 1e-8
        start = perf_counter()
        fn, fi = ctx.query(q,pos)
        fast_setup = perf_counter()-start + (geometry_seconds if ctx.query_uses_positions else 0.)
        (an, ai), both = timed_values(lambda:np.array([fn.apply(ctx.gn),fi.apply(ctx.gi)]), repeats)
        numerator = measure(lambda:fn.apply(ctx.gn), repeats)
        errors = [float(np.max(abs(an-dn))), float(np.max(abs(ai-di)))]
        assert errors[0] <= ctx.bounds[0]+1e-8 and errors[1] <= ctx.bounds[1]+1e-9, errors
        interval = statistic_intervals(an.real, ai.real, *ctx.bounds)
        ok = interval["positive_information_resolved"]
        snr = dn[ok]/np.sqrt(di[ok])
        assert np.all(snr >= interval["snr_interval"][ok,0]-1e-8)
        assert np.all(snr <= interval["snr_interval"][ok,1]+1e-8)
        # Release NUFFT native workspaces before allocating direct caches.
        del fn, fi
        gc.collect()
        start = perf_counter()
        cache = BudgetPixelCache(ctx.detector,pos,di,cache_bytes)
        cache_setup = perf_counter()-start
        (cn,ci), cached = timed_values(lambda:np.array(cache.apply(ctx.images)),repeats)
        assert max(np.max(abs(cn-dn)),np.max(abs(ci-di))) < 1e-8
        cached_count, cached_bytes = cache.cached_queries, cache.persistent_bytes
        del cache
        gc.collect()
        dense_result = None
        # Test a full response cache too when it fits the same explicit budget.
        if count * ctx.detector.frames * ctx.detector.n**2 * 8 + count*8 <= cache_bytes:
            start = perf_counter()
            cache = PixelCache(ctx.detector,pos,backend="dense",max_bytes=cache_bytes)
            dense_setup = perf_counter()-start
            (cn,ci), dense_timing = timed_values(lambda:np.array(cache.apply(ctx.images)),repeats)
            assert max(np.max(abs(cn-dn)),np.max(abs(ci-di))) < 1e-8
            dense_result = {"setup_seconds":dense_setup,"application":dense_timing,
                            "persistent_bytes":cache.persistent_bytes}
            del cache
            gc.collect()
        first_options = {"stream":geometry_seconds+direct["median_seconds"],
                         "partial":geometry_seconds+direct["median_seconds"]+cache_setup+cached["median_seconds"]}
        reuse_options = {"partial":cached["median_seconds"]}
        if dense_result:
            first_options["dense"] = geometry_seconds+dense_setup+dense_timing["median_seconds"]
            reuse_options["dense"] = dense_timing["median_seconds"]
        row = {"queries":count,"geometry_seconds":geometry_seconds,
               "direct_stream":direct,"fast_query_setup_seconds":fast_setup,
               "fast_both":both,"fast_numerator":numerator,
               "partial_cache":{"setup_seconds":cache_setup,"application":cached,
                                "cached_queries":cached_count,"persistent_bytes":cached_bytes},
               "dense_cache":dense_result,"max_errors":errors,
               "independent_pixel_indices":indices.tolist(),
               "max_snr_error_resolved":float(np.max(abs(an.real[ok]/np.sqrt(ai.real[ok])-snr))),
               "unresolved":int(np.sum(~ok)),
               "comparison":{
                   "first_direct_seconds":min(first_options.values()),
                   "first_compressed_seconds":ctx.setup+fast_setup+both["median_seconds"],
                   "first_direct_method":min(first_options,key=first_options.get),
                   "new_image_direct_seconds":min(reuse_options.values()),
                   "new_image_compressed_seconds":prep["median_seconds"]+numerator["median_seconds"],
                   "new_image_direct_method":min(reuse_options,key=reuse_options.get)}}
        result["measurements"].append(row)
        outputs[f"fast_n_{count}"],outputs[f"fast_i_{count}"] = an,ai
        outputs[f"direct_n_{count}"],outputs[f"direct_i_{count}"] = dn,di
        print(json.dumps({"queries":count,**row["comparison"],"errors":errors}),flush=True)
        save(directory/"results.json",result)
        del pos
        gc.collect()
    # Validate reuse with changed image values, not a stored answer vector.
    changed = .7*ctx.images + np.random.default_rng(118).normal(size=ctx.images.shape)*.01
    changed_field = ctx.prepare(changed)
    q = ctx.queries[np.unique(np.r_[np.arange(65),np.linspace(0,len(ctx.queries)-1,32,dtype=int)])]
    pos = ctx.positions(q)
    fn, fi = ctx.query(q,pos)
    actual = fn.apply(changed_field)
    expected = ctx.detector.direct(changed,pos)[0]
    result["changed_image_max_numerator_error"] = float(max(abs(actual-expected)))
    assert result["changed_image_max_numerator_error"] < 1e-4
    if kind == "real":
        from ..universal import apparent
        from ..detector_fields import direct_pixel_statistics
        emitted,_ = apparent(ctx.chart.states(q),ctx.times,ctx.observers)
        delta = emitted[...,:3]-ctx.observers
        independent_pos = np.stack([delta@ctx.axes[1],delta@ctx.axes[2]],axis=-1)/(delta@ctx.axes[0])[...,None]/ARCSEC
        result["sampled_compiled_vs_python_position_error_arcsec"] = float(np.max(abs(pos-independent_pos)))
        ra=np.rad2deg(np.arctan2(delta[...,1],delta[...,0]))%360
        dec=np.rad2deg(np.arctan2(delta[...,2],np.hypot(delta[...,0],delta[...,1])))
        pixels=np.empty_like(pos)
        for j,w in enumerate(ctx.worlds):
            x,y=w.all_world2pix(ra[:,j],dec[:,j],0)
            pixels[:,j]=np.column_stack([x-4,y-4])
        tn,ti=direct_pixel_statistics(ctx.images,ctx.detector.variances,ctx.detector.sigma,pixels,
                                      masks=ctx.detector.weight==0,profile_background=True,
                                      flux_scales=ctx.detector.scales)
        an,ai=ctx.detector.direct(ctx.images,pos)
        result["sampled_full_tpv_snr_error"] = float(np.max(abs(an/np.sqrt(ai)-tn/np.sqrt(ti))))
    np.savez_compressed(directory/"scores.npz",**outputs)
    result["artifact_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in directory.iterdir() if p.name!="results.json"}
    save(directory/"results.json",result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--kind",choices=["synthetic","real"],default="synthetic")
    parser.add_argument("--source",type=Path)
    parser.add_argument("--counts",type=int,nargs="+",default=[4096,8192,16384,32768,65536])
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--cache-bytes",type=int,default=3_000_000_000)
    args=parser.parse_args()
    if min(args.counts)<65 or args.repeats<1 or args.cache_bytes<1:
        parser.error("At least 65 queries and positive repeat/cache budgets required")
    source=args.source or (Path(__file__).parent/"results/dense-1024" if args.kind=="synthetic"
                            else Path(__file__).parents[1]/"chart_results/real_inputs")
    run(args.directory,args.kind,source,sorted(set(args.counts)),args.repeats,args.cache_bytes)
