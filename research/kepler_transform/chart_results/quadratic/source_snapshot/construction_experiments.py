"""Reproduce second full-objective attempts, including failures and total costs.

OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.construction_experiments NEW_DIRECTORY
Use --quick for a smoke experiment; its timings are not the large benchmark.
"""

import argparse
import hashlib
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from magictools.orbits import GM
from .atlas import GeometryAtlas
from .butterfly import Butterfly
from .continuous import build_atlas, verify_partition
from .factorization import ARCSEC
from .geometry_certificate import certify_geometry, score_error_bound
from .kernel import KeplerKernel, physical_problem, phase_coordinates
from .local_transform import LocalTransform, pixel_fields, monomials


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def setup(case, frames=8, modes=512, outer_scale=1.):
    original, _, _, data = physical_problem(8, modes, case=case)
    take = np.linspace(0, 15, frames).astype(int)
    radius = 1.02 if case == "close_observer" else 3.
    factor = {"bound": 1., "mixed_energy": np.sqrt(2), "close_observer": 1.8}[case]
    center = np.array([radius, 0., 0., 0., factor*np.sqrt(GM["sun"]/radius), 0.])
    distance = radius-1
    basis = np.zeros((6, 6))
    basis[0, 0], basis[1, 3] = distance*5e-5*outer_scale, 1e-6*outer_scale
    basis[2, 5] = distance*2*ARCSEC
    basis[3, 1] = basis[4, 2] = distance*10*ARCSEC
    basis[5, 4] = distance*2*ARCSEC
    geometry = KeplerKernel(center[None], original.times[take], original.observers[take],
                            original.visits % frames, original.frequencies, center)
    return center, basis, geometry, data


def fresh_kernel(states, geometry, center):
    return KeplerKernel(states, geometry.times, geometry.observers, geometry.visits, geometry.frequencies, center)


def butterfly_experiment(n, case, cap, adapted=True):
    kernel, rows, columns, data = physical_problem(n, n, case=case)
    adaptation = {"seconds": 0., "additional_geometry_pairs": 0}
    if adapted:
        rows, columns, adaptation = phase_coordinates(kernel, rows)
    plan = Butterfly(kernel, rows, columns, rank_cap=cap, leaf_size=1)
    construction_geometry = kernel.geometry_pairs
    start = perf_counter()
    output = plan.apply(data)
    application = perf_counter()-start
    # A fresh geometry cache is required for a fair one-shot baseline.
    start = perf_counter()
    reference = fresh_kernel(kernel.states, kernel, np.mean(kernel.states, axis=0))
    # Keep the same projection basis, rather than changing the physical field.
    reference.normal, reference.east, reference.north = kernel.normal.copy(), kernel.east.copy(), kernel.north.copy()
    truth = reference.nufft(data)
    direct = perf_counter()-start
    rng = np.random.default_rng(33)
    target = rng.normal(size=n)+1j*rng.normal(size=n)
    adjoint = abs(np.vdot(output, target)-np.vdot(data, plan.apply(target, adjoint=True)))
    return {"case": case, "size": n, "adapted": adapted, **plan.report,
            "partition": adaptation, "construction_geometry_pairs": construction_geometry,
            "all_orbit_time_pairs": n*len(kernel.times), "application_seconds": application,
            "fresh_geometry_plus_nufft_seconds": direct,
            "one_shot_speedup": direct/(adaptation["seconds"]+plan.report["construction_seconds"]+application),
            "relative_score_error": float(np.linalg.norm(output-truth)/np.linalg.norm(truth)),
            "adjoint_absolute_error": float(adjoint), "full_objective_complete": False}


def fourier_control(n):
    x, y = np.arange(n)/n, np.arange(n)
    plan = Butterfly(lambda r, c: np.exp(2j*np.pi*x[r, None]*y[c]), x[:, None], y[:, None], rank_cap=16, leaf_size=1)
    data = np.random.default_rng(34).normal(size=n)
    start = perf_counter()
    output = plan.apply(data)
    apply = perf_counter()-start
    start = perf_counter()
    truth = np.fft.ifft(data)*n
    fft = perf_counter()-start
    return {"size": n, **plan.report, "application_seconds": apply, "fft_seconds": fft,
            "relative_score_error": float(np.linalg.norm(output-truth)/np.linalg.norm(truth)),
            "interpretation": "positive complementary-rank control, not a replacement for the FFT"}


def local_experiment(case, n, width=1., certify=True):
    center, basis, geometry, data = setup(case, frames=8)
    inner = basis[3:]*width
    plan = LocalTransform(center, inner, geometry)
    certificate = certify_geometry(plan) if certify else None
    targets = np.random.default_rng(35).uniform(-1, 1, (n, 3))
    start = perf_counter()
    try:
        output = plan.apply(data, targets)
    except ValueError as failure:
        if "phase-volume budget" not in str(failure):
            raise
        # Preserve the failed full-size request. A small quadratic reference
        # diagnoses physical approximation error; it is not relabeled a fast run.
        approximation = plan.quadratic_direct(data, targets)
        truth = fresh_kernel(center+targets @ inner, geometry, center).nufft(data)
        return {"case": case, "inner_width_factor": width, "queries": n,
                "status": "resource_limit", "reason": str(failure),
                "construction": plan.report,
                "quadratic_geometry_relative_score_error": float(np.linalg.norm(approximation-truth)/np.linalg.norm(truth)),
                "full_objective_complete": False}
    apply = perf_counter()-start
    start = perf_counter()
    truth_kernel = fresh_kernel(center+targets @ inner, geometry, center)
    truth = truth_kernel.nufft(data)
    direct = perf_counter()-start
    construction = plan.report["construction_seconds"]
    certification = certificate["construction_seconds"] if certificate else 0.
    return {"case": case, "inner_width_factor": width, "queries": n, **plan.report,
            "application_seconds": apply, "direct_geometry_plus_nufft_seconds": direct,
            "one_shot_speedup_without_certificate": direct/(construction+apply),
            "one_shot_speedup_with_certificate": direct/(construction+certification+apply),
            "maximum_absolute_score_error": float(max(abs(output-truth))),
            "relative_score_error": float(np.linalg.norm(output-truth)/np.linalg.norm(truth)),
            "uniform_phase_tail_bound": plan.phase_tail_bound(data),
            "uniform_geometry_score_bound": score_error_bound(plan, certificate, data) if certificate else None,
            "certificate": certificate, "center": center.tolist(), "inner_basis": inner.tolist(),
            "full_objective_complete": False}


def atlas_experiment(case, n_inner, *, frames=8, outer_scale=1., certify=True):
    center, basis, geometry, data = setup(case, frames=frames, outer_scale=outer_scale)
    atlas = GeometryAtlas(center, basis, geometry)
    certificate = atlas.certify() if certify else None
    outer_points = np.array(list(product([-.7, .7], repeat=3)))
    targets = np.random.default_rng(36).uniform(-1, 1, (n_inner, 3))
    construction, apply, direct, error2, norm2, max_error, tail, geom_bound = 0., 0., 0., 0., 0., 0., 0., 0.
    for outer in outer_points:
        start = perf_counter()
        plan = atlas.slice(outer)
        construction += perf_counter()-start
        start = perf_counter()
        output = plan.apply(data, targets)
        apply += perf_counter()-start
        start = perf_counter()
        truth = fresh_kernel(center+outer @ basis[:3]+targets @ basis[3:], geometry, center).nufft(data)
        direct += perf_counter()-start
        error2 += float(np.linalg.norm(output-truth)**2)
        norm2 += float(np.linalg.norm(truth)**2)
        max_error = max(max_error, float(max(abs(output-truth))))
        tail = max(tail, plan.phase_tail_bound(data))
        if certificate:
            geom_bound = max(geom_bound, score_error_bound(plan, certificate, data))
    total = atlas.report["construction_seconds"]+construction+apply
    certification = certificate["construction_seconds"] if certificate else 0.
    return {"case": case, "outer_width_factor": outer_scale, "queries": len(targets)*len(outer_points),
            "outer_queries": len(outer_points), "inner_queries_per_outer": len(targets),
            "frames": frames, "modes": len(data), "center": center.tolist(), "basis": basis.tolist(),
            "atlas": atlas.report, "slice_construction_seconds": construction, "application_seconds": apply,
            "direct_geometry_plus_nufft_seconds": direct, "one_shot_seconds_without_certificate": total,
            "one_shot_seconds_with_certificate": total+certification,
            "one_shot_speedup_without_certificate": direct/total,
            "one_shot_speedup_with_certificate": direct/(total+certification),
            "relative_score_error": float(np.sqrt(error2/norm2)), "maximum_absolute_score_error": max_error,
            "phase_tail_bound_max_over_queried_outer_slices": tail,
            "uniform_phase_tail_bound_for_entire_6d_cell": atlas.phase_tail_bound(data),
            "geometry_score_bound_for_entire_6d_cell": geom_bound if certificate else None,
            "certificate": certificate,
            "continuous_coverage_note": "geometry and analytic phase-tail bounds cover all 6D points; floating-point phase and NUFFT arithmetic are not formally certified",
            "full_objective_complete": False}


def pixel_experiment(n_inner, output_directory):
    center, basis, geometry, _ = setup("mixed_energy", frames=8)
    rng, size, sigma = np.random.default_rng(37), 17, 1.5
    images = rng.normal(size=(8, size, size))
    variance = rng.uniform(.8, 1.2, images.shape)
    # The draws are adjusted to have the declared, known noise variance.
    images *= np.sqrt(variance)
    masks = rng.random(images.shape) < .03
    targets = rng.uniform(-.3, .3, (n_inner, 3))
    truth_index, flux = min(37, n_inner-1), 100.
    true_state = center+targets[truth_index] @ basis[3:]
    injected = fresh_kernel(true_state[None], geometry, center)
    positions = injected.projected(np.array([0]), np.arange(8))[0]/ARCSEC
    axis = np.arange(-(size//2), size//2+1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    modes = np.column_stack((xx.ravel(), yy.ravel()))*2*np.pi/size
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    psf_hat = np.exp(-.5*sigma**2*np.sum(modes**2, axis=1))*np.prod(np.sinc(modes/(2*np.pi)), axis=1)
    for visit in range(8):
        template = (np.exp(1j*(pixels-positions[visit]) @ modes.T) @ psf_hat/size**2).real.reshape(size, size)
        images[visit] += flux*template
    np.savez_compressed(output_directory/"pixel-inputs.npz", images=images, variance=variance,
                        masks=masks, true_state=true_state, targets=targets, flux=flux)
    start = perf_counter()
    visits, frequencies, numerator, information = pixel_fields(images, variance, sigma, ARCSEC, masks=masks)
    preparation = perf_counter()-start
    fields = KeplerKernel(center[None], geometry.times, geometry.observers, visits, frequencies, center)
    plan = LocalTransform(center, basis[3:], fields)
    start = perf_counter()
    n_score, i_score = plan.apply(numerator, targets).real, plan.apply(information, targets).real
    application = perf_counter()-start
    start = perf_counter()
    reference = fresh_kernel(center+targets @ basis[3:], fields, center)
    n_direct, i_direct = reference.nufft(numerator).real, reference.nufft(information).real
    direct = perf_counter()-start
    if np.any(i_score <= 0) or np.any(i_direct <= 0):
        raise ValueError("Positive information required in this experiment; do not silently clip")
    score, truth = n_score/np.sqrt(i_score), n_direct/np.sqrt(i_direct)
    # Independently recompute selected likelihoods in the original pixel basis.
    checks = np.unique(np.r_[np.arange(min(16, n_inner)), np.argmax(score), truth_index])
    position = reference.projected(checks, np.arange(8))/ARCSEC
    pixel_n, pixel_i = [], []
    weight = np.where(masks, 0., 1/variance)
    for path in position:
        n, i = 0., 0.
        for visit, xy in enumerate(path):
            template = (np.exp(1j*(pixels-xy) @ modes.T) @ psf_hat/size**2).real.reshape(size, size)
            n += float(np.sum(images[visit]*weight[visit]*template))
            i += float(np.sum(weight[visit]*template**2))
        pixel_n.append(n)
        pixel_i.append(i)
    return {"model": "synthetic periodic detector; finite pixel-integrated Gaussian PSF; known independent Gaussian variances and masks",
            "real_survey_data": False, "queries": n_inner, "frames": 8, "pixels_per_frame": size**2,
            "fourier_modes": len(visits), "psf_sigma_arcsec": sigma,
            "fft_preparation_seconds": preparation, "construction": plan.report,
            "application_seconds_both_statistics": application, "direct_seconds_both_statistics": direct,
            "one_shot_speedup": (preparation+direct)/(preparation+plan.report["construction_seconds"]+application),
            "maximum_snr_difference": float(max(abs(score-truth))),
            "same_top_candidate": bool(np.argmax(score) == np.argmax(truth)),
            "top_snr": float(max(score)), "injected_reference_flux": flux,
            "injected_query_snr": float(score[truth_index]),
            "independent_pixel_numerator_max_error": float(max(abs(n_direct[checks]-pixel_n))),
            "independent_pixel_information_max_error": float(max(abs(i_direct[checks]-pixel_i))),
            "full_objective_complete": False}


def inverse_experiment():
    center, basis, geometry, data = setup("bound", frames=2, modes=16)
    plan = LocalTransform(center, basis[3:], geometry)
    rng = np.random.default_rng(38)
    targets = rng.uniform(-1, 1, (64, 3))
    observed = rng.normal(size=64)+1j*rng.normal(size=64)
    matrix = np.exp(1j*targets @ plan.linear.T)*(monomials(targets, plan.powers) @ plan.coefficients)
    expected = np.linalg.lstsq(np.vstack((matrix, .3*np.eye(16))), np.r_[observed, np.zeros(16)], rcond=None)[0]
    recovered, report = plan.inverse(observed, targets, damping=.3, tolerance=1e-11)
    report.update(maximum_difference_from_dense_solution=float(max(abs(recovered-expected))),
                  adjoint_relative_error=float(abs(np.vdot(plan.apply(data, targets), observed)
                    -np.vdot(data, plan.adjoint(observed, targets)))/max(1., abs(np.vdot(matrix @ data, observed)))))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {"full_objective_complete": False, "quick": args.quick, "cases": {}}

    def run(name, function):
        print(f"Starting {name}", flush=True)
        value = function()
        save(args.output/(name+".json"), value)
        results["cases"][name] = value
        print(f"Completed {name}", flush=True)

    run("fourier-control", lambda: fourier_control(256 if args.quick else 4096))
    for case in ["bound", "mixed_energy", "close_observer"]:
        run("butterfly-"+case, lambda case=case: butterfly_experiment(256 if args.quick else 1024, case, 16))
    if not args.quick:
        run("butterfly-rank-repair", lambda: butterfly_experiment(1024, "bound", 32))
        run("butterfly-scaling", lambda: butterfly_experiment(4096, "bound", 16))
    for case in ["bound", "mixed_energy", "close_observer"]:
        run("local-"+case, lambda case=case: local_experiment(case, 256 if args.quick else 32768))
    run("local-broad", lambda: local_experiment("bound", 128 if args.quick else 4096, width=30., certify=False))
    run("atlas-narrow", lambda: atlas_experiment("mixed_energy", 128 if args.quick else 131072, frames=2 if args.quick else 8))
    run("atlas-broad", lambda: atlas_experiment("bound", 128 if args.quick else 2048,
                                               frames=2 if args.quick else 8, outer_scale=500.))
    run("pixels", lambda: pixel_experiment(128 if args.quick else 32768, args.output))
    run("inverse", inverse_experiment)
    center, _, geometry, data = setup("bound", frames=2, modes=16)
    run("constant-atlas-budget", lambda: build_atlas(center, np.full(6, 1e-5), geometry, data,
                                                     tolerance=1e-8, max_cells=8 if args.quick else 64))
    assert verify_partition(results["cases"]["constant-atlas-budget"])
    results["provenance"] = {"python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "finufft", "python-flint"]},
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for directory in [Path(__file__).parent, Path("research/detection_limit")]
                    for path in sorted(directory.glob("*.py"))}}
    save(args.output/"complete.json", results)


if __name__ == "__main__":
    main()
