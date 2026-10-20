"""Reproduce the full-objective attempts; success is not inferred from speed alone.

Run from the repository root:
OPENBLAS_NUM_THREADS=1 python -m research.kepler_transform.experiments new.json
"""

import argparse
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from magictools.orbits import GM
from .factorization import ARCSEC, global_rank, hierarchical_probe, observer_path
from .rotation import (RotationTransform, DirectionalRotationTransform, evaluate_sky,
                       evaluate_directional, real_coefficients, rotations, reproject)
from .universal import propagate, apparent


def independent_dynamics_check():
    mu, r = GM["sun"], 1.
    rows = []
    for label, factor in [("ellipse", .9), ("near_parabolic_bound", np.sqrt(2)*(1-1e-9)),
                          ("parabola", np.sqrt(2)), ("near_parabolic_unbound", np.sqrt(2)*(1+1e-9)),
                          ("hyperbola", 2.)]:
        initial = np.array([r, 0, 0, 0, factor*np.sqrt(mu/r), 0])
        max_position, max_velocity = 0., 0.
        for sign in (-1, 1):
            times = sign*np.linspace(0, 100, 65)
            def derivative(_, state):
                return np.r_[state[3:], -mu*state[:3]/np.linalg.norm(state[:3])**3]
            ode = solve_ivp(derivative, [0, times[-1]], initial, t_eval=times,
                            method="DOP853", rtol=2e-13, atol=2e-14)
            if not ode.success:
                raise ValueError(ode.message)
            reference = propagate(initial[None], times)[0]
            max_position = max(max_position, float(np.max(np.linalg.norm(reference[:, :3]-ode.y[:3].T, axis=1))))
            max_velocity = max(max_velocity, float(np.max(np.linalg.norm(reference[:, 3:]-ode.y[3:].T, axis=1))))
        rows.append({"conic": label, "max_position_error_au": max_position,
                     "max_velocity_error_au_per_day": max_velocity})
    return rows


def rotation_experiment(seed=2026, degree=6, frames=32):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(-120, 120, frames))
    rows = []
    for label, factor in [("ellipse", .9), ("parabola", np.sqrt(2)), ("hyperbola", 1.8)]:
        initial = np.array([[2., 0, 0, 0, factor*np.sqrt(GM["sun"]/2), 0]])
        # A central observer permits an orientation-independent retarded time.
        emitted, _ = apparent(initial, times, np.zeros((frames, 3)))
        phases = np.arctan2(emitted[0, :, 1], emitted[0, :, 0])
        coeff = real_coefficients(rng, frames, degree)
        plan = RotationTransform(phases, degree, "nufft")
        start = perf_counter()
        transformed = plan.forward(coeff)
        transform_seconds = perf_counter()-start
        unit = emitted[0, :, :3]/np.linalg.norm(emitted[0, :, :3], axis=1)[:, None]
        directions = np.einsum("rij,tj->rti", plan.grid_rotations().reshape(-1, 3, 3), unit)
        start = perf_counter()
        direct = evaluate_sky(coeff, directions)
        reference_seconds = perf_counter()-start
        error = np.max(np.abs(transformed.ravel()-direct))
        target = rng.normal(size=transformed.shape)+1j*rng.normal(size=transformed.shape)
        left, right = np.vdot(transformed, target), np.vdot(coeff, plan.adjoint(target))
        recovered = plan.inverse_wigner(transformed)
        expected = plan.coefficients(coeff)
        inverse_error = max(np.max(np.abs(a-b)) for a, b in zip(recovered, expected))
        rows.append({"conic": label, "degree": degree, "frames": frames,
                     "orientations": transformed.size, "max_absolute_score_error": float(error),
                     "relative_score_error": float(error/max(1., np.max(np.abs(direct)))),
                     "relative_adjoint_error": float(abs(left-right)/max(1., abs(left), abs(right))),
                     "inverse_wigner_coefficient_error": float(inverse_error),
                     "construction_seconds": plan.construction_seconds, "transform_seconds": transform_seconds,
                     "direct_harmonic_reference_seconds": reference_seconds,
                     "timing_interpretation": "harmonic input only; excludes real-image SHT/reprojection; reference is not an optimized image-search baseline"})
    return rows


def moving_observer_experiment(seed=2026):
    rng = np.random.default_rng(seed)
    times = np.linspace(-30, 30, 24)
    observers = observer_path(times)
    coeff = real_coefficients(rng, len(times), 3)
    a, g = rng.uniform(-np.pi, np.pi, (2, 96))
    b = np.arccos(rng.uniform(-1, 1, 96))
    rotation = rotations(a, b, g)
    rows = []
    for radius in [30., 3., 1.02]:
        initial = np.array([[radius, 0, 0, 0, np.sqrt(GM["sun"]/radius), 0]])
        state = propagate(initial, times)[0]
        radii = np.linalg.norm(state[:, :3], axis=1)
        phases = np.arctan2(state[:, 1], state[:, 0])
        directions = np.einsum("rij,tj->rti", rotation, state[:, :3])-observers
        direct = evaluate_sky(coeff, directions)
        for degree in [4, 8, 12]:
            projected = reproject(coeff, radii, observers, degree)
            unit = np.column_stack((np.cos(phases), np.sin(phases), np.zeros(len(phases))))
            approximation = evaluate_sky(projected, np.einsum("rij,tj->rti", rotation, unit))
            # This is the image-reprojection error independently of Wigner/FFT.
            error = np.max(np.abs(approximation-direct))
            rows.append({"radius_au": radius, "input_degree": 3, "reprojected_degree": degree,
                         "max_absolute_score_error": float(error),
                         "relative_score_error": float(error/max(1., np.max(np.abs(direct)))),
                         "passes_1e_minus_8_target": bool(error <= 1e-8*max(1., np.max(np.abs(direct)))),
                         "includes_light_time": False})
    return rows


def light_time_experiment(seed=2026):
    rng = np.random.default_rng(seed)
    times = np.linspace(-40, 40, 32)
    observers = observer_path(times)
    a, g = rng.uniform(-np.pi, np.pi, (2, 128))
    b = np.arccos(rng.uniform(-1, 1, 128))
    rotation = rotations(a, b, g)
    rows = []
    for label, radius, factor in [("distant_bound", 40., 1.), ("main_belt_bound", 3., 1.),
                                  ("near_parabolic", 1.2, np.sqrt(2)), ("hyperbolic", 1.2, 2.)]:
        state = np.array([[radius, 0, 0, 0, factor*np.sqrt(GM["sun"]/radius), 0.]])
        central, _ = apparent(state, times, np.zeros_like(observers))
        approximate_position = np.einsum("rij,tj->rti", rotation, central[0, :, :3])
        rotated_states = np.concatenate((np.einsum("rij,j->ri", rotation, state[0, :3]),
                                         np.einsum("rij,j->ri", rotation, state[0, 3:])), axis=1)
        exact, _ = apparent(rotated_states, times, observers)
        approximate_direction, exact_direction = approximate_position-observers, exact[..., :3]-observers
        approximate_direction /= np.linalg.norm(approximate_direction, axis=2)[..., None]
        exact_direction /= np.linalg.norm(exact_direction, axis=2)[..., None]
        error = np.arctan2(np.linalg.norm(np.cross(approximate_direction, exact_direction), axis=2),
                          np.sum(approximate_direction*exact_direction, axis=2))/ARCSEC
        # Matched-filter response ratio for equal-weight Gaussian PSFs in the
        # small-angle approximation, along the same physical orbit. No refit.
        response = np.mean(np.exp(-error**2/4), axis=1)
        rows.append({"conic": label, "orientations": len(rotation), "frames": len(times),
                     "max_direction_error_arcsec": float(np.max(error)),
                     "median_direction_error_arcsec": float(np.median(error)),
                     "minimum_fixed_orbit_snr_response_at_one_arcsec_sigma": float(np.min(response)),
                     "interpretation": "common central-observer emission times vs orientation-dependent light time; not an impossibility proof for other transforms"})
    return rows


def bandwidth_experiment(sigma_arcsec=1.):
    sigma = sigma_arcsec*ARCSEC
    ell = np.arange(int(np.ceil(6/sigma))+1, dtype=float)
    energy = (2*ell+1)*np.exp(-sigma**2*ell*(ell+1))
    cumulative = np.cumsum(energy)/energy.sum()
    degree99 = int(np.searchsorted(cumulative, .99))
    orientation_values = (degree99+1)*(2*degree99+1)**2
    return {"model": "spherical heat-kernel PSF, coefficient decay exp(-sigma^2*l*(l+1)/2)",
            "sigma_arcsec": sigma_arcsec, "degree_for_99_percent_template_energy": degree99,
            "snr_retention_at_degree_12": float(np.sqrt(cumulative[12])),
            "dense_orientation_grid_values_at_99_percent": orientation_values,
            "float64_grid_bytes": 8*orientation_values,
            "qualification": "sum truncated at l=ceil(6/sigma); tiny omitted tail; dense global representation only, not a lower bound on every sparse/local algorithm"}


def directional_identity(seed=2026):
    rng = np.random.default_rng(seed)
    times, degree, spin = 13, 5, 2
    phases = rng.uniform(-np.pi, np.pi, times)
    data = rng.normal(size=(times, (degree+1)**2, 2*spin+1))+1j*rng.normal(size=(times, (degree+1)**2, 2*spin+1))
    for s in range(-spin, spin+1):
        data[:, :s*s, s+spin] = 0
    plan = DirectionalRotationTransform(phases, degree, spin, "nufft")
    output = plan.forward(data)
    reference_rotations = plan.grid_rotations().reshape(-1, 3, 3)
    local = reference_rotations[:, None] @ rotations(phases, np.pi/2, 0)[None]
    expected = evaluate_directional(data, local)
    target = rng.normal(size=output.shape)+1j*rng.normal(size=output.shape)
    left, right = np.vdot(output, target), np.vdot(data, plan.adjoint(target))
    return {"degree": degree, "spin_limit": spin, "frames": times,
            "max_score_error": float(np.max(np.abs(output.ravel()-expected))),
            "relative_adjoint_error": float(abs(left-right)/max(1., abs(left), abs(right))),
            "scope": "algebraic directional composition identity; input response coefficients already supplied"}


def directional_physical_experiment(seed=2026):
    """Follow through the failed scalar-light-time model with a directional one.

    Full light-time geometry is used to build the local response harmonics.
    The expensive construction is counted explicitly and fails the fast-
    construction requirement; it must not disappear into a preprocessing label.
    """
    rng = np.random.default_rng(seed)
    times, radius, degree = np.linspace(-40, 40, 8), 10., 10
    observers = observer_path(times)
    state = np.array([[radius, 0., 0., 0., np.sqrt(GM["sun"]/radius), 0.]])
    central, emissions = apparent(state, times, np.zeros_like(observers))
    phases = np.arctan2(central[0, :, 1], central[0, :, 0])
    grid = RotationTransform([0.], degree)
    orientation = grid.grid_rotations().reshape(-1, 3, 3)
    q = len(orientation)
    sky = real_coefficients(rng, len(times), 2)
    coefficients = np.zeros((len(times), (degree+1)**2, 2*degree+1), complex)
    start = perf_counter()
    for j in range(len(times)):
        r = np.linalg.norm(central[0, j, :3])
        vr = np.dot(central[0, j, :3], central[0, j, 3:])/r
        vt = np.linalg.norm(np.cross(central[0, j, :3], central[0, j, 3:]))/r
        local_states = np.concatenate((r*orientation[:, :, 2],
                                       vr*orientation[:, :, 2]+vt*orientation[:, :, 1]), axis=1)
        emitted, _ = apparent(local_states, [times[j]-emissions[0, j]], observers[j:j+1])
        response = evaluate_sky(sky[j:j+1], emitted[..., :3]-observers[j:j+1])
        wigner = grid.inverse_wigner(response.reshape(degree+1, 2*degree+1, 2*degree+1))
        for l, block in enumerate(wigner):
            coefficients[j, l*l:(l+1)**2, degree-l:degree+l+1] = block
    construction_seconds = perf_counter()-start
    global_states = np.concatenate((np.einsum("rij,j->ri", orientation, state[0, :3]),
                                    np.einsum("rij,j->ri", orientation, state[0, 3:])), axis=1)
    start = perf_counter()
    exact, _ = apparent(global_states, times, observers)
    direct = evaluate_sky(sky, exact[..., :3]-observers)
    direct_seconds = perf_counter()-start
    rows = []
    for spin in [0, 1, 2, 4]:
        plan = DirectionalRotationTransform(phases, degree, spin, "nufft")
        start = perf_counter()
        output = plan.forward(coefficients[:, :, degree-spin:degree+spin+1]).ravel()
        transform_seconds = perf_counter()-start
        error = np.max(np.abs(output-direct))/max(1., np.max(np.abs(direct)))
        rows.append({"spin_limit": spin, "relative_score_error": float(error),
                     "passes_1e_minus_8_target": bool(error < 1e-8), "transform_seconds": transform_seconds,
                     "construction_inclusive_speedup": direct_seconds/(construction_seconds+plan.construction_seconds+transform_seconds)})
    return {"radius_au": radius, "response_degree": degree, "sky_field_degree": 2,
            "frames": len(times), "orientations": q, "includes_orientation_dependent_light_time": True,
            "response_construction_orbit_frame_evaluations": q*len(times),
            "direct_orbit_frame_evaluations": q*len(times),
            "response_construction_seconds": construction_seconds,
            "direct_geometry_and_harmonic_evaluation_seconds": direct_seconds, "spin_sweep": rows,
            "interpretation": "Tests a physical repair to the scalar transform, but builds response coefficients with one evaluation per orientation per frame; not the full fast transform."}


def roll_bandwidth_experiment():
    rows = []
    samples = 32768
    gamma = 2*np.pi*np.arange(samples)/samples
    orientation = rotations(0., np.pi/2, gamma)
    for radius in [40., 3., 1.2, 1.02]:
        vt = np.sqrt(GM["sun"]/radius)
        local = np.concatenate((radius*orientation[:, :, 2], vt*orientation[:, :, 1]), axis=1)
        # Circular orbit: central-observer emission is exactly -r/c.
        from magictools.orbits import C_AU_DAY
        emitted, _ = apparent(local, [radius/C_AU_DAY], np.array([[1., 0., 0.]]))
        directions = emitted[:, 0, :3]-np.array([1., 0., 0.])
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        angle = np.arctan2(np.linalg.norm(np.cross(directions, directions[0]), axis=1), directions @ directions[0])
        response = np.exp(-angle**2/(4*ARCSEC**2))
        power = np.abs(np.fft.fft(response)/samples)**2
        modes = np.abs(np.rint(np.fft.fftfreq(samples)*samples)).astype(int)
        by_mode = np.bincount(modes, weights=power)
        cumulative = np.cumsum(by_mode)/power.sum()
        rows.append({"radius_au": radius, "roll_samples": samples, "psf_sigma_arcsec": 1.,
                     "spin_for_99_percent_roll_response_energy": int(np.searchsorted(cumulative, .99)),
                     "energy_fraction_retained_at_spin_4": float(cumulative[4])})
    return rows


def inverse_counterexample(seed=2026):
    rng = np.random.default_rng(seed)
    templates = rng.uniform(0, 1, (256, 25))
    templates /= templates.sum(axis=1)[:, None]
    g, common = rng.normal(size=(2, 25))
    h = np.roll(g, 1)  # Same norm, different spatial pattern, not a frame swap.
    a, b = np.stack((common+g, common-g)), np.stack((common+h, common-h))
    differences = []
    for flux in (-10., 0., 10.):
        first = -.5*np.sum((a[None]-flux*templates[:, None])**2, axis=(1, 2))
        second = -.5*np.sum((b[None]-flux*templates[:, None])**2, axis=(1, 2))
        differences.append(float(np.max(np.abs(first-second))))
    return {"setup": "two independent same-epoch, same-response exposures; images (c+g,c-g) and (c+h,c-h), ||g||=||h||",
            "input_euclidean_distance": float(np.linalg.norm(a-b)),
            "maximum_full_gaussian_log_likelihood_difference": max(differences),
            "proof": "For every template p and flux F, ||c+g-Fp||^2+||c-g-Fp||^2=2||c-Fp||^2+2||g||^2. Replace g by any h of equal norm: distinct image pairs have identical full likelihoods, not just profiled scores, even with a common signal c.",
            "scope": "rules out a universal exact image-reconstruction inverse on this allowed input class; does not rule out an adjoint, quotient-space inverse, or fast forward transform"}


def run_all():
    stages = [("independent_dynamics", independent_dynamics_check),
              ("global_rank", global_rank),
              ("hierarchical_phase", lambda: [hierarchical_probe(n, case) for n in [128, 256, 512]
                                              for case in ["bound", "mixed_energy", "close_observer"]]
                +[hierarchical_probe(256, "bound", relative_tolerance=1e-3),
                  hierarchical_probe(256, "bound", frequency_fraction=1e-8),
                  hierarchical_probe(256, "bound", frequency_fraction=0.)]),
              ("rotation_identity", rotation_experiment),
              ("moving_observer_reprojection", moving_observer_experiment),
              ("light_time_coupling", light_time_experiment),
              ("directional_identity", directional_identity),
              ("directional_physical_repair", directional_physical_experiment),
              ("directional_roll_bandwidth", roll_bandwidth_experiment),
              ("physical_bandwidth", bandwidth_experiment),
              ("inverse_counterexample", inverse_counterexample)]
    result = {"seed": 2026,
              "status": "research attempts; conditional identities are not a full Kepler transform",
              "software": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                              for name in ["numpy", "scipy", "finufft", "const-ellation"]}}}
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(__file__).parent.glob("*.py"))}
    for name, run in stages:
        print(f"Running {name}", flush=True)
        result[name] = run()
    result["assessment"] = {
        "all_conic_reference_dynamics_validated": all(r["max_position_error_au"] < 1e-8 for r in result["independent_dynamics"]),
        "conditional_rotational_identity_validated": all(r["relative_score_error"] < 1e-8 for r in result["rotation_identity"]),
        "directional_repair_meets_error_target_on_smooth_test": any(r["passes_1e_minus_8_target"] for r in result["directional_physical_repair"]["spin_sweep"]),
        "continuous_six_dimensional_coverage_proved": False,
        "astronomical_resolution_construction_and_error_target_met": False,
        "first_use_full_image_transform_scaling_established": False,
        "inverse_identifiability": "universal exact image inverse obstructed; adjoints and harmonic coefficient inverse implemented",
        "novelty_established": False}
    result["full_objective_met"] = all(result["assessment"][k] for k in [
        "continuous_six_dimensional_coverage_proved", "astronomical_resolution_construction_and_error_target_met",
        "first_use_full_image_transform_scaling_established", "novelty_established"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New JSON report; never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output file already exists")
    result = run_all()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}; full_objective_met={result['full_objective_met']}")
