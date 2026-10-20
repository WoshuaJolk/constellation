"""Reproducible demonstrations for the finite-model research tools."""

import json
from pathlib import Path

import numpy as np

from .ejector import run_ejector
from .galactic import GalacticPotential, integrate_stars
from .photometry import gaussian_psf
from .prove_floor import run_floor


def _directory(directory):
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Demo directory must be new or empty; existing files are never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save(directory, filename, value):
    (directory / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def create_floor_demo(directory, seed=2026):
    directory = _directory(directory)
    yy, xx = np.mgrid[:7, :7]
    templates = []
    for rate in [-0.6, -0.2, 0.2, 0.6]:
        templates.append(np.concatenate([gaussian_psf(xx, yy, 3 + rate * (t - 2.5), 3, 0.8).ravel() for t in range(6)]))
    templates = np.array(templates)
    # Known neighboring-pixel correlations, with independent frames.
    covariance = np.eye(294) * 4
    for frame in range(6):
        for pixel in range(48):
            i = frame * 49 + pixel
            covariance[i, i + 1] = covariance[i + 1, i] = 0.4
    background = np.zeros((294, 6))
    for frame in range(6):
        background[frame * 49:(frame + 1) * 49, frame] = 1
    outputs = {}
    for label, flux in [("faint", 4), ("bright", 14)]:
        manifest = {"schema_version": 1, "templates": templates.tolist(), "covariance": covariance.tolist(),
                    "background_design": background.tolist(), "prior_weights": [1, 1, 1, 1],
                    "flux": flux, "alpha": 0.01, "target_power": 0.9,
                    "calibration_samples": 8192, "evaluation_samples": 8192, "seed": seed}
        _save(directory, f"{label}-input.json", manifest)
        result = run_floor(directory / f"{label}-input.json")
        _save(directory, f"{label}-result.json", result)
        outputs[label] = {"flux": flux, "analytic_power_upper_bound": result["bounds"]["best_power_upper_bound"],
                          "detector_power_estimate": result["detector"]["mixture_power_estimate"],
                          "detector_power_interval_95": result["detector"]["mixture_power_interval_95"]}
    summary = {"synthetic": True, "directory": str(directory), "seed": seed,
        "description": "Four trajectory templates over six frames, correlated Gaussian pixels, six unknown backgrounds",
        "alpha": 0.01, "target_power": 0.9,
        "necessary_flux_lower_bound": result["necessary_flux_lower_bound"], "experiments": outputs}
    _save(directory, "summary.json", summary)
    return summary


def create_ejector_demo(directory, seed=2026):
    directory = _directory(directory)
    rng = np.random.default_rng(seed)
    specification = {"components": [
        {"type": "plummer", "mass_msun": 5e9, "scale_pc": 300},
        {"type": "miyamoto_nagai", "mass_msun": 6.8e10, "a_pc": 3000, "b_pc": 280},
        {"type": "log_halo", "speed_kms": 220, "core_pc": 12000}]}
    potential = GalacticPotential(specification)
    origin = np.array([-8000., 200., 40.])
    cluster_velocity = np.array([0., 220., 0.])
    age = 2.4
    offsets = np.array([[100, 60, 25], [-90, 80, -30], [60, -100, 40], [-70, -80, -35]])
    birth = np.column_stack([origin + rng.normal(0, 0.1, (4, 3)), cluster_velocity + offsets])
    present = integrate_stars(birth, [0, age], potential, 0.001)[-1]
    stars = []
    covariance = np.diag([0.2**2] * 3 + [0.15**2] * 3)
    # Include a modest explicit position/velocity correlation.
    covariance[0, 3] = covariance[3, 0] = 0.01
    for i, state in enumerate(present):
        stars.append({"id": f"ejected-{i + 1}", "state_pc_kms": state.tolist(), "covariance": covariance.tolist()})
    for i in range(4):
        state = np.r_[origin + rng.normal(0, 500, 3), cluster_velocity + rng.normal(0, 110, 3)]
        stars.append({"id": f"field-{i + 1}", "state_pc_kms": state.tolist(), "covariance": covariance.tolist()})
    cluster = integrate_stars(np.r_[origin, cluster_velocity], [0, age], potential, 0.001)[-1]
    manifest = {"schema_version": 1, "epoch_label": "synthetic present; all states simultaneous",
        "potential": specification, "stars": stars,
        "counterparts": [{"id": "birth-cluster", "label": "Injected moving cluster",
                          "state_pc_kms": cluster.tolist(), "covariance": (covariance / 4).tolist()}],
        "search": {"lookback_max_myr": 4, "time_step_myr": 0.01, "radius_pc": 5,
                   "group_size": 4, "n_samples": 64, "n_controls": 63, "max_step_myr": 0.01, "seed": seed}}
    _save(directory, "input.json", manifest)
    _save(directory, "truth.json", {"age_myr": age, "origin_pc": origin.tolist(),
          "member_ids": [row["id"] for row in stars[:4]], "synthetic": True})
    result = run_ejector(directory / "input.json")
    _save(directory, "result.json", result)
    best = result["candidates"][0] if result["candidates"] else None
    summary = {"synthetic": True, "directory": str(directory), "seed": seed,
               "injected_age_myr": age, "injected_origin_pc": origin.tolist(),
               "best_candidate": best, "control_trials": result["controls"]["n_controls"],
               "numerics": result["numerics"]}
    # Keep stdout concise; the full time profile remains in result.json.
    if best:
        summary["best_candidate"] = {key: best[key] for key in ["star_ids", "lookback_myr", "support",
            "origin_given_convergence_percentiles_pc", "search_wide_velocity_permutation_p", "supplied_counterpart_comparisons"]}
    _save(directory, "summary.json", summary)
    return summary
