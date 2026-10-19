"""Finite-catalogue, uncertainty-aware searches for stellar co-convergence."""

from pathlib import Path

import numpy as np

from .galactic import GalacticPotential, integrate_stars, energy_diagnostics, KMS_TO_PC_MYR
from .io import _fields, _named_rows, _provenance, read_json
from .prove_floor import binomial_interval
from .stellar_inputs import sample_catalogue


def _cliques(adjacency, size):
    """Enumerate every fixed-size clique once; pair gating is only a necessary test."""
    def visit(chosen, available):
        if len(chosen) == size:
            yield tuple(chosen)
            return
        for index, node in enumerate(available):
            if len(chosen) + len(available) - index < size:
                break
            remaining = [other for other in available[index + 1:] if adjacency[node, other]]
            yield from visit(chosen + [node], remaining)
    yield from visit([], list(range(len(adjacency))))


def scan_convergences(paths, lookback_myr, radius_pc, group_size=3, min_support=0.2,
                      max_group_evaluations=200000):
    """Search every eligible group at every specified time in the sampled paths.

    Event: all members lie within radius_pc of their arithmetic centroid in one
    joint posterior draw. Pair-distance gating cannot discard a group meeting the
    minimum support in these exact samples: that event implies every pair <= 2R.
    This is not completeness between times or outside the supplied posterior.
    """
    paths, times = np.asarray(paths, float), np.asarray(lookback_myr, float)
    if paths.ndim != 4 or paths.shape[-1] != 6 or paths.shape[0] != len(times) or not np.all(np.isfinite(paths)):
        raise ValueError("Paths must have shape (times, samples, stars, 6)")
    if not np.isfinite(radius_pc) or radius_pc <= 0 or not 0 < min_support <= 1:
        raise ValueError("Radius must be positive and support in (0,1]")
    n_samples, n_stars = paths.shape[1:3]
    if not isinstance(group_size, int) or not 3 <= group_size <= min(6, n_stars):
        raise ValueError("group_size must be an integer between three and min(6, catalogue size)")
    best, evaluated = {}, 0
    required = int(np.ceil(min_support * n_samples - 1e-12))
    for time_index, positions in enumerate(paths[..., :3]):
        adjacency = np.zeros((n_stars, n_stars), dtype=bool)
        for i in range(n_stars - 1):
            distance2 = np.sum((positions[:, i:i + 1] - positions[:, i + 1:])**2, axis=-1)
            eligible = np.count_nonzero(distance2 <= (2 * radius_pc)**2, axis=0) >= required
            adjacency[i, i + 1:] = eligible
            adjacency[i + 1:, i] = eligible
        for group in _cliques(adjacency, group_size):
            evaluated += 1
            if evaluated > max_group_evaluations:
                raise ValueError("Group-evaluation cap exceeded; no truncated search result is reported")
            points = positions[:, group, :]
            residual = points - points.mean(axis=1, keepdims=True)
            distances2 = np.sum(residual**2, axis=-1)
            accepted = np.max(distances2, axis=1) <= radius_pc**2
            count = int(accepted.sum())
            if count < required:
                continue
            support = count / n_samples
            mean_rms2 = float(distances2.mean())
            # A compactness tie break smaller than one Monte Carlo support step.
            score = support + float(np.exp(-mean_rms2 / (2 * radius_pc**2))) / (2 * (n_samples + 1))
            row = {"indices": group, "time_index": time_index, "lookback_myr": float(times[time_index]),
                   "score": score, "support": support, "support_count": count,
                   "support_mc_interval_95": binomial_interval(count, n_samples),
                   "ensemble_rms_radius_pc": float(np.sqrt(mean_rms2))}
            if group not in best or score > best[group]["score"]:
                best[group] = row
    ranked = sorted(best.values(), key=lambda row: (-row["score"], row["indices"]))
    return ranked, evaluated


def _settings(value, n_stars):
    _fields(value, ["lookback_max_myr", "time_step_myr", "radius_pc"],
            ["group_size", "min_support", "n_samples", "max_step_myr", "n_controls", "seed",
             "max_results", "max_group_evaluations", "association_radius_pc"])
    result = {"group_size": 3, "min_support": 0.2, "n_samples": 128, "max_step_myr": 0.02,
              "n_controls": 63, "seed": 2026, "max_results": 20, "max_group_evaluations": 200000,
              "association_radius_pc": value["radius_pc"], **value}
    for key in ["lookback_max_myr", "time_step_myr", "radius_pc", "max_step_myr", "association_radius_pc"]:
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key, low, high in [("group_size", 3, min(6, n_stars)), ("n_samples", 16, 1024),
                           ("n_controls", 0, 999), ("max_results", 1, 100),
                           ("max_group_evaluations", 1, 1000000), ("seed", 0, 2**32 - 1)]:
        if isinstance(result[key], bool) or not isinstance(result[key], int) or not low <= result[key] <= high:
            raise ValueError(f"{key} must be an integer in [{low},{high}]")
    if not np.isfinite(result["min_support"]) or not 0 < result["min_support"] <= 1:
        raise ValueError("min_support must lie in (0,1]")
    if result["lookback_max_myr"] > 100 or result["time_step_myr"] > result["lookback_max_myr"]:
        raise ValueError("Search supports at most 100 Myr, with time step <= search interval")
    if int(np.ceil(result["lookback_max_myr"] / result["time_step_myr"])) > 5000:
        raise ValueError("Time grid exceeds 5001 points")
    return result


def _dossier(candidate, paths, times, stars, counterparts, counterpart_paths, settings):
    indices, time_index = candidate.pop("indices"), candidate["time_index"]
    points = paths[:, :, indices, :3]
    centers = points.mean(axis=2)
    residual2 = np.sum((points - centers[:, :, None, :])**2, axis=-1)
    events = np.max(residual2, axis=2) <= settings["radius_pc"]**2
    accepted = events[time_index]
    closest_indices = np.argmin(residual2.mean(axis=2), axis=0)
    ever = events.any(axis=0)
    candidate.update(star_ids=[stars[i]["id"] for i in indices],
        origin_given_convergence_percentiles_pc=np.quantile(centers[time_index, accepted], [0.16, 0.5, 0.84], axis=0).tolist(),
        percentile_levels=[0.16, 0.5, 0.84],
        closest_rms_epoch_percentiles_myr=np.quantile(times[closest_indices], [0.16, 0.5, 0.84]).tolist(),
        support_at_any_grid_time=float(ever.mean()),
        support_time_profile=events.mean(axis=1).tolist(),
        peak_on_search_boundary=bool(time_index in (0, len(times) - 1)),
        member_velocity_given_convergence_percentiles_kms=np.quantile(paths[time_index, accepted][:, indices, 3:], [0.16, 0.5, 0.84], axis=0).tolist())
    associations = []
    for i, counterpart in enumerate(counterparts):
        separation = np.linalg.norm(counterpart_paths[time_index, accepted, i, :3] - centers[time_index, accepted], axis=-1)
        probability = float(np.mean(separation <= settings["association_radius_pc"]))
        associations.append({"id": counterpart["id"], "label": counterpart.get("label", counterpart["id"]),
            "conditional_proximity_fraction": probability,
            "separation_percentiles_pc": np.quantile(separation, [0.16, 0.5, 0.84]).tolist()})
    candidate["supplied_counterpart_comparisons"] = sorted(associations, key=lambda row: -row["conditional_proximity_fraction"])
    return candidate


def run_ejector(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ["schema_version", "epoch_label", "potential", "stars", "search"],
            ["counterparts", "galactocentric_frame"])
    if manifest["schema_version"] != 1 or not isinstance(manifest["epoch_label"], str) or not manifest["epoch_label"]:
        raise ValueError("Require schema_version 1 and a nonempty common epoch label")
    _named_rows(manifest["stars"], "stars")
    stars = manifest["stars"]
    if not 3 <= len(stars) <= 128:
        raise ValueError("Supply 3--128 preselected stars at a common epoch")
    counterparts = manifest.get("counterparts", [])
    if not isinstance(counterparts, list):
        raise ValueError("counterparts must be a list")
    if counterparts:
        _named_rows(counterparts, "counterparts")
        if len(counterparts) > 128 or {s["id"] for s in stars} & {s["id"] for s in counterparts}:
            raise ValueError("Counterparts must be distinct from searched stars, with at most 128 entries")
    settings = _settings(manifest["search"], len(stars))
    potential = GalacticPotential(manifest["potential"])
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence(settings["seed"]).spawn(3)]
    states = sample_catalogue(stars, settings["n_samples"], generators[0], manifest.get("galactocentric_frame"))
    count = int(np.ceil(settings["lookback_max_myr"] / settings["time_step_myr"]))
    times = np.linspace(0, settings["lookback_max_myr"], count + 1)
    paths = integrate_stars(states, -times, potential, settings["max_step_myr"])
    # Always verify observed trajectories against a smaller integration step.
    fine = integrate_stars(states, -times, potential, settings["max_step_myr"] / 2)
    difference = np.linalg.norm(paths[..., :3] - fine[..., :3], axis=-1)
    max_difference = float(difference.max())
    del fine
    if max_difference > 0.1 * settings["radius_pc"]:
        raise ValueError("Halving the integration step changes positions by >10% of the convergence radius; reduce max_step_myr")
    scan_options = {key: settings[key] for key in ["radius_pc", "group_size", "min_support", "max_group_evaluations"]}
    ranked, evaluations = scan_convergences(paths, times, **scan_options)
    strata = {}
    for i, star in enumerate(stars):
        label = star.get("control_stratum", "all")
        if not isinstance(label, str) or not label:
            raise ValueError("Control stratum must be a nonempty string")
        strata.setdefault(label, []).append(i)
    if settings["n_controls"] and all(len(indices) == 1 for indices in strata.values()):
        raise ValueError("Every control stratum is a singleton; velocity permutation cannot test this catalogue")
    maxima = []
    for _ in range(settings["n_controls"]):
        permutation = np.arange(len(stars))
        for indices in strata.values():
            permutation[indices] = generators[2].permutation(indices)
        scrambled = states.copy()
        # Preserve each star's velocity cloud intact, but break its assignment to
        # the position cloud. This is an explicitly specified conditional null.
        scrambled[..., 3:] = states[:, permutation, 3:]
        controls = integrate_stars(scrambled, -times, potential, settings["max_step_myr"])
        controls_ranked, _ = scan_convergences(controls, times, **scan_options)
        maxima.append(controls_ranked[0]["score"] if controls_ranked else 0.0)
    counterpart_paths = None
    if counterparts:
        counterpart_states = sample_catalogue(counterparts, settings["n_samples"], generators[1], manifest.get("galactocentric_frame"))
        counterpart_paths = integrate_stars(counterpart_states, -times, potential, settings["max_step_myr"] / 2)
    results = []
    for row in ranked[:settings["max_results"]]:
        row = _dossier(row.copy(), paths, times, stars, counterparts, counterpart_paths, settings)
        if maxima:
            exceedances = int(np.count_nonzero(np.asarray(maxima) >= row["score"]))
            row["search_wide_velocity_permutation_p"] = (1 + exceedances) / (len(maxima) + 1)
            row["control_exceedances"] = exceedances
        else:
            row["search_wide_velocity_permutation_p"] = None
        results.append(row)
    max_speed = float(np.max(np.linalg.norm(paths[..., 3:], axis=-1)))
    grid_displacement = max_speed * KMS_TO_PC_MYR * float(np.max(np.diff(times)))
    return {"schema_version": 1, "tool": "find_ejector", "provenance": _provenance([path]),
            "epoch_label": manifest["epoch_label"], "potential": manifest["potential"],
            "galactocentric_frame": manifest.get("galactocentric_frame"), "settings": settings,
            "lookback_grid_myr": times.tolist(), "candidates": results,
            "search_summary": {"qualifying_groups": len(ranked), "reported_groups": len(results),
                "group_time_evaluations": evaluations, "catalogue_size": len(stars)},
            "numerics": {**energy_diagnostics(paths, potential), "half_step_max_position_difference_pc": max_difference,
                "max_speed_grid_displacement_pc": float(grid_displacement),
                "time_grid_may_skip_convergences": bool(grid_displacement > settings["radius_pc"])},
            "controls": {"method": "velocity-cloud permutation within declared strata; complete group/time search repeated",
                "max_search_scores": maxima, "n_controls": len(maxima),
                "minimum_resolvable_p": 1 / (len(maxima) + 1) if maxima else None,
                "assumption": "velocity-cloud assignments are exchangeable within each stratum under the chosen null"},
            "interpretation": "Kinematic co-convergence in a supplied catalogue; neither proof of ejection nor evidence of a dark source"}
