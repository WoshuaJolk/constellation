"""Multi-exposure diagnostics; no invented combined real/bogus probability."""

from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .image_bank import ImageBank, shared_comparison
from .io import _fields, _provenance, read_json


def run_judge(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ["schema_version", "photometry_manifest", "candidate_id", "times_mjd_tdb"], ["alpha", "jackknife_min_snr"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    alpha, minimum = manifest.get("alpha", 0.01), manifest.get("jackknife_min_snr", 3)
    if not np.all(np.isfinite([alpha, minimum])) or not 0 < alpha < 1 or minimum < 0:
        raise ValueError("alpha must be in (0,1); jackknife_min_snr must be nonnegative")
    bank = ImageBank(path.parent / manifest["photometry_manifest"])
    candidate_id = manifest["candidate_id"]
    if candidate_id not in bank.tracks:
        raise ValueError("Unknown candidate_id")
    times = manifest["times_mjd_tdb"]
    if not isinstance(times, dict) or set(times) != set(bank.frames) or not all(np.isfinite(t) for t in times.values()):
        raise ValueError("times_mjd_tdb needs one finite midpoint for every exposure")
    track = bank.tracks[candidate_id]
    measured, unavailable = bank.measure(track)
    joint = bank.combine(measured, trials=len(bank.tracks), alpha=alpha)
    ordered = sorted(track, key=lambda eid: (times[eid], eid))
    checks, flags = [], []
    if ordered:
        # Three static sky positions span the candidate's arc. Reverse about its
        # middle sampled position; neither transform is a calibrated null draw.
        anchors = sorted(set([0, len(ordered) // 2, len(ordered) - 1]))
        for index in anchors:
            anchor = bank.sky(ordered[index], track[ordered[index]])
            static = {}
            for eid in ordered:
                x, y = bank.frames[eid]["wcs"].world_to_pixel(anchor)
                static[eid] = dict(x=float(x), y=float(y), trail_dx=0, trail_dy=0)
            control, missing = bank.measure(static)
            checks.append({"id": f"stationary-at-{ordered[index]}", "kind": "stationary", "unavailable": missing,
                           **shared_comparison(bank, measured, control)})
        anchor = bank.sky(ordered[len(ordered) // 2], track[ordered[len(ordered) // 2]])
        reverse = {}
        for eid in ordered:
            east, north = anchor.spherical_offsets_to(bank.sky(eid, track[eid]))
            coordinate = anchor.spherical_offsets_by(-east, -north)
            x, y = bank.frames[eid]["wcs"].world_to_pixel(coordinate)
            reverse[eid] = {"x": float(x), "y": float(y), "trail_dx": -track[eid]["trail_dx"], "trail_dy": -track[eid]["trail_dy"]}
        control, missing = bank.measure(reverse)
        checks.append({"id": "reversed-track", "kind": "reversed", "unavailable": missing,
                       **shared_comparison(bank, measured, control)})
    leave_one_out = []
    for eid in measured:
        fit = bank.combine({key: value for key, value in measured.items() if key != eid}, len(bank.tracks), alpha)
        leave_one_out.append({"omitted_exposure_id": eid, "joint": fit})
    halves = []
    ordered_measured = [eid for eid in ordered if eid in measured]
    if len(ordered_measured) >= 2:
        for ids in [ordered_measured[:len(ordered_measured) // 2], ordered_measured[len(ordered_measured) // 2:]]:
            halves.append({"exposure_ids": ids, "joint": bank.combine({eid: measured[eid] for eid in ids}, len(bank.tracks), alpha)})
    if joint is None:
        flags.append("no_usable_measurements")
    elif not joint["passes_gaussian_threshold"]:
        flags.append("fails_supplied_bank_gaussian_threshold")
    if len(measured) < 3:
        flags.append("fewer_than_three_usable_exposures")
    if any(row["joint"] is None or row["joint"]["snr"] < minimum for row in leave_one_out):
        flags.append("jackknife_below_requested_snr")
    if any(row["candidate_snr_minus_control_snr"] is not None and row["candidate_snr_minus_control_snr"] <= 1e-8 for row in checks):
        flags.append("a_control_matches_or_exceeds_candidate_on_common_exposures")
    if any(len(row["common_exposure_ids"]) < len(measured) for row in checks):
        flags.append("control_coverage_incomplete")
    baseline = max(times.values()) - min(times.values())
    if baseline == 0:
        flags.append("no_temporal_baseline")
    residuals = [{"exposure_id": eid, "chi2": value.chi2, "dof": value.dof,
                  "gaussian_residual_p": float(chi2.sf(value.chi2, value.dof)), **value.to_dict()} for eid, value in measured.items()]
    flux_p = float(chi2.sf(joint["constant_flux_chi2"], joint["constant_flux_dof"])) if joint and joint["constant_flux_dof"] > 0 else None
    provenance = _provenance([path] + [row["path"] for row in bank.original_report["provenance"]["inputs"]])
    return {"schema_version": 1, "tool": "judge_line", "candidate_id": candidate_id, "provenance": provenance,
            "joint": joint, "measurements": residuals, "unavailable": unavailable, "leave_one_out": leave_one_out,
            "chronological_halves": halves, "controls": checks, "constant_flux_gaussian_p": flux_p,
            "time_baseline_days": float(baseline),
            "flags": flags, "jackknife_min_snr": minimum, "combined_false_positive_probability": None,
            "scope": "Candidate diagnostics for a supplied bank: Gaussian residuals, constant flux, temporal halves, leave-one-out, three stationary controls and a reversed track. No orbit refit, astrometric confirmation, or empirical discovery probability."}
