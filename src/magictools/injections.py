"""Run the finite trajectory search after signal injections into supplied images."""

from pathlib import Path

import numpy as np

from .image_bank import ImageBank
from .io import _fields, _provenance, read_json


def run_injections(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ["schema_version", "photometry_manifest", "injection_trajectory_id", "reference_fluxes", "offsets_arcsec"], ["alpha", "injection_psf_sigma_factor"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    alpha = manifest.get("alpha", 0.01)
    factor = manifest.get("injection_psf_sigma_factor", 1)
    fluxes = np.asarray(manifest["reference_fluxes"], dtype=float)
    offsets = np.asarray(manifest["offsets_arcsec"], dtype=float)
    if fluxes.ndim != 1 or not 1 <= len(fluxes) <= 100 or not np.all(np.isfinite(fluxes)) or np.any(fluxes < 0) or len(np.unique(fluxes)) != len(fluxes):
        raise ValueError("Supply 1..100 distinct finite nonnegative reference_fluxes")
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not 1 <= len(offsets) <= 1000 or not np.all(np.isfinite(offsets)) or np.any(np.abs(offsets) > 3600):
        raise ValueError("Supply 1..1000 finite east/north offsets within +/-3600 arcsec")
    if len(np.unique(offsets, axis=0)) != len(offsets):
        raise ValueError("Duplicate injection offsets would duplicate the same experiment")
    if not np.isfinite(alpha) or not 0 < alpha < 1 or not np.isfinite(factor) or not 0.5 <= factor <= 2:
        raise ValueError("alpha must be in (0,1), injection_psf_sigma_factor in [0.5,2]")
    bank = ImageBank(path.parent / manifest["photometry_manifest"])
    true_id = manifest["injection_trajectory_id"]
    if true_id not in bank.tracks or not bank.tracks[true_id]:
        raise ValueError("Injection trajectory must exist and contain predictions")
    if factor != 1 and any(frame["kernel"] is not None for frame in bank.frames.values()):
        raise ValueError("PSF width perturbations require Gaussian PSFs")
    if len(fluxes) * len(offsets) * len(bank.tracks) * len(bank.frames) > 200000:
        raise ValueError("Injection experiment exceeds 200,000 frame fits; split explicitly")
    def search(tracks, additions=None):
        rows = []
        for tid, track in tracks.items():
            measurements, missing = bank.measure(track, additions)
            rows.append({"id": tid, "joint": bank.combine(measurements, len(tracks), alpha), "unavailable": missing})
        return sorted(rows, key=lambda row: row["joint"]["snr"] if row["joint"] else -np.inf, reverse=True)
    baselines, trials = [], []
    for offset_index, offset in enumerate(offsets):
        tracks = {tid: bank.shift(track, *offset) for tid, track in bank.tracks.items()}
        baseline = search(tracks)
        baseline_detected = bool(baseline[0]["joint"] and baseline[0]["joint"]["passes_gaussian_threshold"])
        baselines.append({"offset_index": offset_index, "offset_arcsec": offset.tolist(), "detected_before_injection": baseline_detected, "search": baseline})
        source = {eid: bank.template(eid, position, factor) * bank.frames[eid]["exposure"].get("flux_scale", 1)
                  for eid, position in tracks[true_id].items()}
        for flux in fluxes:
            result = search(tracks, {eid: flux * template for eid, template in source.items()})
            truth = next(row for row in result if row["id"] == true_id)
            joint = truth["joint"]
            recovered = bool(result[0]["id"] == true_id and joint and joint["passes_gaussian_threshold"])
            # Ties in a degenerate bank are not a correctly identified orbit.
            tied = bool(joint and any(row["id"] != true_id and row["joint"] and abs(row["joint"]["snr"] - joint["snr"]) <= 1e-10 for row in result))
            trials.append({"offset_index": offset_index, "reference_flux": float(flux), "top_trajectory_id": result[0]["id"],
                           "true_trajectory_joint": joint, "true_trajectory_unavailable": truth["unavailable"],
                           "recovered": recovered and not tied, "tied_with_another_trajectory": tied,
                           "new_recovery": recovered and not tied and not baseline_detected})
    summary = []
    for flux in fluxes:
        selected = [row for row in trials if row["reference_flux"] == flux]
        errors = [(row["true_trajectory_joint"]["flux"] - flux) / row["true_trajectory_joint"]["flux_error"] for row in selected if row["true_trajectory_joint"]]
        summary.append({"reference_flux": float(flux), "n_placements": len(selected),
                        "n_recovered": sum(row["recovered"] for row in selected), "recovery_fraction": float(np.mean([row["recovered"] for row in selected])),
                        "new_recovery_fraction": float(np.mean([row["new_recovery"] for row in selected])),
                        "flux_error_in_sigma_percentiles_16_50_84": np.percentile(errors, [16, 50, 84]).tolist() if errors else None})
    return {"schema_version": 1, "tool": "inject_tracks", "provenance": _provenance([path] + [r["path"] for r in bank.original_report["provenance"]["inputs"]]),
            "injection_trajectory_id": true_id, "alpha": alpha, "injection_psf_sigma_factor": factor,
            "summary": summary, "baselines": baselines, "trials": trials,
            "scope": "Signals added individually to unchanged supplied backgrounds; all supplied tracks shifted and searched at every placement. No added source shot noise. Fractions describe these placements, which may overlap and are not independent binomial draws or population completeness. Baseline detections remain visible; no survey-wide false-positive calibration."}
