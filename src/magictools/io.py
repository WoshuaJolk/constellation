"""Versioned JSON manifests and local FITS inputs; no implicit archive access."""

import hashlib
import json
from pathlib import Path

import astropy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import scipy

from . import __version__
from .photometry import forced_photometry
from .search import combine_measurements
from .nulls import gaussian_log_null, rank_exposures, update_nulls


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-standard JSON constant: {value}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with open(path) as stream:
        return json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"Invalid fields: missing={sorted(missing)}, unknown={sorted(extra)}")


def _named_rows(rows, kind):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{kind} must be a nonempty list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"Each {kind} entry needs a nonempty string id")
        if row["id"] in seen:
            raise ValueError(f"Duplicate {kind} id: {row['id']}")
        seen.add(row["id"])
    return seen


def _provenance(paths):
    inputs = []
    for path in sorted(set(Path(p).resolve() for p in paths)):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        inputs.append({"path": str(path), "sha256": digest.hexdigest()})
    return {"magictools": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "astropy": astropy.__version__, "inputs": inputs}


def _fits_plane(path, selector):
    # Explicit HDU indexing supports both names and indices, without getdata's
    # implicit fallback from an empty primary HDU to the first image extension.
    with fits.open(path, memmap=False) as hdus:
        if hdus[selector].data is None:
            raise ValueError(f"No image data in {path}, HDU {selector}")
        return np.array(hdus[selector].data, dtype=float)


def run_photometry(manifest_path, joint=False, alpha=0.01):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ["schema_version", "exposures", "trajectories"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    exposure_ids = _named_rows(manifest["exposures"], "exposures")
    _named_rows(manifest["trajectories"], "trajectories")
    predictions, results = {}, {}
    for trajectory in manifest["trajectories"]:
        _fields(trajectory, ["id", "positions"])
        if not isinstance(trajectory["positions"], list):
            raise ValueError("Trajectory positions must be a list (empty means no coverage)")
        lookup = {}
        for position in trajectory["positions"]:
            _fields(position, ["exposure_id"], ["x", "y", "ra_deg", "dec_deg", "trail_dx", "trail_dy"])
            exp_id = position["exposure_id"]
            if exp_id not in exposure_ids or exp_id in lookup:
                raise ValueError("Position references an unknown or repeated exposure_id")
            keys = set(position) - {"exposure_id", "trail_dx", "trail_dy"}
            if keys not in ({"x", "y"}, {"ra_deg", "dec_deg"}):
                raise ValueError("Position requires exactly x,y or ra_deg,dec_deg")
            if not all(isinstance(position[k], (int, float)) and np.isfinite(position[k]) for k in keys):
                raise ValueError("Position coordinates must be finite numbers")
            if "dec_deg" in position and not (-90 <= position["dec_deg"] <= 90 and 0 <= position["ra_deg"] < 360):
                raise ValueError("ICRS RA must be in [0,360), Dec in [-90,90] degrees")
            if not np.all(np.isfinite([position.get("trail_dx", 0), position.get("trail_dy", 0)])):
                raise ValueError("Trail offsets must be finite pixels")
            lookup[exp_id] = position
        predictions[trajectory["id"]] = lookup
        results[trajectory["id"]] = {"id": trajectory["id"], "measurements": [], "unavailable": []}
    files, image_sources = [path], set()
    fits_by_trajectory = {key: [] for key in results}
    scales_by_trajectory = {key: [] for key in results}

    for exposure in manifest["exposures"]:
        _fields(exposure, ["id", "image", "variance", "psf_sigma_pixels"],
                ["image_hdu", "variance_hdu", "mask", "mask_hdu", "flux_scale", "psf_image", "psf_hdu"])
        image_path = (path.parent / exposure["image"]).resolve()
        image_hdu = exposure.get("image_hdu", 0)
        # Resolve numeric and named HDUs to the same identity to prevent double use.
        with fits.open(image_path, memmap=False) as hdus:
            hdu_index = hdus.index_of(image_hdu)
            source = (image_path, hdu_index)
            if source in image_sources:
                raise ValueError("The same image HDU cannot count as independent exposures twice")
            image_sources.add(source)
            image = np.array(hdus[hdu_index].data, dtype=float)
            header = hdus[hdu_index].header.copy()
        files.append(image_path)
        if image.ndim != 2:
            raise ValueError(f"Exposure {exposure['id']} must have a 2D science image")
        variance = exposure["variance"]
        if isinstance(variance, str):
            variance_path = (path.parent / variance).resolve()
            variance = _fits_plane(variance_path, exposure.get("variance_hdu", 0))
            files.append(variance_path)
            if variance.shape != image.shape:
                raise ValueError("Variance image shape must match science image")
        elif not isinstance(variance, (int, float)) or not np.isfinite(variance) or variance <= 0:
            raise ValueError("variance must be a positive finite scalar or FITS path")
        mask = None
        if "mask" in exposure:
            mask_path = (path.parent / exposure["mask"]).resolve()
            mask = _fits_plane(mask_path, exposure.get("mask_hdu", 0))
            files.append(mask_path)
            if mask.shape != image.shape:
                raise ValueError("Mask shape must match science image")
        sigma, scale = exposure["psf_sigma_pixels"], exposure.get("flux_scale", 1.0)
        psf_kernel = None
        if "psf_image" in exposure:
            psf_path = (path.parent / exposure["psf_image"]).resolve()
            psf_kernel = _fits_plane(psf_path, exposure.get("psf_hdu", 0))
            from .photometry import validate_psf_kernel
            psf_kernel = validate_psf_kernel(psf_kernel)
            files.append(psf_path)
        if not np.all(np.isfinite([sigma, scale])) or sigma <= 0 or scale <= 0:
            raise ValueError("PSF sigma and flux scale must be positive and finite")
        celestial_wcs = None
        for trajectory_id, lookup in predictions.items():
            result = results[trajectory_id]
            position = lookup.get(exposure["id"])
            if position is None:
                result["unavailable"].append({"exposure_id": exposure["id"], "reason": "No supplied prediction"})
                continue
            if "x" in position:
                x, y = position["x"], position["y"]
            else:
                if celestial_wcs is None:
                    celestial_wcs = WCS(header).celestial
                    if not celestial_wcs.has_celestial:
                        raise ValueError("RA/Dec predictions require a celestial WCS")
                coordinate = SkyCoord(position["ra_deg"], position["dec_deg"], unit="deg", frame="icrs")
                x, y = map(float, celestial_wcs.world_to_pixel(coordinate))
            try:
                measurement = forced_photometry(image, variance, x, y, sigma, mask,
                                                position.get("trail_dx", 0), position.get("trail_dy", 0), psf_kernel)
            except ValueError as error:
                result["unavailable"].append({"exposure_id": exposure["id"], "reason": str(error)})
                continue
            result["measurements"].append({"exposure_id": exposure["id"], "flux_scale": scale,
                                           "psf_model": "native_pixel_empirical" if psf_kernel is not None else "gaussian",
                                           "trail_dx": position.get("trail_dx", 0), "trail_dy": position.get("trail_dy", 0),
                                           **measurement.to_dict()})
            fits_by_trajectory[trajectory_id].append(measurement)
            scales_by_trajectory[trajectory_id].append(scale)
    trials = len(results)
    for key, result in results.items():
        result["status"] = "ok" if result["measurements"] else "no_usable_measurements"
        if joint:
            result["joint"] = combine_measurements(fits_by_trajectory[key], scales_by_trajectory[key], trials, alpha) if result["measurements"] else None
    candidates = list(results.values())
    if joint:
        candidates.sort(key=lambda row: row["joint"]["snr"] if row["joint"] else -np.inf, reverse=True)
    return {"schema_version": 1, "tool": "dig_floor" if joint else "force_track",
            "model": "Gaussian or supplied native-pixel empirical PSF, optional linear trail; constant local background; independent supplied Gaussian variance",
            "provenance": _provenance(files), "trajectories": candidates}


def run_nulls(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ["schema_version", "hypotheses", "non_detections"], ["candidate_exposures", "calibrations", "ranking_target"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    _named_rows(manifest["hypotheses"], "hypotheses")
    for hypothesis in manifest["hypotheses"]:
        _fields(hypothesis, ["id", "prior_weight"], ["orbit_id", "reference_flux"])
        if "orbit_id" in hypothesis and (not isinstance(hypothesis["orbit_id"], str) or not hypothesis["orbit_id"]):
            raise ValueError("orbit_id must be a nonempty string")
        if "reference_flux" in hypothesis and (not np.isfinite(hypothesis["reference_flux"]) or hypothesis["reference_flux"] < 0):
            raise ValueError("reference_flux must be finite and nonnegative")
    grouped = any("orbit_id" in h for h in manifest["hypotheses"])
    if grouped and not all("orbit_id" in h for h in manifest["hypotheses"]):
        raise ValueError("Supply orbit_id for every hypothesis or none")
    ranking_target = manifest.get("ranking_target", "hypothesis")
    if ranking_target not in {"hypothesis", "orbit"} or (ranking_target == "orbit" and not grouped):
        raise ValueError("ranking_target is hypothesis or orbit; orbit requires orbit_id labels")
    n = len(manifest["hypotheses"])
    observed = manifest["non_detections"]
    future = manifest.get("candidate_exposures", [])
    if not isinstance(observed, list) or not isinstance(future, list):
        raise ValueError("non_detections and candidate_exposures must be lists")
    if observed:
        _named_rows(observed, "non_detections")
    if future:
        _named_rows(future, "candidate_exposures")
    if {row["id"] for row in observed} & {row["id"] for row in future}:
        raise ValueError("An observed exposure cannot also be a future candidate")
    calibration_rows = manifest.get("calibrations", [])
    if not isinstance(calibration_rows, list):
        raise ValueError("calibrations must be a list")
    if calibration_rows:
        _named_rows(calibration_rows, "calibrations")
    calibrations, null_counts = {}, {}
    for row in calibration_rows:
        _fields(row, ["id", "recovered", "injected"], ["beta_prior_alpha", "beta_prior_beta"])
        recovered, injected = np.asarray(row["recovered"], dtype=float), np.asarray(row["injected"], dtype=float)
        if (recovered.shape != (n,) or injected.shape != (n,) or not np.all(np.isfinite([recovered, injected]))
                or np.any(recovered < 0) or np.any(injected < recovered) or np.any(injected > 1e9)
                or np.any(recovered != np.floor(recovered)) or np.any(injected != np.floor(injected))):
            raise ValueError("Calibration needs integer 0 <= recovered <= injected <= 1e9 per hypothesis")
        a, b = row.get("beta_prior_alpha", .5), row.get("beta_prior_beta", .5)
        if not np.all(np.isfinite([a, b])) or min(a, b) <= 0 or max(a, b) > 1e9:
            raise ValueError("Calibration Beta prior parameters must be in (0,1e9]")
        calibrations[row["id"]] = (a + recovered, b + injected - recovered)
        null_counts[row["id"]] = 0
    direct_observed = []
    for row in observed:
        if "calibration_id" in row:
            _fields(row, ["id", "calibration_id"])
            if row["calibration_id"] not in calibrations:
                raise ValueError("Unknown calibration_id")
            null_counts[row["calibration_id"]] += 1
        else:
            direct_observed.append(row)
    def log_probabilities(row):
        if "calibration_id" in row:
            _fields(row, ["id", "calibration_id"])
            if row["calibration_id"] not in calibrations:
                raise ValueError("Unknown calibration_id")
            a, b = calibrations[row["calibration_id"]]
            b = b + null_counts[row["calibration_id"]]
            return -np.logaddexp(0, np.log(a) - np.log(b))
        if "detection_probability" in row:
            _fields(row, ["id", "detection_probability"])
            values = np.asarray(row["detection_probability"], dtype=float)
            if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
                raise ValueError("One detection probability in [0,1] is required per hypothesis")
            with np.errstate(divide="ignore"):
                return np.log1p(-values)
        _fields(row, ["id", "expected_snr", "threshold_snr"], ["coverage"])
        if np.shape(row["expected_snr"]) != (n,):
            raise ValueError("One expected SNR is required per hypothesis")
        result = gaussian_log_null(row["expected_snr"], row["threshold_snr"], row.get("coverage", 1))
        if result.shape != (n,):
            raise ValueError("Threshold/coverage must be scalar or one value per hypothesis")
        return result
    likelihood_rows = [log_probabilities(row) for row in direct_observed]
    # The same unknown efficiency reused m times must be integrated jointly:
    # E[(1-p)^m] = B(a,b+m)/B(a,b), NOT E[1-p]^m.
    for key, (a, b) in calibrations.items():
        count = null_counts[key]
        if count:
            # Log rising-factor ratio avoids cancellation in betaln at large n.
            likelihood_rows.append(-sum(np.logaddexp(0, np.log(a) - np.log(b + j)) for j in range(count)))
    likelihood = np.array(likelihood_rows).reshape(len(likelihood_rows), n)
    result = update_nulls([h["prior_weight"] for h in manifest["hypotheses"]], likelihood)
    candidates = np.array([-np.expm1(log_probabilities(row)) for row in future]).reshape(len(future), n)
    posterior = np.asarray(result["posterior_weights"])
    orbit_summary, orbit_ranking = [], []
    if grouped:
        orbit_ids = list(dict.fromkeys(h["orbit_id"] for h in manifest["hypotheses"]))
        indices = [np.array([h["orbit_id"] == oid for h in manifest["hypotheses"]]) for oid in orbit_ids]
        orbit_weights = np.array([posterior[index].sum() for index in indices])
        orbit_probabilities = np.column_stack([candidates[:, index] @ posterior[index] / weight if weight > 0 else np.zeros(len(future))
                                               for index, weight in zip(indices, orbit_weights)])
        orbit_ranking = rank_exposures(orbit_weights, orbit_probabilities)
        orbit_summary = [{"orbit_id": oid, "posterior_weight": float(weight),
                          "hypothesis_ids": [h["id"] for h, select in zip(manifest["hypotheses"], index) if select]}
                         for oid, weight, index in zip(orbit_ids, orbit_weights, indices)]
    ranking = orbit_ranking if ranking_target == "orbit" else rank_exposures(posterior, candidates)
    for row in ranking:
        row["id"] = future[row.pop("index")]["id"]
    return {"schema_version": 1, "tool": "learn_null", "provenance": _provenance([path]),
            "hypothesis_ids": [h["id"] for h in manifest["hypotheses"]],
            "n_non_detections": len(observed), **result, "ranked_exposures": ranking,
            "ranking_target": ranking_target, "orbit_posterior": orbit_summary,
            "calibration_posteriors_conditional_on_hypothesis": [
                {"id": key, "alpha": a.tolist(), "beta": (b + null_counts[key]).tolist(), "observed_nulls": null_counts[key]}
                for key, (a, b) in calibrations.items()]}
