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
    _fields(manifest, ["schema_version", "hypotheses", "non_detections"], ["candidate_exposures"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    _named_rows(manifest["hypotheses"], "hypotheses")
    for hypothesis in manifest["hypotheses"]:
        _fields(hypothesis, ["id", "prior_weight"])
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
    def log_probabilities(row):
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
    likelihood = np.array([log_probabilities(row) for row in observed]).reshape(len(observed), n)
    result = update_nulls([h["prior_weight"] for h in manifest["hypotheses"]], likelihood)
    candidates = np.array([-np.expm1(log_probabilities(row)) for row in future]).reshape(len(future), n)
    ranking = rank_exposures(result["posterior_weights"], candidates)
    for row in ranking:
        row["id"] = future[row.pop("index")]["id"]
    return {"schema_version": 1, "tool": "learn_null", "provenance": _provenance([path]),
            "hypothesis_ids": [h["id"] for h in manifest["hypotheses"]],
            "n_non_detections": len(observed), **result, "ranked_exposures": ranking}
