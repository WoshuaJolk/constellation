"""Orbit-family coverage and noise-based sensitivity of a supplied FITS inventory."""

from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from .io import _fields, _fits_plane, _named_rows, _provenance, read_json
from .nulls import _weights, gaussian_log_null
from .orbits import astrometric_direction, propagate_family
from .photometry import forced_photometry


def run_cover(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    _fields(manifest, ["schema_version", "epoch_mjd_tdb", "orbits", "exposures"], ["dynamics", "threshold_snr"])
    if manifest["schema_version"] != 1:
        raise ValueError("Only schema_version 1 is supported")
    _named_rows(manifest["orbits"], "orbits")
    _named_rows(manifest["exposures"], "exposures")
    for orbit in manifest["orbits"]:
        _fields(orbit, ["id", "state_au_au_per_day", "prior_weight", "reference_flux"])
        if not np.isfinite(orbit["reference_flux"]) or orbit["reference_flux"] < 0:
            raise ValueError("Reference flux must be finite and nonnegative")
    weights = _weights([row["prior_weight"] for row in manifest["orbits"]])
    threshold = manifest.get("threshold_snr", 5.0)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold_snr must be positive and finite")
    supplied_options = manifest.get("dynamics", {})
    _fields(supplied_options, [], ["kernel", "light_time_margin_days", "rtol", "atol", "ephemeris_step_days"])
    options = {"kernel": "builtin", "light_time_margin_days": 10, "rtol": 1e-11,
               "atol": 1e-13, "ephemeris_step_days": 0.25, **supplied_options}
    files = [path]
    if options.get("kernel", "builtin") != "builtin":
        options["kernel"] = str((path.parent / options["kernel"]).resolve(strict=True))
        files.append(options["kernel"])
    samples = []
    for exposure in manifest["exposures"]:
        _fields(exposure, ["id", "image", "variance", "psf_sigma_pixels", "mjd_tdb"],
                ["image_hdu", "variance_hdu", "mask", "mask_hdu", "flux_scale", "exposure_seconds", "observer"])
        mid, duration = exposure["mjd_tdb"], exposure.get("exposure_seconds", 0)
        if not np.all(np.isfinite([mid, duration])) or duration < 0:
            raise ValueError("Exposure midpoint must be finite and duration nonnegative")
        samples.extend([mid - duration / 172800, mid, mid + duration / 172800])
        if "observer" in exposure:
            observer = exposure["observer"]
            _fields(observer, ["position_au", "velocity_au_per_day"])
            for values in observer.values():
                if np.shape(values) != (3,) or not np.all(np.isfinite(values)):
                    raise ValueError("Observer position and velocity must be finite 3-vectors")
    field, solutions = propagate_family([row["state_au_au_per_day"] for row in manifest["orbits"]],
                                       manifest["epoch_mjd_tdb"], samples, **options)
    photometry = {"schema_version": 1, "exposures": [], "trajectories": [
        {"id": row["id"], "positions": []} for row in manifest["orbits"]]}
    planning = {"schema_version": 1, "hypotheses": [
        {"id": row["id"], "prior_weight": float(weight)} for row, weight in zip(manifest["orbits"], weights)],
        "non_detections": [], "candidate_exposures": []}
    exposure_reports, image_sources = [], set()
    for exposure in manifest["exposures"]:
        science_path = (path.parent / exposure["image"]).resolve()
        with fits.open(science_path, memmap=False) as hdus:
            index = hdus.index_of(exposure.get("image_hdu", 0))
            if (science_path, index) in image_sources:
                raise ValueError("The same image HDU cannot count as independent exposures twice")
            image_sources.add((science_path, index))
            data = np.array(hdus[index].data, dtype=float)
            wcs = WCS(hdus[index].header).celestial
        if data.ndim != 2 or min(data.shape) < 2 or not wcs.has_celestial:
            raise ValueError("Each exposure needs a 2D science image with celestial WCS")
        files.append(science_path)
        clean = {key: value for key, value in exposure.items() if key not in {"mjd_tdb", "exposure_seconds", "observer"}}
        clean["image"] = str(science_path)
        variance = exposure["variance"]
        if isinstance(variance, str):
            variance_path = (path.parent / variance).resolve()
            variance = _fits_plane(variance_path, exposure.get("variance_hdu", 0))
            clean["variance"] = str(variance_path)
            files.append(variance_path)
            if variance.shape != data.shape:
                raise ValueError("Variance image shape must match science image")
        elif not np.isfinite(variance) or variance <= 0:
            raise ValueError("Scalar variance must be finite and positive")
        invalid = ~np.isfinite(data) | ~np.isfinite(variance) | (np.asarray(variance) <= 0)
        if "mask" in exposure:
            mask_path = (path.parent / exposure["mask"]).resolve()
            mask = _fits_plane(mask_path, exposure.get("mask_hdu", 0))
            if mask.shape != data.shape:
                raise ValueError("Mask shape must match science image")
            invalid |= mask != 0
            clean["mask"] = str(mask_path)
            files.append(mask_path)
        sigma, scale = exposure["psf_sigma_pixels"], exposure.get("flux_scale", 1)
        if not np.all(np.isfinite([sigma, scale])) or sigma <= 0 or scale <= 0:
            raise ValueError("PSF sigma and flux scale must be finite and positive")
        photometry["exposures"].append(clean)
        midpoint, duration = exposure["mjd_tdb"], exposure.get("exposure_seconds", 0)
        times = [midpoint - duration / 172800, midpoint, midpoint + duration / 172800]
        if "observer" in exposure:
            observer = exposure["observer"]
            observers = [np.array(observer["position_au"]) + (time - midpoint) * np.array(observer["velocity_au_per_day"]) for time in times]
        else:
            observers = field.earth_position(times)
        height, width = data.shape
        blank = np.zeros(data.shape)
        rows, expected_snr, coverage = [], [], []
        for i, (orbit, solution) in enumerate(zip(manifest["orbits"], solutions)):
            directions = [astrometric_direction(solution, time, observer) for time, observer in zip(times, observers)]
            coordinates = SkyCoord([row["ra_deg"] for row in directions], [row["dec_deg"] for row in directions], unit="deg", frame="icrs")
            px, py = wcs.world_to_pixel(coordinates)
            x, y = float(px[1]), float(py[1])
            finite = np.all(np.isfinite([*px, *py]))
            on_chip = bool(finite and -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5)
            eligible = on_chip and not invalid[int(np.floor(y + 0.5)), int(np.floor(x + 0.5))]
            row = {"orbit_id": orbit["id"], **directions[1], "on_chip": on_chip,
                   "usable": False, "x": x if np.isfinite(x) else None, "y": y if np.isfinite(y) else None}
            snr = 0.0
            if eligible:
                dx, dy = float(px[2] - px[0]), float(py[2] - py[0])
                # Linear-trail fit is only allowed if midpoint curvature is small.
                curvature = float(np.hypot(x - (px[0] + px[2]) / 2, y - (py[0] + py[2]) / 2))
                row.update(trail_dx=dx, trail_dy=dy, midpoint_curvature_pixels=curvature)
                try:
                    if curvature > 0.05:
                        raise ValueError("Trajectory curvature exceeds 0.05 pixel; use shorter exposures")
                    measurement = forced_photometry(blank, variance, x, y, sigma, invalid, dx, dy)
                except ValueError as error:
                    row["reason"] = str(error)
                else:
                    snr = orbit["reference_flux"] * scale / measurement.flux_error
                    row.update(usable=True, flux_error=measurement.flux_error, psf_fraction=measurement.psf_fraction)
                    photometry["trajectories"][i]["positions"].append({"exposure_id": exposure["id"],
                        "x": x, "y": y, "trail_dx": dx, "trail_dy": dy})
            else:
                row["reason"] = "Outside detector projection" if not on_chip else "Centroid pixel masked or invalid"
            row["expected_snr"] = float(snr)
            expected_snr.append(float(snr))
            coverage.append(int(row["usable"]))
            rows.append(row)
        probabilities = -np.expm1(gaussian_log_null(expected_snr, threshold, coverage))
        exposure_reports.append({"id": exposure["id"], "mjd_tdb": midpoint,
            "weighted_on_chip_fraction": float(weights @ [row["on_chip"] for row in rows]),
            "weighted_usable_fraction": float(weights @ coverage),
            "weighted_gaussian_detection_probability": float(weights @ probabilities), "members": rows})
        planning["candidate_exposures"].append({"id": exposure["id"], "expected_snr": expected_snr,
                                              "threshold_snr": threshold, "coverage": coverage})
    return {"schema_version": 1, "tool": "map_cover", "provenance": _provenance(files),
            "dynamics": {"kernel": field.kernel, "force_model": "Newtonian Sun + eight planets + Moon; massless targets",
                         "frame": "barycentric ICRS", "time_scale": "MJD TDB", "options": options,
                         "integration_evaluations": [solution.nfev for solution in solutions]},
            "scope": "Supplied local FITS inventory; geometric astrometric directions with light time; Gaussian sensitivity",
            "exposures": exposure_reports, "photometry_manifest": photometry, "planning_manifest": planning}
