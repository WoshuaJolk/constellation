"""Orbit-driven synthetic images plus known artifact controls; no discovery claims."""

import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from .coverage import run_cover
from .io import run_nulls, run_photometry
from .orbits import astrometric_direction, propagate_family
from .photometry import trailed_psf
from .stamps import run_stamp


def create_orbit_demo(directory, seed=2026):
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Demo directory must be new or empty; existing files are never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Ceres barycentric ICRF state at MJD 60000 TDB (JPL Horizons, solution #48).
    # The images and family perturbations below are synthetic, not Ceres data.
    state = [-2.512136901356719, 0.05592387543738495, 0.5361712539349595,
             -0.001470661278658604, -0.01011189067455166, -0.004469093415014039]
    times = 60000 + np.arange(8) * 0.25
    duration = 120.0
    field, solutions = propagate_family([state], 60000, [times.min() - duration / 172800, times.max() + duration / 172800])
    yy, xx = np.mgrid[:51, :51]
    exposures = []
    for i, midpoint in enumerate(times):
        direction = astrometric_direction(solutions[0], midpoint, field.earth_position(midpoint))
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [26 + (i % 3 - 1) * 3, 26]
        wcs.wcs.crval = [direction["ra_deg"], direction["dec_deg"]]
        wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        points = [astrometric_direction(solutions[0], t, field.earth_position(t))
                  for t in [midpoint - duration / 172800, midpoint, midpoint + duration / 172800]]
        px, py = wcs.world_to_pixel_values([p["ra_deg"] for p in points], [p["dec_deg"] for p in points])
        sigma, noise, scale = 1.1 + i % 3 * 0.1, 5 + i % 2, 0.8 + i % 4 * 0.1
        data = 70 + 100 * scale * trailed_psf(xx, yy, px[1], py[1], sigma, px[2] - px[0], py[2] - py[0])
        data += rng.normal(0, noise, data.shape)
        mask = np.zeros(data.shape, dtype=np.uint8)
        mask[:, 38:41] = 1  # explicit synthetic detector gap
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, wcs.to_header(), name="SCI"),
                      fits.ImageHDU(mask, name="MASK")]).writeto(directory / f"frame-{i}.fits")
        exposures.append({"id": f"visit-{i}", "image": f"frame-{i}.fits", "image_hdu": "SCI",
            "mask": f"frame-{i}.fits", "mask_hdu": "MASK", "variance": noise**2,
            "psf_sigma_pixels": sigma, "flux_scale": scale, "mjd_tdb": float(midpoint), "exposure_seconds": duration})
    orbits = []
    for name, delta in [("injected-orbit", 0), ("nearby-alternative", 0.00012), ("gap-alternative", -0.00012), ("off-field", 0.003)]:
        value = state.copy()
        value[1] += delta
        orbits.append({"id": name, "state_au_au_per_day": value, "prior_weight": 1, "reference_flux": 100})
    manifest = {"schema_version": 1, "epoch_mjd_tdb": 60000, "orbits": orbits, "exposures": exposures}
    def save(name, value):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    save("orbits.json", manifest)
    cover = run_cover(directory / "orbits.json")
    save("coverage.json", cover)
    save("photometry-input.json", cover["photometry_manifest"])
    save("planning-input.json", cover["planning_manifest"])
    joint = run_photometry(directory / "photometry-input.json", joint=True)
    planning = run_nulls(directory / "planning-input.json")
    save("joint.json", joint)
    save("planning.json", planning)
    sy, sx = np.mgrid[:25, :25]
    base = 20 + rng.normal(0, 2, (25, 25))
    mover = base + 400 * trailed_psf(sx, sy, 12.2, 11.8, 1.1, 8, 2)
    cosmic = base.copy()
    cosmic[12, 12] += 400
    artifact_results = {}
    for name, image in [("moving-source", mover), ("cosmic-ray", cosmic)]:
        fits.writeto(directory / f"{name}.fits", image)
        stamp_manifest = {"schema_version": 1, "image": f"{name}.fits", "variance": 4,
                          "x": 12.2, "y": 11.8, "psf_sigma_pixels": 1.1,
                          "trail_dx": 8, "trail_dy": 2, "flux_prior_sigma": 500}
        save(f"{name}-input.json", stamp_manifest)
        result = run_stamp(directory / f"{name}-input.json")
        save(f"{name}-result.json", result)
        artifact_results[name] = result["preferred_model"]
    source = next(row for row in joint["trajectories"] if row["id"] == "injected-orbit")
    summary = {"synthetic": True, "seed": seed, "directory": str(directory),
               "state_seed": "JPL Horizons Ceres at MJD 60000 TDB; images are simulated",
               "n_orbit_hypotheses": len(orbits), "n_exposures": len(exposures),
               "top_trajectory": joint["trajectories"][0]["id"],
               "injected_reference_flux": 100, "recovered_flux": source["joint"]["flux"],
               "flux_error": source["joint"]["flux_error"], "joint_snr": source["joint"]["snr"],
               "coverage_fractions": [row["weighted_usable_fraction"] for row in cover["exposures"]],
               "artifact_control_preferences": artifact_results}
    save("summary.json", summary)
    return summary
