"""Deterministic injection experiment, deliberately not an astronomical discovery."""

import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from .io import run_nulls, run_photometry
from .photometry import gaussian_psf


def create_demo(directory, seed=2026):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Demo directory must be new or empty; existing files are never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    exposures = []
    # Predictions are fixed before drawing noise; controls include a static source.
    trajectories = [{"id": name, "positions": []} for name in ["injected-mover", "static-control", "wrong-rate"]]
    yy, xx = np.mgrid[:64, :64]
    truth_flux = 85.0
    for index in range(16):
        sigma = 1.05 + (index % 4) * 0.15
        noise = 8.0 + (index % 3)
        scale = 0.8 + (index % 5) * 0.1
        x, y = 18.2 + 1.4 * index, 22.4 + 0.5 * index
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [32, 32]
        wcs.wcs.cdelt = [-0.00028, 0.00028]
        wcs.wcs.crval = [150 + index * 0.0001, 12]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header = wcs.to_header()
        header["MJD-OBS"] = 60000 + index / 48
        header["BUNIT"] = "adu"
        data = 100 + truth_flux * scale * gaussian_psf(xx, yy, x, y, sigma)
        data += rng.normal(0, noise, data.shape)
        filename = f"exposure-{index:02d}.fits"
        fits.writeto(directory / filename, data, header)
        exp_id = f"e{index:02d}"
        exposures.append({"id": exp_id, "image": filename, "variance": noise**2,
                          "psf_sigma_pixels": sigma, "flux_scale": scale})
        for trajectory, (px, py) in zip(trajectories, [(x, y), (30.2, 28.4), (43.2 - index, 20.4 + 0.8 * index)]):
            # Demonstrate ICRS/WCS conversion, including the varying pointings.
            coord = wcs.pixel_to_world(px, py).icrs
            trajectory["positions"].append({"exposure_id": exp_id, "ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg)})
    manifest = {"schema_version": 1, "exposures": exposures, "trajectories": trajectories}
    null_manifest = {"schema_version": 1,
        "hypotheses": [{"id": name, "prior_weight": 1} for name in ["bright-covered", "faint-covered", "off-chip"]],
        "non_detections": [{"id": "past-frame", "expected_snr": [10, 2, 10], "threshold_snr": 5, "coverage": [1, 1, 0]}],
        "candidate_exposures": [
            {"id": "distinguishes-survivors", "detection_probability": [0.5, 0.95, 0.05]},
            {"id": "detects-everything-equally", "detection_probability": [0.95, 0.95, 0.95]}]}
    def save(name, value):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    save("images.json", manifest)
    save("nulls.json", null_manifest)
    save("truth.json", {"synthetic": True, "seed": seed, "reference_flux": truth_flux,
                        "trajectory_id": "injected-mover", "n_exposures": len(exposures)})
    forced = run_photometry(directory / "images.json")
    joint = run_photometry(directory / "images.json", joint=True)
    nulls = run_nulls(directory / "nulls.json")
    save("force-track.json", forced)
    save("dig-floor.json", joint)
    save("learn-null.json", nulls)
    source = next(t for t in joint["trajectories"] if t["id"] == "injected-mover")
    summary = {"synthetic": True, "seed": seed, "directory": str(directory.resolve()),
               "injected_reference_flux": truth_flux,
               "recovered_reference_flux": source["joint"]["flux"],
               "reference_flux_error": source["joint"]["flux_error"],
               "max_single_frame_snr": max(m["snr"] for m in source["measurements"]),
               "joint_snr": source["joint"]["snr"],
               "top_trajectory": joint["trajectories"][0]["id"],
               "posterior_weights": nulls["posterior_weights"],
               "recommended_exposure": nulls["ranked_exposures"][0]["id"]}
    save("summary.json", summary)
    return summary
