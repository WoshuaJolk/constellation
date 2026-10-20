"""Reproducible physical-orbit injection and fair direct/fold timing benchmark."""

import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from .fold_kepler import ELEMENTS, kepler_positions, project_kepler, run_fold
from .orbits import GM
from .photometry import gaussian_psf, sampled_psf


def create_fold_demo(directory, seed=2026, grid_size=24, n_frames=32):
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or not 3 <= grid_size <= 40:
        raise ValueError("grid_size must be an integer in 3..40")
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or not 2 <= n_frames <= 64:
        raise ValueError("n_frames must be an integer in 2..64")
    if grid_size**3*n_frames > 4_000_000:
        raise ValueError("Demo must contain at most 4 million track/frame pairs")
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Demo directory must be new or empty; existing files are never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    epoch = 60000.
    times = epoch+np.linspace(0, 8, n_frames)
    # Explicit illustrative circular Earth, not an ephemeris of the real Earth.
    phase = .3 + np.sqrt(GM["sun"])*(times-epoch)
    observers = np.column_stack((np.cos(phase), np.sin(phase), np.zeros(n_frames)))
    truth = np.array([2.6, .12, 10., 80., 30., 40.])
    center = kepler_positions([truth], times-epoch)[0]-observers
    ra = np.degrees(np.arctan2(center[:, 1], center[:, 0])) % 360
    dec = np.degrees(np.arctan2(center[:, 2], np.hypot(center[:, 0], center[:, 1])))
    wcs_list = []
    for j in range(n_frames):
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [33, 33]
        wcs.wcs.crval = [ra[j], dec[j]]
        wcs.wcs.cdelt = [-1.5/3600, 1.5/3600]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs_list.append(wcs)
    grid = (np.arange(grid_size)-grid_size//2)/grid_size*2
    da, di, dm = np.meshgrid(grid*.002, grid*.02, grid*.01, indexing="ij")
    elements = np.tile(truth, (da.size, 1))
    elements[:, 0] += da.ravel()
    elements[:, 2] += di.ravel()
    elements[:, 5] += dm.ravel()
    injected_index = int(np.flatnonzero(np.all(elements == truth, axis=1))[0])
    projected = project_kepler([truth], epoch, times, observers, wcs_list)[0]
    exposures = []
    yy, xx = np.mgrid[:64, :64]
    sigma, flux, variance, background = 1.2, 55., 16., 100.
    ky, kx = np.mgrid[-6:7, -6:7]
    kernel = gaussian_psf(kx, ky, 0, 0, sigma)
    for j, (x, y) in enumerate(projected):
        scale = .85+.1*(j % 4)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:, 47:49] = 1
        # Inject with the documented continuous native-pixel model. This tests
        # transform fidelity, not robustness to PSF or orbit model mismatch.
        data = background + rng.normal(0, np.sqrt(variance), (64, 64))
        data += flux*scale*sampled_psf(xx, yy, x, y, kernel)
        name = f"frame-{j:03d}.fits"
        fits.HDUList([fits.PrimaryHDU(data, wcs_list[j].to_header()), fits.ImageHDU(mask, name="MASK")]).writeto(directory/name)
        exposures.append({"id": f"visit-{j}", "image": name, "variance": variance, "background": background,
                          "mask": name, "mask_hdu": "MASK", "flux_scale": scale, "psf_sigma_pixels": sigma,
                          "mjd_tdb": float(times[j]), "observer_heliocentric_au": observers[j].tolist()})
    manifest = {"schema_version": 1, "epoch_mjd_tdb": epoch, "exposures": exposures,
                "orbits": [{"id": f"orbit-{i}", **dict(zip(ELEMENTS, row.tolist()))} for i, row in enumerate(elements)],
                "cell_subdivisions": 1, "alpha": .01, "top_k": 10, "validate_direct": True, "benchmark_repeats": 3}
    def save(name, value):
        with (directory/name).open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False)+"\n")
    save("input.json", manifest)
    result = run_fold(directory/"input.json")
    save("result.json", result)
    summary = {"seed": seed, "experiment": "synthetic elliptic orbit bank, illustrative circular Earth, model-matched pixel injection",
               "n_orbits": len(elements), "n_exposures": n_frames, "injected_id": f"orbit-{injected_index}",
               "injected_reference_flux": flux, "top": result["top"][0],
               "injected_orbit_detected": any(row["id"] == f"orbit-{injected_index}" for row in result["detections"]),
               "n_detections": len(result["detections"]), "work": result["work"], "validation": result["validation"], "benchmark": result["benchmark"],
               "interpretation": "Nearby orbit hypotheses are correlated and can be indistinguishable; these are not separate source discoveries."}
    save("summary.json", summary)
    return summary
