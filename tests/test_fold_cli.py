import json
import subprocess
import sys

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest

from magictools.fold_demo import create_fold_demo
from magictools.fold_kepler import project_kepler, kepler_positions, run_fold
from magictools.orbits import C_AU_DAY


def test_orbit_projection_uses_retarded_position_and_observer():
    elements = [[2.6, .1, 20, 30, 40, 50]]
    epoch, times = 60000, np.array([60001., 60002.])
    observers = np.array([[1, 0, 0], [1, .02, 0]])
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [20, 20]
    wcs.wcs.crval = [130, 20]
    wcs.wcs.cdelt = [-.01, .01]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    xy = project_kepler(elements, epoch, times, observers, [wcs, wcs])[0]
    for j, t in enumerate(times):
        offset = t-epoch
        for _ in range(30):
            delta = kepler_positions(elements, [offset])[0, 0]-observers[j]
            offset = t-epoch-np.linalg.norm(delta)/C_AU_DAY
        ra = np.degrees(np.arctan2(delta[1], delta[0])) % 360
        dec = np.degrees(np.arctan2(delta[2], np.hypot(delta[0], delta[1])))
        np.testing.assert_allclose(xy[j], wcs.world_to_pixel_values(ra, dec), atol=1e-8)


def test_small_physical_orbit_demo_and_cli(tmp_path):
    summary = create_fold_demo(tmp_path/"demo", grid_size=6, n_frames=6)
    assert summary["injected_orbit_detected"]
    assert summary["validation"]["identical_top_and_detections"]
    assert summary["benchmark"]["repeats"] == 3
    path = tmp_path/"demo"/"input.json"
    command = [sys.executable, "-m", "magictools", "fold_kepler", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = json.loads(result.stdout)
    assert result["tool"] == "fold_kepler"
    assert len(result["provenance"]["inputs"]) == 7
    assert all(len(p["sha256"]) == 64 for p in result["provenance"]["inputs"])
    output = tmp_path/"precious.json"
    output.write_text("keep")
    failed = subprocess.run(command+["-o", str(output)], capture_output=True, text=True)
    assert failed.returncode == 2
    assert output.read_text() == "keep"


def test_pixel_bank_schema_and_duplicate_images(tmp_path):
    fits.writeto(tmp_path/"image.fits", np.ones((12, 12)))
    value = {"schema_version": 1,
             "exposures": [{"id": "a", "image": "image.fits", "variance": 1,
                            "background": 1, "psf_sigma_pixels": 1}],
             "trajectories": [{"id": "one", "xy": [[4.1, 4.2]]}, {"id": "off", "xy": [[-1, -1]]}],
             "validate_direct": True}
    path = tmp_path/"input.json"
    path.write_text(json.dumps(value))
    report = run_fold(path)
    assert len(report["top"]) == 1
    assert report["top"][0]["snr"] == 0
    value["exposures"].append({**value["exposures"][0], "id": "b"})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="same image HDU"):
        run_fold(path)
