import json
import subprocess
import sys

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest

from magictools.demo import create_demo
from magictools.io import read_json, run_nulls, run_photometry
from magictools.photometry import gaussian_psf


@pytest.fixture
def manifest(tmp_path):
    yy, xx = np.mgrid[:21, :21]
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [11, 11]
    wcs.wcs.crval = [359.999, 20]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    data = 100 * gaussian_psf(xx, yy, 10.3, 9.6, 1.2) + 20
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, wcs.to_header(), name="SCI"),
                  fits.ImageHDU(np.full_like(data, 4), name="VAR"),
                  fits.ImageHDU(np.zeros_like(data), name="MASK")]).writeto(tmp_path / "data.fits")
    coordinate = wcs.pixel_to_world(10.3, 9.6)
    value = {"schema_version": 1,
             "exposures": [{"id": "one", "image": "data.fits", "image_hdu": "SCI",
                            "variance": "data.fits", "variance_hdu": "VAR", "mask": "data.fits",
                            "mask_hdu": "MASK", "psf_sigma_pixels": 1.2}],
             "trajectories": [
                 {"id": "pixel", "positions": [{"exposure_id": "one", "x": 10.3, "y": 9.6}]},
                 {"id": "sky", "positions": [{"exposure_id": "one", "ra_deg": coordinate.ra.deg, "dec_deg": coordinate.dec.deg}]}]}
    path = tmp_path / "images.json"
    path.write_text(json.dumps(value))
    return path, value


def test_fits_extensions_wcs_equivalence_and_provenance(manifest):
    path, _ = manifest
    result = run_photometry(path, joint=True)
    for row in result["trajectories"]:
        assert row["joint"]["flux"] == pytest.approx(100, abs=1e-7)
        assert row["joint"]["trials"] == 2
    assert len(result["provenance"]["inputs"]) == 2
    assert all(len(p["sha256"]) == 64 for p in result["provenance"]["inputs"])


def test_off_image_is_reported_not_zero_flux(manifest):
    path, value = manifest
    value["trajectories"][0]["positions"][0]["x"] = 200
    path.write_text(json.dumps(value))
    row = next(r for r in run_photometry(path, joint=True)["trajectories"] if r["id"] == "pixel")
    assert row["joint"] is None
    assert row["status"] == "no_usable_measurements"
    assert "outside" in row["unavailable"][0]["reason"]


def test_duplicate_image_cannot_inflate_significance(manifest):
    path, value = manifest
    duplicate = {**value["exposures"][0], "id": "duplicate", "image_hdu": 1}
    value["exposures"].append(duplicate)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="same image HDU"):
        run_photometry(path, joint=True)


def test_cli_json_and_refuses_overwrite(manifest, tmp_path):
    path, _ = manifest
    command = [sys.executable, "-m", "magictools", "dig_floor", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["tool"] == "dig_floor"
    output = tmp_path / "existing.json"
    output.write_text("precious")
    result = subprocess.run(command + ["-o", str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert output.read_text() == "precious"


def test_synthetic_end_to_end_recovery(tmp_path):
    result = create_demo(tmp_path / "demo")
    assert result["top_trajectory"] == "injected-mover"
    assert result["max_single_frame_snr"] < 5
    assert result["joint_snr"] > 7
    assert abs(result["recovered_reference_flux"] - result["injected_reference_flux"]) < 3 * result["reference_flux_error"]
    assert result["posterior_weights"][0] < 1e-5
    assert result["recommended_exposure"] == "distinguishes-survivors"
    with pytest.raises(ValueError, match="never overwritten"):
        create_demo(tmp_path / "demo")


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}'])
def test_reject_ambiguous_json(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_json(path)


def test_calibrated_null_cli_schema(tmp_path):
    value = {"schema_version": 1, "hypotheses": [{"id": "a", "prior_weight": 1}, {"id": "b", "prior_weight": 1}],
             "non_detections": [{"id": "past", "detection_probability": [0.9, 0.1]}]}
    path = tmp_path / "null.json"
    path.write_text(json.dumps(value))
    result = run_nulls(path)
    np.testing.assert_allclose(result["posterior_weights"], [0.1, 0.9])
    assert result["ranked_exposures"] == []
