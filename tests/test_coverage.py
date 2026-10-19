import json
import subprocess
import sys

from astropy.io import fits
import numpy as np
import pytest

from magictools.coverage import run_cover
from magictools.io import run_nulls, run_photometry
from magictools.orbit_demo import create_orbit_demo


@pytest.fixture
def experiment(tmp_path):
    directory = tmp_path / "experiment"
    summary = create_orbit_demo(directory)
    return directory, summary


def test_orbit_to_pixels_to_joint_likelihood_and_controls(experiment):
    directory, summary = experiment
    assert summary["top_trajectory"] == "injected-orbit"
    assert abs(summary["recovered_flux"] - 100) < 3 * summary["flux_error"]
    assert summary["joint_snr"] > 8
    assert summary["artifact_control_preferences"] == {"moving-source": "moving_source", "cosmic-ray": "cosmic_ray_spike"}
    assert set(summary["coverage_fractions"]) == {0.5, 0.75}
    report = json.loads((directory / "coverage.json").read_text())
    assert report["planning_manifest"]["non_detections"] == []
    for exposure in report["exposures"]:
        off = next(row for row in exposure["members"] if row["orbit_id"] == "off-field")
        assert not off["usable"] and not off["on_chip"]
        assert off["expected_snr"] == 0
    photometry = run_photometry(directory / "photometry-input.json", joint=True)
    assert all(row["joint"]["trials"] == 4 for row in photometry["trajectories"] if row["joint"])
    off = next(row for row in photometry["trajectories"] if row["id"] == "off-field")
    assert off["joint"] is None
    plan = run_nulls(directory / "planning-input.json")
    np.testing.assert_allclose(plan["posterior_weights"], [0.25] * 4)


def test_sensitivity_does_not_use_measured_flux(experiment):
    directory, _ = experiment
    before = json.loads((directory / "coverage.json").read_text())
    with fits.open(directory / "frame-0.fits", mode="update") as hdus:
        hdus["SCI"].data *= 1000
    after = run_cover(directory / "orbits.json")
    assert before["exposures"] == after["exposures"]


def test_cli_exports_are_portable_and_do_not_overwrite(experiment, tmp_path):
    directory, _ = experiment
    photo, plan, report = [tmp_path / filename for filename in ["photo.json", "plan.json", "report.json"]]
    command = [sys.executable, "-m", "magictools", "map_cover", str(directory / "orbits.json"),
               "--photometry-output", str(photo), "--planning-output", str(plan), "-o", str(report)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert run_photometry(photo, joint=True)["trajectories"][0]["id"] == "injected-orbit"
    assert len(run_nulls(plan)["ranked_exposures"]) == 8
    contents = report.read_bytes()
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert report.read_bytes() == contents
    stamp = subprocess.run([sys.executable, "-m", "magictools", "weigh_stamp", str(directory / "cosmic-ray-input.json")], capture_output=True, text=True)
    assert stamp.returncode == 0, stamp.stderr
    assert json.loads(stamp.stdout)["preferred_model"] == "cosmic_ray_spike"


def test_completely_masked_frame_remains_uninformative(experiment):
    directory, _ = experiment
    with fits.open(directory / "frame-0.fits", mode="update") as hdus:
        hdus["MASK"].data[:] = 1
    report = run_cover(directory / "orbits.json")
    assert report["exposures"][0]["weighted_usable_fraction"] == 0
    assert report["exposures"][0]["weighted_gaussian_detection_probability"] == 0


def test_zero_motion_degeneracy_is_explicit(experiment):
    from magictools.stamps import run_stamp
    directory, _ = experiment
    path = directory / "moving-source-input.json"
    value = json.loads(path.read_text())
    value.update(trail_dx=0, trail_dy=0)
    path.write_text(json.dumps(value))
    result = run_stamp(path)
    assert result["zero_motion_degeneracy"]
    assert result["models"]["stationary_point"]["log_bayes_factor_moving_vs_model"] == 0
