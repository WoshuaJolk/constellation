import json
import os
from pathlib import Path

from astropy.coordinates import SkyCoord
import numpy as np
import pytest

from magictools.orbits import AU_KM, C_AU_DAY, GM, OrbitSolution, astrometric_direction, propagate_family


def two_body_acceleration(time, position):
    return -GM["sun"] * position / np.linalg.norm(position)**3


def test_circular_orbit_analytic_forward_and_backward():
    speed = np.sqrt(GM["sun"])
    solution = OrbitSolution([1, 0, 0, 0, speed, 0], 60000, 59600, 60400, two_body_acceleration)
    offsets = np.array([-365, -60, 0, 60, 365])
    states = solution.at(60000 + offsets)
    phase = offsets * speed
    expected = np.column_stack([np.cos(phase), np.sin(phase), np.zeros(len(phase))])
    np.testing.assert_allclose(states[:, :3], expected, atol=2e-9)
    energy = np.sum(states[:, 3:]**2, axis=1) / 2 - GM["sun"] / np.linalg.norm(states[:, :3], axis=1)
    np.testing.assert_allclose(energy, -GM["sun"] / 2, rtol=1e-9)


def test_hyperbolic_energy_and_time_reversal():
    state = np.array([1, 0, 0, 0, 2 * np.sqrt(GM["sun"]), 0])
    solution = OrbitSolution(state, 60000, 60000, 60100, two_body_acceleration)
    end = solution.at(60100)
    energy = lambda s: np.sum(s[3:]**2) / 2 - GM["sun"] / np.linalg.norm(s[:3])
    assert energy(end) > 0
    assert energy(end) == pytest.approx(energy(state), rel=1e-9)
    backward = OrbitSolution(end, 60100, 60000, 60100, two_body_acceleration)
    np.testing.assert_allclose(backward.at(60000), state, atol=1e-9)


def test_light_time_with_exact_uniform_radial_motion():
    state = [10, 0, 0, 0.1, 0, 0]
    solution = OrbitSolution(state, 60000, 59999, 60001, lambda t, p: np.zeros(3))
    result = astrometric_direction(solution, 60000, [1, 0, 0])
    expected_delay = 9 / (C_AU_DAY + 0.1)
    assert result["light_time_days"] == pytest.approx(expected_delay, abs=1e-10)
    assert result["ra_deg"] == pytest.approx(0)
    assert result["dec_deg"] == pytest.approx(0)


def test_observer_parallax_and_bounds():
    solution = OrbitSolution([10, 0, 0, 0, 0, 0], 60000, 59999, 60001, lambda t, p: np.zeros(3))
    result = astrometric_direction(solution, 60000, [0, 1, 0])
    assert result["ra_deg"] == pytest.approx(360 - np.degrees(np.arctan(0.1)))
    with pytest.raises(ValueError, match="outside"):
        solution.at(60002)
    far = OrbitSolution([1e5, 0, 0, 0, 0, 0], 60000, 59999, 60001, lambda t, p: np.zeros(3))
    with pytest.raises(ValueError, match="outside"):
        astrometric_direction(far, 60000, [0, 0, 0])


@pytest.mark.parametrize("kernel", ["builtin", "local-jpl"])
def test_against_independent_horizons_ceres_fixture(kernel):
    if kernel == "local-jpl":
        kernel = os.environ.get("MAGICTOOLS_JPL_KERNEL")
        if not kernel:
            pytest.skip("Set MAGICTOOLS_JPL_KERNEL to a local SPK kernel for optional JPL validation")
    reference = json.loads((Path(__file__).parent / "data/ceres_horizons.json").read_text())
    first, last = reference["barycentric"]
    field, solutions = propagate_family([first["state"]], first["mjd_tdb"], [first["mjd_tdb"], last["mjd_tdb"]], kernel=kernel)
    # Actual observed discrepancy ~0.18 km over 10 days. Bound allows versions
    # of ERFA's approximate builtin ephemeris; this is not a universal accuracy claim.
    assert np.linalg.norm(solutions[0].at(last["mjd_tdb"])[:3] - last["state"][:3]) * AU_KM < (1 if kernel == "builtin" else 0.1)
    for row in reference["geocentric_light_time"]:
        prediction = astrometric_direction(solutions[0], row["mjd_tdb"], field.earth_position(row["mjd_tdb"]))
        x, y, z = row["state"][:3]
        expected = SkyCoord(np.degrees(np.arctan2(y, x)), np.degrees(np.arctan2(z, np.hypot(x, y))), unit="deg")
        actual = SkyCoord(prediction["ra_deg"], prediction["dec_deg"], unit="deg")
        assert actual.separation(expected).arcsec < (0.1 if kernel == "builtin" else 0.001)


def test_ephemeris_grid_and_integration_convergence():
    reference = json.loads((Path(__file__).parent / "data/ceres_horizons.json").read_text())
    state = reference["barycentric"][0]["state"]
    _, coarse = propagate_family([state], 60000, [60010])
    _, fine = propagate_family([state], 60000, [60010], rtol=1e-12, atol=1e-14, ephemeris_step_days=0.125)
    np.testing.assert_allclose(coarse[0].at(60010), fine[0].at(60010), atol=1e-11, rtol=0)
