import numpy as np
import pytest
from scipy.integrate import solve_ivp

from magictools.galactic import GalacticPotential, integrate_stars, energy_diagnostics, KMS_TO_PC_MYR
from magictools.stellar_inputs import icrs_to_galactocentric, sample_catalogue


COMPONENTS = [
    {"type": "plummer", "mass_msun": 5e9, "scale_pc": 300},
    {"type": "miyamoto_nagai", "mass_msun": 6.8e10, "a_pc": 3000, "b_pc": 280},
    {"type": "log_halo", "speed_kms": 220, "core_pc": 12000}]


@pytest.mark.parametrize("component", COMPONENTS)
def test_acceleration_is_negative_potential_gradient(component):
    potential = GalacticPotential({"components": [component]})
    r = np.array([-8100., 430., 57.])
    gradient = []
    for axis in range(3):
        offset = np.eye(3)[axis] * 0.01
        gradient.append((potential.value(r + offset) - potential.value(r - offset)) / 0.02)
    np.testing.assert_allclose(potential.acceleration(r), -np.array(gradient), rtol=1e-7)


def test_free_motion_and_reversibility():
    potential = GalacticPotential({"components": []})
    state = np.array([100, 200, -50, 10, -20, 30.])
    result = integrate_stars(state, [0, -2, -4], potential, 0.1)
    np.testing.assert_allclose(result[-1, :3], state[:3] - 4 * state[3:] * KMS_TO_PC_MYR, atol=1e-10)
    reverse = integrate_stars(result[-1], [0, 4], potential, 0.1)
    np.testing.assert_allclose(reverse[-1], state, atol=1e-10)


def test_against_independent_dop853_and_energy_conservation():
    potential = GalacticPotential({"components": COMPONENTS})
    state = np.array([-8000, 0, 40, 80, 220, 50.])
    def rhs(t, y):
        return np.r_[y[3:] * KMS_TO_PC_MYR, potential.acceleration(y[:3]) / KMS_TO_PC_MYR]
    reference = solve_ivp(rhs, [0, -20], state, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
    result = integrate_stars(state, np.linspace(0, -20, 101), potential, 0.01)
    assert np.linalg.norm(result[-1, :3] - reference[:3]) < 0.001
    assert energy_diagnostics(result, potential)["max_scaled_energy_drift"] < 1e-7
    reverse = integrate_stars(result[-1], [0, 20], potential, 0.01)
    np.testing.assert_allclose(reverse[-1], state, atol=1e-7)


def test_correlated_phase_space_sampling():
    covariance = np.eye(6)
    covariance[0, 3] = covariance[3, 0] = 0.8
    rows = [{"id": "star", "state_pc_kms": [0] * 6, "covariance": covariance.tolist()}]
    cloud = sample_catalogue(rows, 20000, np.random.default_rng(8))[:, 0]
    assert np.corrcoef(cloud[:, 0], cloud[:, 3])[0, 1] == pytest.approx(0.8, abs=0.02)
    rows[0]["covariance"][0][0] = -1
    with pytest.raises(ValueError, match="semidefinite"):
        sample_catalogue(rows, 16, np.random.default_rng(8))


def test_icrs_distance_and_radial_velocity_are_required():
    frame = {"galcen_distance_pc": 8122, "z_sun_pc": 20.8, "galcen_v_sun_kms": [12.9, 245.6, 7.78]}
    samples = [[30, 20, 100, 0, 0, 0], [30, 20, 200, 0, 0, 0]]
    result = icrs_to_galactocentric(samples, frame)
    assert np.linalg.norm(result[1, :3] - result[0, :3]) == pytest.approx(100)
    np.testing.assert_allclose(result[:, 3:], np.tile(frame["galcen_v_sun_kms"], (2, 1)), atol=1e-10)
    with pytest.raises(ValueError):
        icrs_to_galactocentric([[30, 20, -1, 0, 0, 0]], frame)
    with pytest.raises(ValueError, match="explicit"):
        sample_catalogue([{"id": "a", "icrs_samples": samples}], 16, np.random.default_rng(1))
