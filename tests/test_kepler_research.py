"""Independent checks for research attempts; none assert the full goal is met."""

import importlib.util

import numpy as np
import pytest

from magictools.orbits import GM
from research.kepler_transform.universal import propagate
from research.kepler_transform.rotation import (RotationTransform, DirectionalRotationTransform,
    evaluate_sky, evaluate_directional, real_coefficients, rotations)
from research.kepler_transform.factorization import global_rank, BlockFactorization
from research.kepler_transform.experiments import (independent_dynamics_check, inverse_counterexample,
    bandwidth_experiment, directional_physical_experiment)


def require_nufft(backend):
    if backend == "nufft" and importlib.util.find_spec("finufft") is None:
        pytest.skip("Install the optional research extra for NUFFT tests")


def test_universal_dynamics_all_energy_classes_against_independent_ode():
    for row in independent_dynamics_check():
        assert row["max_position_error_au"] < 1e-9
        assert row["max_velocity_error_au_per_day"] < 1e-11


def test_universal_energy_angular_momentum_and_time_reversal():
    mu = GM["sun"]
    states = np.array([[2., 0, 0, -.002, factor*np.sqrt(mu/2), .001] for factor in [.7, 1.4, 2.]])
    result = propagate(states, [-100, 0, 100])
    energy = np.sum(result[..., 3:]**2, axis=2)/2-mu/np.linalg.norm(result[..., :3], axis=2)
    np.testing.assert_allclose(energy, np.repeat(energy[:, 1, None], 3, axis=1), atol=2e-16)
    momentum = np.cross(result[..., :3], result[..., 3:])
    np.testing.assert_allclose(momentum, np.repeat(momentum[:, 1, None], 3, axis=1), atol=2e-14)
    np.testing.assert_allclose(propagate(result[:, 2], [-100])[:, 0], states, atol=1e-11)


@pytest.mark.parametrize("backend", ["direct", "nufft"])
@pytest.mark.parametrize("degree", [2, 5])
def test_rotation_identity_against_rotated_cartesian_sky(backend, degree):
    require_nufft(backend)
    rng = np.random.default_rng(40)
    phases = rng.uniform(-10, 10, 11)
    data = real_coefficients(rng, len(phases), degree)
    plan = RotationTransform(phases, degree, backend)
    output = plan.forward(data)
    unit = np.column_stack((np.cos(phases), np.sin(phases), np.zeros(len(phases))))
    direction = np.einsum("rij,tj->rti", plan.grid_rotations().reshape(-1, 3, 3), unit)
    np.testing.assert_allclose(output.ravel(), evaluate_sky(data, direction), atol=1e-9)
    assert np.max(np.abs(output.imag)) < 1e-9
    for actual, expected in zip(plan.inverse_wigner(output), plan.coefficients(data)):
        np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("backend", ["direct", "nufft"])
def test_complex_rotation_adjoint_is_not_an_inverse(backend):
    require_nufft(backend)
    rng = np.random.default_rng(41)
    plan = RotationTransform(rng.uniform(-np.pi, np.pi, 20), 3, backend)
    x = rng.normal(size=(20, 16))+1j*rng.normal(size=(20, 16))
    y = rng.normal(size=(4, 7, 7))+1j*rng.normal(size=(4, 7, 7))
    np.testing.assert_allclose(np.vdot(plan.forward(x), y), np.vdot(x, plan.adjoint(y)), rtol=1e-10)
    # Constant angular responses with zero sum over frames are a null mode.
    null = np.zeros_like(x)
    null[0, 0], null[1, 0] = 1, -1
    np.testing.assert_allclose(plan.forward(null), 0, atol=1e-10)


@pytest.mark.parametrize("backend", ["direct", "nufft"])
def test_directional_composition_and_adjoint(backend):
    require_nufft(backend)
    rng = np.random.default_rng(42)
    phases, degree, spin = rng.uniform(-np.pi, np.pi, 9), 4, 2
    plan = DirectionalRotationTransform(phases, degree, spin, backend)
    data = rng.normal(size=(9, 25, 5))+1j*rng.normal(size=(9, 25, 5))
    for s in range(-spin, spin+1):
        data[:, :s*s, s+spin] = 0
    output = plan.forward(data)
    q = plan.grid_rotations().reshape(-1, 3, 3)[:, None] @ rotations(phases, np.pi/2, 0)[None]
    np.testing.assert_allclose(output.ravel(), evaluate_directional(data, q), atol=1e-8)
    target = rng.normal(size=output.shape)+1j*rng.normal(size=output.shape)
    np.testing.assert_allclose(np.vdot(output, target), np.vdot(data, plan.adjoint(target)), rtol=1e-10)


def test_directional_spin_zero_agrees_with_spherical_case():
    rng = np.random.default_rng(43)
    phases, degree = rng.uniform(-3, 3, 7), 4
    data = real_coefficients(rng, 7, degree)
    directional = data[:, :, None].copy()
    for l in range(degree+1):
        directional[:, l*l:(l+1)**2] *= np.sqrt((2*l+1)/(4*np.pi))
    expected = RotationTransform(phases, degree).forward(data)
    actual = DirectionalRotationTransform(phases, degree, 0).forward(directional)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_directional_repair_retains_construction_work_and_physical_error():
    require_nufft("nufft")
    result = directional_physical_experiment()
    assert not result["spin_sweep"][0]["passes_1e_minus_8_target"]
    assert result["spin_sweep"][1]["passes_1e_minus_8_target"]
    # A correct low-degree experiment does not erase its O(NT) construction.
    assert result["response_construction_orbit_frame_evaluations"] == result["direct_orbit_frame_evaluations"]


def test_physical_global_rank_obstruction_and_positive_control():
    clustered, separated = global_rank(side=8)
    assert clustered["rank_for_99_percent_template_energy"] == 1
    assert separated["rank_for_99_percent_template_energy"] >= .99*separated["orbits"]
    assert separated["gershgorin_eigenvalue_lower_bound"] > .99999


def test_block_compression_error_bound_and_adjoint():
    rng = np.random.default_rng(44)
    left = rng.normal(size=(64, 1))+1j*rng.normal(size=(64, 1))
    right = rng.normal(size=(1, 64))+1j*rng.normal(size=(1, 64))
    matrix = left @ right+1e-10*rng.normal(size=(64, 64))
    plan = BlockFactorization(matrix, rng.normal(size=(64, 6)), rng.normal(size=(64, 3)))
    x, y = rng.normal(size=(2, 64))
    error = np.linalg.norm(plan.apply(x)-matrix @ x)
    bound = plan.report["relative_frobenius_error_bound"]*np.linalg.norm(matrix)*np.linalg.norm(x)
    assert error <= bound+1e-11
    np.testing.assert_allclose(np.vdot(plan.apply(x), y), np.vdot(x, plan.apply(y, adjoint=True)), rtol=1e-11)
    assert plan.report["storage_ratio"] < .5
    assert plan.report["construction_kernel_entries"] == matrix.size


def test_inverse_counterexample_uses_full_likelihood_not_profiled_score():
    result = inverse_counterexample()
    assert result["input_euclidean_distance"] > 1
    assert result["maximum_full_gaussian_log_likelihood_difference"] < 1e-10


def test_astronomical_bandwidth_is_not_toy_bandwidth():
    result = bandwidth_experiment()
    assert result["degree_for_99_percent_template_energy"] > 400000
    assert result["snr_retention_at_degree_12"] < 1e-3
    assert result["dense_orientation_grid_values_at_99_percent"] > 1e17


def test_research_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        propagate([[0, 0, 0, 0, 0, 0]], [0])
    with pytest.raises(ValueError):
        RotationTransform([0], 40)
    with pytest.raises(ValueError):
        DirectionalRotationTransform([0], 2, 3)
