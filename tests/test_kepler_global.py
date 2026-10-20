import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("finufft")

from research.kepler_transform.source_butterfly import (
    SourceButterfly, cardinal, conditional_error_bound, contract, node_rule,
    theoretical_counts,
)


def polynomial_phase(x, y):
    t, kx, ky = y.T
    px = x[:, 0, None] + x[:, 2, None]*t + x[:, 4, None]*t*t
    py = x[:, 1, None] + x[:, 3, None]*t + x[:, 5, None]*t*t
    return .3 * (px*kx + py*ky)


def test_tensor_contraction_and_reverse_are_adjoint():
    rng = np.random.default_rng(28)
    matrices = [rng.normal(size=(3, 3)) for _ in range(3)]
    a, b = rng.normal(size=(2, 3, 3, 3)) + 1j*rng.normal(size=(2, 3, 3, 3))
    assert abs(np.vdot(contract(a, matrices), b)-np.vdot(a, contract(b, matrices, adjoint=True))) < 1e-12


def test_cardinal_reproduces_polynomials_including_nodes_and_boundaries():
    nodes, weights = node_rule(8)
    x = np.r_[nodes, np.linspace(-1, 1, 33)]
    basis = cardinal(x, nodes, weights)
    for power in range(8):
        np.testing.assert_allclose(basis @ nodes**power, x**power, atol=2e-15)


@pytest.mark.parametrize("depth", [0, 1])
def test_continuous_six_dimensional_field_and_adjoint(depth):
    rng = np.random.default_rng(29)
    sources = rng.uniform(-1, 1, (40, 3))
    sources[:2] = [-1, -1, -1], [1, 1, 1]
    plan = SourceButterfly(polynomial_phase, sources, output_dimension=6, depth=depth, order=14)
    data = rng.normal(size=40) + 1j*rng.normal(size=40)
    field = plan.transform(data)
    # Queries are chosen only after constructing the entire continuous field.
    queries = rng.uniform(-1, 1, (31, 6))
    queries[:3] = np.zeros(6), -np.ones(6), np.ones(6)
    matrix = np.exp(1j*polynomial_phase(queries, sources))
    actual = field.evaluate(queries)
    np.testing.assert_allclose(actual, matrix @ data, atol=2e-7, rtol=0)
    values = rng.normal(size=31) + 1j*rng.normal(size=31)
    adjoint = plan.adjoint(queries, values)
    assert abs(np.vdot(actual, values)-np.vdot(data, adjoint)) < 1e-11
    np.testing.assert_allclose(adjoint, matrix.conj().T @ values, atol=2e-7, rtol=0)
    assert len(field.coefficients) == 2**(6*depth)


def test_nested_levels_reproduce_oscillatory_fourier_kernel():
    rng = np.random.default_rng(30)
    sources = rng.uniform(-1, 1, (91, 1))
    phase = lambda x, y: 8*x @ y.T
    plan = SourceButterfly(phase, sources, output_dimension=1, depth=3, order=12)
    data = rng.normal(size=91)+1j*rng.normal(size=91)
    queries = rng.uniform(-1, 1, (71, 1))
    actual = plan.transform(data).evaluate(queries)
    np.testing.assert_allclose(actual, np.exp(1j*phase(queries, sources)) @ data, atol=2e-11)
    values = rng.normal(size=71)+1j*rng.normal(size=71)
    assert abs(np.vdot(actual, values)-np.vdot(data, plan.adjoint(queries, values))) < 2e-11


def test_budget_failure_precedes_kernel_calls_and_preserves_domain():
    def phase(*args):
        raise AssertionError("Resource refusal must precede phase construction")
    with pytest.raises(ValueError, match="domain not reduced"):
        SourceButterfly(phase, np.zeros((2, 3)), output_dimension=6, depth=4,
                        order=8, max_gate_bytes=1_000_000)


def test_repeated_sources_have_a_real_nullspace():
    rng = np.random.default_rng(31)
    sources = rng.uniform(-1, 1, (8, 3))
    sources[1] = sources[0]
    plan = SourceButterfly(polynomial_phase, sources, output_dimension=6, depth=1, order=5)
    data = np.zeros(8)
    data[0], data[1] = 1, -1
    np.testing.assert_allclose(plan.transform(data).evaluate(rng.uniform(-1, 1, (30, 6))), 0, atol=1e-14)


def test_full_tree_count_and_conditional_bound_are_explicit():
    result = theoretical_counts(6, 3, 4, 8)
    assert result["continuous_output_leaves"] == 16_777_216
    assert result["full_tree_pairs"] == 2**12*(2**15-1)//7
    bound = conditional_error_bound(2, 16, 3, 2., 1.)
    better = conditional_error_bound(2, 64, 3, 2., 1.)
    assert better["log_absolute_kernel_error_bound"] < bound["log_absolute_kernel_error_bound"]
    assert not better["hypotheses_verified"]


def test_retarded_global_phase_matches_independent_all_conic_solver():
    pytest.importorskip("numba")
    from research.kepler_transform.chart_experiments import setup
    from research.kepler_transform.factorization import ARCSEC
    from research.kepler_transform.global_kepler import ContinuousKeplerPhase
    from magictools.orbits import GM
    rng = np.random.default_rng(32)
    chart, center, widths, _, _ = setup("stress")
    phase = ContinuousKeplerPhase(chart, center, widths, time_half_span=4,
                                  frequency_scale=1/ARCSEC)
    queries = rng.uniform(-.95, .95, (32, 6))
    sources = rng.uniform(-1, 1, (15, 3))
    np.testing.assert_allclose(phase(queries, sources), phase.independent(queries, sources),
                               atol=2e-9, rtol=2e-12)
    states = chart.states(center+queries*widths)
    energy = np.sum(states[:, 3:]**2, axis=1)/2-GM["sun"]/np.linalg.norm(states[:, :3], axis=1)
    assert np.min(energy) < 0 < np.max(energy)


@pytest.mark.parametrize("kwargs", [{"order": 1}, {"depth": -1}, {"output_dimension": 0}])
def test_invalid_hierarchy_inputs(kwargs):
    options = dict(output_dimension=2, order=4, depth=1)
    options.update(kwargs)
    with pytest.raises(ValueError):
        SourceButterfly(lambda x, y: x @ y.T, np.zeros((3, 2)), **options)


def test_sundman_identity_all_energy_signs_and_nonshared_clock():
    from research.kepler_transform.sundman import regularized
    from research.kepler_transform.universal import propagate
    from magictools.orbits import GM
    mu = GM["sun"]
    states = np.array([[r, 0, 0, .001, scale*np.sqrt(2*mu/r), .0002]
                       for r in [1.5, 3., 20.] for scale in [.6, 1., 1.4]])
    values, times, info = regularized(states, np.linspace(-.7, .7, 15))
    independent = propagate(states, times)
    np.testing.assert_allclose(values, independent, atol=3e-13, rtol=2e-12)
    np.testing.assert_allclose(np.linalg.norm(values[..., :3], axis=2), info["dt_ds"], atol=3e-13)
    assert info["twice_energy"].min() < 0 < info["twice_energy"].max()
    assert np.ptp(times[:, -1]) > 10
    # The parabolic formula has no division by energy.
    parabolic = np.array([[2., 0, 0, 0, np.sqrt(mu), 0]])
    values, times, _ = regularized(parabolic, np.array([-1., 0., 1.]))
    np.testing.assert_allclose(values, propagate(parabolic, times), atol=3e-13)


def test_finite_feature_counterexample_is_an_actual_retarded_kepler_family():
    from magictools.orbits import GM, C_AU_DAY
    from research.kepler_transform.universal import apparent
    radii = np.array([1.5, 3., 10.])
    omega = np.sqrt(GM["sun"]/radii**3)
    reference_phase = omega*radii/C_AU_DAY
    states = np.column_stack((radii*np.cos(reference_phase), radii*np.sin(reference_phase),
        np.zeros(3), -radii*omega*np.sin(reference_phase), radii*omega*np.cos(reference_phase),
        np.zeros(3)))
    times = np.linspace(-4., 4., 13)
    emitted, _ = apparent(states, times, np.zeros((len(times), 3)))
    np.testing.assert_allclose(emitted[:, :, 1]/emitted[:, :, 0],
                               np.tan(omega[:, None]*times), atol=2e-14, rtol=0)


@pytest.mark.parametrize("orders", [(7,), (6, 9), (5, 7, 8)])
def test_fast_chebyshev_moments_and_adjoint_match_dense(orders):
    from research.kepler_transform.chebyshev_moments import CosineMoments, DenseMoments
    rng = np.random.default_rng(34)
    points = rng.uniform(-1, 1, (73, len(orders)))
    points[0], points[1] = -1, 1
    data = rng.normal(size=73)+1j*rng.normal(size=73)
    coefficients = rng.normal(size=orders)+1j*rng.normal(size=orders)
    fast = CosineMoments(points, orders)
    dense = DenseMoments(points, [node_rule(p) for p in orders], orders)
    np.testing.assert_allclose(fast.forward(data), dense.forward(data), atol=2e-11)
    np.testing.assert_allclose(fast.adjoint(coefficients), dense.adjoint(coefficients), atol=2e-11)
    assert abs(np.vdot(fast.forward(data), coefficients)-np.vdot(data, fast.adjoint(coefficients))) < 3e-11


def test_anisotropic_global_hierarchy_with_fast_moments():
    rng = np.random.default_rng(35)
    sources = rng.uniform(-1, 1, (97, 3))
    data = rng.normal(size=97)+1j*rng.normal(size=97)
    queries = rng.uniform(-1, 1, (40, 6))
    fast = SourceButterfly(polynomial_phase, sources, output_dimension=6, depth=1,
                           order=(14, 10, 9), moment_backend="nufft")
    actual = fast.transform(data).evaluate(queries)
    expected = np.exp(1j*polynomial_phase(queries, sources)) @ data
    np.testing.assert_allclose(actual, expected, atol=3e-7)
    values = rng.normal(size=40)+1j*rng.normal(size=40)
    assert abs(np.vdot(actual, values)-np.vdot(data, fast.adjoint(queries, values))) < 1e-10


def test_native_global_kepler_evaluator_matches_same_dense_expansion():
    pytest.importorskip("numba")
    from research.kepler_transform.chart_experiments import setup
    from research.kepler_transform.global_kepler import ContinuousKeplerPhase
    from research.kepler_transform.factorization import ARCSEC
    rng = np.random.default_rng(36)
    chart, center, widths, _, _ = setup("near")
    phase = ContinuousKeplerPhase(chart, center, widths, time_half_span=4, frequency_scale=1/ARCSEC)
    sources = rng.uniform(-1, 1, (50, 3))
    plan = SourceButterfly(phase, sources, output_dimension=6, depth=1, order=(8, 7, 6))
    data = rng.normal(size=50)+1j*rng.normal(size=50)
    field = plan.transform(data)
    queries = np.concatenate((rng.uniform(-1, 1, (50, 6)), rng.uniform(.2, .8, (40, 6))))
    actual = field.evaluate(queries, accelerated=True)
    np.testing.assert_allclose(actual, field.evaluate(queries),
                               atol=2e-9, rtol=2e-11)
    values = rng.normal(size=len(queries))+1j*rng.normal(size=len(queries))
    assert abs(np.vdot(actual, values)-np.vdot(data, plan.adjoint(queries, values))) < 2e-8
    targets = rng.uniform(-1, 1, (2, 6))
    factors = phase.factorized_gate(targets, plan.nodes)
    phases = phase(targets, plan.grid)
    np.testing.assert_allclose(factors.dense(), np.exp(1j*(phases[0]-phases[1])), atol=2e-12)
