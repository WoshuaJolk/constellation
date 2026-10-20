import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("numba")
from flint import arb, ctx
from scipy.integrate import solve_ivp

from research.kepler_transform.chart_experiments import setup
from research.kepler_transform.factorization import ARCSEC
from research.kepler_transform.reception_flow import (
    CircularObserver, PolynomialObserver, analytic_bound, box_balls, enclose,
    float_box, reception_cover, rhs,
)
from research.kepler_transform.segment_field import SegmentTransform, TensorGrid
from research.kepler_transform.segment_field_certificate import certify
from research.kepler_transform.segment_experiments import positions_from_states
from research.kepler_transform.cascade_transfer import CascadeTransfer
from research.kepler_transform.segment_field_experiments import real_pixels


@pytest.fixture
def domain():
    old, center, widths, _, _ = setup("near")
    center, widths = reception_cover(old, center, widths)
    return CircularObserver(), center, widths


@pytest.mark.parametrize("case", ["near", "distant", "stress"])
def test_entire_old_domain_is_covered_and_rebased_enclosure_contains_witnesses(case):
    old, center, widths, geometry, _ = setup(case)
    observer = CircularObserver()
    c, w = reception_cover(old, center, widths)
    rng = np.random.default_rng(98)
    original = center+rng.uniform(-1, 1, (128, 6))*widths
    parameters = observer.chart(0.).parameters(old.states(original))
    assert np.all(abs(parameters-c) < w)
    for time in [-float(geometry.times[-1])/2, float(geometry.times[-1])/2]:
        endpoint, _, work = enclose(box_balls(c, w), 0., time, observer)
        converted = observer.chart(0.).rebase(parameters, time, observer.chart(time))
        ec, ew = float_box(endpoint)
        assert np.all(abs(converted-ec) < ew)
        assert work["steps"] > 0


def test_reception_ode_matches_independent_cartesian_propagation(domain):
    observer, c, w = domain
    q = c+.3*w

    def derivative(t, y):
        return [float(a.mid()) for a in rhs(arb(float(t)), list(map(arb, y)), observer)]
    solution = solve_ivp(derivative, [0., 2.], q, method="DOP853", rtol=2e-12, atol=1e-15)
    reference = observer.chart(0.).rebase(q[None], 2., observer.chart(2.))[0]
    np.testing.assert_allclose(solution.y[:, -1], reference, rtol=1e-10, atol=1e-13)
    # The observed angular rates are exactly the position derivatives.
    value = derivative(0., q)
    assert value[3] == q[5] and value[4] == q[2]


def test_interval_flow_refuses_nonphysical_domain(domain):
    observer, c, w = domain
    w[0] = c[0]*2
    with pytest.raises(ValueError, match="range"):
        enclose(box_balls(c, w), 0., 1., observer)


def test_polynomial_observer_derivatives_are_exact_for_model():
    observer = PolynomialObserver([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    p, v, a = observer.jet(arb(2))
    np.testing.assert_allclose(list(map(float, p)), [37., 44., 51.])
    np.testing.assert_allclose(list(map(float, v)), [32., 37., 42.])
    np.testing.assert_allclose(list(map(float, a)), [14., 16., 18.])


def test_anisotropic_piecewise_interpolation_preserves_polynomials_and_adjoint(domain):
    _, c, w = domain
    grid = TensorGrid(c, w, (1, 1, 1, 2, 2, 1), (2, 2, 2, 4, 3, 2))
    rng = np.random.default_rng(65)
    q = c+rng.uniform(-.99, .99, (100, 6))*w
    def polynomial(x):
        z = (x-c)/w
        return z[:, 3]**3*z[:, 4]**2+z[:, 0]*z[:, 5]
    matrix = grid.matrix(q)
    np.testing.assert_allclose(matrix @ polynomial(grid.points()), polynomial(q), atol=2e-14)
    a, b = rng.normal(size=grid.size), rng.normal(size=len(q))
    np.testing.assert_allclose(np.vdot(b, matrix @ a), np.vdot(matrix.T @ b, a), atol=3e-14)
    q[0, 0] = c[0]+2*w[0]
    with pytest.raises(ValueError, match="leaves"):
        grid.matrix(q)
    with pytest.raises(MemoryError):
        grid.matrix(q[1:], max_entries=1)


@pytest.mark.parametrize("exact_leaves", [False, True])
def test_actual_bottom_up_field_and_full_adjoint(domain, exact_leaves):
    observer, c, w = domain
    times = np.linspace(-.1, .1, 8)
    frequencies = np.tile([[.003/ARCSEC, .002/ARCSEC]], (len(times), 1, 1))
    operator = SegmentTransform(observer, c, w, times, frequencies, order=3,
                                exact_leaves=exact_leaves, root_epoch=0.)
    rng = np.random.default_rng(33)
    data = rng.normal(size=len(times))+1j*rng.normal(size=len(times))
    q = c+rng.uniform(-.9, .9, (24, 6))*w
    field = operator.transform(data)
    positions = positions_from_states(observer.chart(0.).states(q), times)
    reference = np.exp(1j*np.einsum("nti,ti->nt", positions, frequencies[:, 0])) @ data
    assert np.max(abs(field.evaluate(q)-reference))/sum(abs(data)) < 1e-5
    y = rng.normal(size=len(q))+1j*rng.normal(size=len(q))
    np.testing.assert_allclose(np.vdot(y, field.evaluate(q)),
                               np.vdot(field.evaluate_adjoint(q, y), data), rtol=1e-12, atol=1e-11)
    assert operator.stats["parent_values_use_only_child_fields"]
    assert not operator.stats["full_objective_complete"]


def test_regularized_inverse_agrees_with_dense_reference(domain):
    observer, c, w = domain
    times = [-.05, .05]
    frequencies = np.array([[[.01, .002], [-.01, -.002]],
                            [[.003, -.007], [-.003, .007]]])/ARCSEC
    operator = SegmentTransform(observer, c, w, times, frequencies, order=2,
                                exact_leaves=True)
    matrix = np.column_stack([operator.transform(x).values for x in np.eye(4)])
    rng = np.random.default_rng(44)
    values = rng.normal(size=operator.output_size)+1j*rng.normal(size=operator.output_size)
    damping = .3
    actual, report = operator.regularized_inverse(values, damping=damping, tolerance=1e-13)
    expected = np.linalg.solve(matrix.conj().T @ matrix+damping*damping*np.eye(4),
                               matrix.conj().T @ values)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    assert report["converged"] and not report["exact_unregularized_inverse"]
    assert report["normal_residual_norm"] < 1e-9


def test_tensor_error_certificate_includes_child_errors_and_is_not_accuracy_claim(domain):
    observer, c, w = domain
    operator = SegmentTransform(observer, c, w, [-.05, .05],
                                [[[10., 5.]], [[-4., 8.]]], order=2)
    result = certify(operator, ellipse=1.5)
    assert len(result["segments"]) == 3
    root = result["segments"][-1]
    assert root["subtree_error_per_source_l1"] > root["local_error_per_source_l1"]
    assert result["uniform_error_per_source_l1"] > 1e-8
    assert result["exact_arithmetic_only"]
    assert not result["floating_point_evaluation_certified"]


def test_complex_time_flow_bound_covers_all_real_times(domain):
    observer, c, w = domain
    proof = analytic_bound(c, w, 0., [-1., 1.], observer, [.01/ARCSEC]*2)
    assert all(np.isfinite(proof["kernel_modulus_bounds"]))
    assert all(m >= 1 for m in proof["kernel_modulus_bounds"])
    assert proof["whole_box_exact_arithmetic"]


def test_cascade_removes_tensor_stencil_and_retains_adjoint(domain):
    observer, c, w = domain
    target, _, _ = enclose(box_balls(c, w), 0., .1, observer)
    cc, cw = float_box(target)
    parent, child = TensorGrid(c, w, (1,)*6, 3), TensorGrid(cc, cw, (1,)*6, 3)
    mapping = lambda q: observer.chart(0.).rebase(q, .1, observer.chart(.1))
    cascade = CascadeTransfer(parent, child, mapping)
    assert cascade.entries == 6*parent.size*3
    assert cascade.entries < parent.size*child.size/20
    rng = np.random.default_rng(41)
    x, y = rng.normal(size=child.size), rng.normal(size=parent.size)
    np.testing.assert_allclose(np.vdot(y, cascade @ x), np.vdot(cascade.adjoint(y), x), atol=3e-12)
    assert not cascade.report["intermediate_domains_certified"]


def test_cascade_is_not_allowed_to_inherit_tensor_certificate(domain):
    observer, c, w = domain
    op = SegmentTransform(observer, c, w, np.linspace(-.05, .05, 4),
                          [[[2., 3.]]]*4, order=2, transfer_backend="cascade")
    with pytest.raises(ValueError, match="separate certificate"):
        certify(op)


def test_budget_failure_never_falls_back_to_direct_orbits(domain):
    observer, c, w = domain
    with pytest.raises(MemoryError, match="budget"):
        SegmentTransform(observer, c, w, [-1., 1.], [[[1., 1.]]]*2,
                         order=10, max_nodes=1000)
    with pytest.raises(ValueError, match="increasing"):
        SegmentTransform(observer, c, w, [0., 0.], [[[1., 1.]]]*2)


def test_segment_fields_on_actual_ztf_pixels_full_wcs_and_profiled_background():
    result = real_pixels(14)
    assert result["uses_full_recorded_TPV_WCS"] and result["real_survey_data"]
    assert result["numerator_relative_sup_error"] < 3e-7
    assert result["information_relative_sup_error"] < 2e-7
    assert result["minimum_interpolated_information"] > 0
    assert result["observer_at_exposure_max_absolute_error_au"] < 3e-15
    assert not result["uniform_detector_interpolation_error_certified"]
    assert not result["full_objective_complete"]


def test_single_exposure_does_not_waive_physical_or_full_domain_validation(domain):
    observer, c, w = domain
    op = SegmentTransform(observer, c, w, [0.], [[[1., 2.]]], order=2)
    field = op.transform([1.])
    bad = c.copy()
    bad[0] += 2*w[0]
    with pytest.raises(ValueError, match="six-coordinate"):
        field.evaluate(bad[None])
    with pytest.raises(ValueError, match="six-coordinate"):
        field.evaluate_adjoint(bad[None], [1.])
    c[0] = -1.
    with pytest.raises(ValueError, match="range"):
        SegmentTransform(observer, c, w, [0.], [[[1., 2.]]], order=2)
