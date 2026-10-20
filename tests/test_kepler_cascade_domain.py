import itertools

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("numba")
from flint import acb, arb, arb_mat, ctx

from research.kepler_transform.chart_experiments import setup
from research.kepler_transform.cascade_domain import certify_atlas, certify_domain, solve_hybrid
from research.kepler_transform.certified_cascade import (
    CertifiedCascadeTransfer, cadence_count, fourier_modulus_bound,
    interpolation_error, sufficient_uniform_order,
)
from research.kepler_transform.factorization import ARCSEC
from research.kepler_transform.reception_flow import CircularObserver, box_balls, reception_cover, rhs
from research.kepler_transform.segment_field import TensorGrid
from research.kepler_transform.certified_segment_field import CertifiedAtlasTransform
from research.kepler_transform.segment_experiments import positions_from_states
from research.kepler_transform.tensor_prolongation import TensorProlongation
from research.kepler_transform.variational_flow import flow_jacobian, peano_increment, rhs_jacobian
from research.kepler_transform.verify_constructive_campaign import verify_rectangular_cover


@pytest.fixture(autouse=True)
def precision():
    previous = ctx.prec
    ctx.prec = 120
    yield
    ctx.prec = previous


def domain(name):
    old, c, w, _, _ = setup(name)
    c, w = reception_cover(old, c, w)
    return CircularObserver(), c, w


@pytest.mark.parametrize("case", ["near", "distant", "stress"])
def test_automatic_derivative_against_high_precision_differences(case):
    observer, c, w = domain(case)
    q = list(map(arb, c))
    actual = rhs_jacobian(arb(".25"), q, observer, w)
    step = arb("1e-12")
    for axis in range(6):
        plus, minus = q.copy(), q.copy()
        plus[axis] += step*float(w[axis])
        minus[axis] -= step*float(w[axis])
        fp, fm = rhs(arb(".25"), plus, observer), rhs(arb(".25"), minus, observer)
        expected = [(a-b)/(2*step*float(width)) for a, b, width in zip(fp, fm, w)]
        np.testing.assert_allclose(
            [float(actual[i, axis].mid()) for i in range(6)],
            [float(a.mid()) for a in expected], rtol=1e-10, atol=1e-12)


def test_peano_encloses_noncommuting_time_ordered_product_in_both_directions():
    first = arb_mat([[0, 2], [0, 0]])
    second = arb_mat([[0, 0], [-3, 0]])
    enclosure = arb_mat([[0, arb(0).union(2)], [arb(-3).union(0), 0]])
    identity = arb_mat([[1, 0], [0, 1]])
    for h in [arb(".125"), arb("-.125")]:
        delta, _ = peano_increment(enclosure, h)
        # Both generators are nilpotent; these exact products differ by order.
        exact = (identity+second*h/2)*(identity+first*h/2)
        reversed_product = (identity+first*h/2)*(identity+second*h/2)
        assert not exact[0, 0].overlaps(reversed_product[0, 0])
        for i, j in itertools.product(range(2), repeat=2):
            assert (identity+delta)[i, j].contains(exact[i, j])


@pytest.mark.parametrize("case", ["near", "distant", "stress"])
def test_flow_derivative_contains_independent_cartesian_witnesses(case):
    observer, c, w = domain(case)
    _, delta, _ = flow_jacobian(box_balls(c, w), 0., .125, observer, w)
    rng = np.random.default_rng(29)
    for q in c+rng.uniform(-.9, .9, (4, 6))*w:
        h = 1e-3
        points = q+np.r_[np.eye(6), -np.eye(6)]*w*h
        mapped = observer.chart(0.).rebase(points, .125, observer.chart(.125))
        numerical = ((mapped[:6]-mapped[6:])/(2*h)).T/w[:, None]-np.eye(6)
        for i, j in itertools.product(range(6), repeat=2):
            # Float finite differences are diagnostics, not the enclosure proof.
            assert float(delta[i, j].lower())-2e-7 <= numerical[i, j]
            assert numerical[i, j] <= float(delta[i, j].upper())+2e-7


@pytest.mark.parametrize("case", ["near", "distant", "stress"])
@pytest.mark.parametrize("offset", [-.125, .125])
def test_all_hybrids_have_whole_box_and_complex_certificates(case, offset):
    observer, c, w = domain(case)
    proof = certify_domain(observer, c, w, 0., offset, complex_ellipse=1.05)
    assert proof["all_coordinate_hybrids_certified"]
    assert proof["complex_inverse_certified"]
    assert proof["weighted_contraction_bound"] < 1
    assert proof["complex_weighted_contraction_bound"] < 1
    assert not proof["floating_point_evaluation_certified"]
    assert not proof["full_objective_complete"]
    if case == "stress":
        assert proof["largest_range_partition"] > 1
    rebase = lambda q: observer.chart(0.).rebase(q, offset, observer.chart(offset))
    rng = np.random.default_rng(192)
    cc, cw = np.array(proof["child_center"]), np.array(proof["child_widths"])
    for mask in range(64):
        fixed = [a for a in range(6) if mask & (1 << a)]
        points = cc+rng.uniform(-1., 1., (3, 6))*cw
        points[:, fixed] = c[fixed]+rng.uniform(-1., 1., (3, len(fixed)))*w[fixed]
        q, numerical = solve_hybrid(proof, points, fixed, rebase, tolerance=2e-9)
        assert numerical["scaled_residual"] <= 2e-9
        assert np.all(abs((q-c)/w) <= proof["normalized_enclosing_radii"])
        result = rebase(q)
        assert np.all(abs(result-cc) <= cw*(1+1e-8))


def test_certificate_drives_executable_transfer_and_adjoint():
    observer, c, w = domain("near")
    rng = np.random.default_rng(8)
    errors = []
    for order in [3, 5]:
        parent = TensorGrid(c, w, (1,)*6, order)
        transfer = CertifiedCascadeTransfer(
            observer, parent, 0., .125, certificate_options={"complex_ellipse": 1.2})
        k = np.array([.006, -.004])/ARCSEC
        source = np.exp(1j*transfer.child.points()[:, 3:5] @ k)
        actual = transfer @ source
        q = observer.chart(0.).rebase(parent.points(), .125, observer.chart(.125))
        errors.append(np.max(abs(actual-np.exp(1j*q[:, 3:5] @ k))))
        y = rng.normal(size=parent.size)+1j*rng.normal(size=parent.size)
        np.testing.assert_allclose(np.vdot(y, actual),
                                   np.vdot(transfer.adjoint(y), source), atol=1e-11)
        assert transfer.report["intermediate_domains_certified"]
        assert not transfer.report["whole_box_error_certified"]
    assert errors[1] < errors[0]/100


def test_separable_refinement_preserves_discontinuous_piecewise_polynomials():
    _, c, w = domain("near")
    source = TensorGrid(c, w, (2, 1, 1, 1, 1, 1), 2)
    target = TensorGrid(c, w, (4, 2, 1, 1, 1, 1), 3)
    transfer = TensorProlongation(source, target)
    rng = np.random.default_rng(55)
    # Independent source cells intentionally have different one-sided traces.
    values = rng.normal(size=source.size)+1j*rng.normal(size=source.size)
    q = c+rng.uniform(-.99, .99, (40, 6))*w
    np.testing.assert_allclose(target.matrix(q) @ (transfer @ values),
                               source.matrix(q) @ values, rtol=1e-12, atol=1e-12)
    y = rng.normal(size=target.size)+1j*rng.normal(size=target.size)
    np.testing.assert_allclose(np.vdot(y, transfer @ values),
                               np.vdot(transfer.adjoint(y), values), atol=2e-11)
    assert transfer.entries < target.size
    with pytest.raises(MemoryError):
        TensorProlongation(source, target, max_work_nodes=10)
    with pytest.raises(ValueError, match="nested"):
        TensorProlongation(source, TensorGrid(c, w, (3, 1, 1, 1, 1, 1), 3))


def test_refine_before_mixing_repairs_coarse_coordinate_error():
    observer, c, w = domain("near")
    parent = TensorGrid(c, w, (1,)*6, 5)
    errors = []
    for refine in [False, True]:
        transfer = CertifiedCascadeTransfer(observer, parent, 0., .5,
                                             child_order=2, refine=refine)
        def polynomial(q):
            z = (q-transfer.child.center)/transfer.child.widths
            return z[:, 3]+.3*z[:, 4]+.2*z[:, 0]*z[:, 1]+.1*z[:, 2]*z[:, 5]
        actual = transfer @ polynomial(transfer.child.points())
        mapped = observer.chart(0.).rebase(parent.points(), .5, observer.chart(.5))
        errors.append(np.max(abs(actual-polynomial(mapped))))
    assert errors[0] > 1e-4
    assert errors[1] < 1e-8
    assert errors[1] < errors[0]*1e-5


def test_complex_certificate_yields_executable_error_order_selection():
    observer, c, w = domain("stress")
    proof = certify_domain(observer, c, w, 0., .125, complex_ellipse=1.2)
    modulus = fourier_modulus_bound(proof, observer, [1/ARCSEC, 1/ARCSEC])
    selected = sufficient_uniform_order(
        proof["complex_ellipse"], modulus["log_modulus_per_source_l1"], 1e-8)
    assert selected["log_uniform_error_per_source_l1"] < np.log(1e-8)
    assert not selected["grid_constructed"]
    assert selected["cascade_weights_on_equal_uniform_grids"] == 6*selected["order"]**7
    lower = interpolation_error((selected["order"]-1,)*6, proof["complex_ellipse"],
                                 modulus["log_modulus_per_source_l1"])
    assert lower["log_uniform_error_per_source_l1"] > np.log(1e-8)


def test_irregular_cadence_count_and_duplicate_times():
    rng = np.random.default_rng(12)
    for times in [np.zeros(100), np.arange(100)/99,
                  np.sort(np.r_[rng.uniform(0, 1e-7, 900), rng.uniform(.9, 1., 100)])]:
        result = cadence_count(times, 100)
        assert result["modeled_total"] <= result["modeled_upper_bound"]*(1+1e-14)
        assert result["resolution_law_is_assumed"]
    assert cadence_count([0.], 3.)["modeled_total"] == 9.


def test_wide_domain_atlas_retains_original_box():
    observer, c, w = domain("stress")
    proof = certify_atlas(observer, c, w, 0., 2., complex_ellipse=1.05, max_iterations=8)
    assert proof["leaf_count"] > 1
    assert proof["whole_original_box_covered_by_outward_bisections"]
    assert proof["complex_inverse_certified_on_each_patch"]
    rng = np.random.default_rng(62)
    q = c+rng.uniform(-.99999, .99999, (512, 6))*w
    covered = np.zeros(len(q), bool)
    for leaf in proof["leaves"]:
        covered |= np.all(abs(q-leaf["parent_center"]) <= leaf["parent_widths"], axis=1)
    assert np.all(covered)
    with pytest.raises(ValueError, match="no partial cover"):
        certify_atlas(observer, c, w, 0., 2., max_leaves=1,
                      complex_ellipse=1.05, max_iterations=8)


def test_full_certified_hierarchy_query_and_adjoint():
    observer, c, w = domain("near")
    times = np.linspace(-.125, .125, 8)
    frequencies = np.tile([[[.003/ARCSEC, .002/ARCSEC]]], (8, 1, 1))
    transform = CertifiedAtlasTransform(
        observer, c, w, times, frequencies, order=3, exact_leaves=True,
        certificate_options={"complex_ellipse": 1.05})
    rng = np.random.default_rng(202)
    data = rng.normal(size=8)+1j*rng.normal(size=8)
    q = c+rng.uniform(-.95, .95, (24, 6))*w
    field = transform.transform(data)
    positions = positions_from_states(observer.chart(0.).states(q), times)
    expected = np.exp(1j*np.einsum("nti,ti->nt", positions, frequencies[:, 0])) @ data
    assert np.max(abs(field.evaluate(q)-expected))/sum(abs(data)) < 1e-5
    y = rng.normal(size=len(q))+1j*rng.normal(size=len(q))
    np.testing.assert_allclose(np.vdot(y, field.evaluate(q)),
                               np.vdot(field.evaluate_adjoint(q, y), data), atol=2e-11)
    for graph in transform.stats["graphs"]:
        assert graph["cascade_intermediate_domains_certified"]
        assert graph["complex_cascade_inverses_certified"]
        assert not graph["interpolation_accuracy_certified"]


def test_refuse_invalid_domain_and_unproven_error_input():
    observer, c, w = domain("near")
    proof = certify_domain(observer, c, w, 0., .125)
    with pytest.raises(ValueError, match="complex inverse"):
        fourier_modulus_bound(proof, observer, [1., 1.])
    with pytest.raises(ValueError):
        certify_domain(observer, c, -w, 0., .1)
    with pytest.raises(ValueError):
        rhs_jacobian(arb(0), list(map(arb, c)), observer, [0.]*6)
    with pytest.raises(ValueError):
        sufficient_uniform_order(1.2, 10., max_order=2)


def test_exact_atlas_verifier_detects_a_hole_without_sampling():
    center, widths = [0.]*6, [1.]*6
    patches = [{"center": [-.5]+[0.]*5, "widths": [.5]+[1.]*5},
               {"center": [.5]+[0.]*5, "widths": [.5]+[1.]*5}]
    verify_rectangular_cover(center, widths, patches)
    with pytest.raises(ValueError, match="hole"):
        verify_rectangular_cover(center, widths, patches[:1])
