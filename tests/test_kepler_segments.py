import math

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("numba")

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from magictools.orbits import C_AU_DAY, GM
from research.kepler_transform.chart_experiments import setup
from research.kepler_transform.factorization import ARCSEC, observer_path
from research.kepler_transform.segment_composition import (
    ReceptionChart, conditional_tree_counts, finite_difference_bound, finite_difference_vector,
)
from research.kepler_transform.segment_experiments import (
    composition_experiment, endpoint_experiment, endpoint_rank_certificate,
    inverse_certificate, real_pixel_composition, reception_chart,
)


def test_reception_coordinates_against_independent_ode_and_light_time():
    chart = reception_chart(0.)
    parameters = np.array([[.5, .001, 3*ARCSEC, 8*ARCSEC, -7*ARCSEC, -2*ARCSEC]])
    emitted, delay = chart.emitted(parameters)
    initial = emitted[0]

    def rhs(t, state):
        r = state[:3]
        return np.r_[state[3:], -GM["sun"]*r/np.linalg.norm(r)**3]

    # Integrate from the physical emission epoch; no universal-variable solver.
    solution = solve_ivp(rhs, [-delay[0], .02], initial, method="DOP853",
                         rtol=2e-13, atol=2e-15, dense_output=True)
    assert solution.success
    np.testing.assert_allclose(chart.states(parameters)[0], solution.sol(0.), atol=3e-13, rtol=0)
    positions = []
    for reception in [0., 1e-3, 2e-3]:
        observer = observer_path(np.array([reception]))[0]
        function = lambda t: t+np.linalg.norm(solution.sol(t)[:3]-observer)/C_AU_DAY-reception
        emission = brentq(function, -delay[0]-1e-10, reception, xtol=1e-15)
        delta = solution.sol(emission)[:3]-observer
        positions.append(delta[1:]/delta[0])
    positions = np.asarray(positions)
    np.testing.assert_allclose(positions[0], parameters[0, [3, 4]], atol=2e-15)
    # Second-order one-sided derivative; exact rates are defined at reception 0.
    derivative = (-3*positions[0]+4*positions[1]-positions[2])/.002
    np.testing.assert_allclose(derivative, parameters[0, [5, 2]], atol=2e-12, rtol=0)
    generic = np.array(chart.scaled_emitted(parameters[0]))
    np.testing.assert_allclose(generic[:3]/parameters[0, 0], emitted[0, :3])
    np.testing.assert_allclose(generic[3:]/parameters[0, 0], emitted[0, 3:])
    assert chart.digest != setup("near")[0].digest


@pytest.mark.parametrize("case", ["near", "distant", "stress"])
def test_reception_rebasing_preserves_orbits_and_segment_likelihood(case):
    result = composition_experiment(case)
    assert result["state_roundtrip_max_absolute"] < 3e-13
    assert result["rebase_composition_scaled_error"] < 1e-10
    assert result["observed_rate_finite_difference_error_radians_day"] < 2e-12
    assert result["segment_score_identity_max_error"] < 2e-9
    assert result["bound_states"] > 0 and result["unbound_states"] > 0


def test_parabolic_state_is_not_excluded_by_reception_chart():
    state = np.array([[3., 0., 0., 0., np.sqrt(2*GM["sun"]/3), 0.]])
    chart = reception_chart(0.)
    np.testing.assert_allclose(chart.states(chart.parameters(state)), state, atol=2e-14, rtol=0)


def test_two_identical_endpoints_do_not_define_the_kepler_track():
    witness = endpoint_experiment("stress")
    assert witness["independently_checked_endpoint_error_arcsec"] < 1e-7
    assert witness["independently_checked_midpoint_separation_arcsec"] > 1.
    for name in ["reference_normalized", "alternative_normalized"]:
        assert np.all(np.abs(witness[name]) < 1)
    proof = endpoint_rank_certificate("stress")
    for name in ["endpoint_minor_determinant_interval", "three_epoch_determinant_interval"]:
        lo, hi = proof[name]
        assert lo <= hi and (hi < 0 or lo > 0)
    assert proof["local_endpoint_fiber_dimension"] == 2


def test_inverse_obstruction_uses_distinct_times_nonzero_frequency_and_whole_box():
    proof = inverse_certificate(order=32, total_span=1.)
    assert proof["uniform_l2_input_to_sup_response_upper"] < 1e-17
    assert proof["real_input_inverse_lipschitz_lower_bound"] > 1e17
    assert proof["distinct_exposures"] == 33
    assert not proof["zero_spatial_frequencies_used"]
    assert proof["whole_original_domain"]
    np.testing.assert_array_equal(proof["root_widths"], setup("near")[2])
    # No false assertion that float64 cancellation verifies the tiny exact bound.
    assert not proof["ordinary_float_input_vector_certified"]


def test_segment_identity_on_real_ztf_pixels_and_full_wcs():
    result = real_pixel_composition()
    assert result["real_survey_data"] and result["exposures"] == 3
    assert result["uses_full_recorded_TPV_WCS"]
    assert result["position_identity_max_pixels"] < 1e-7
    assert result["numerator_identity_max_absolute"] < 1e-8
    assert result["information_identity_max_absolute"] < 1e-8
    assert result["profiled_background_shift_numerator_max_absolute"] < 1e-10
    assert not result["fast_transform_or_discovery_claimed"]


@pytest.mark.parametrize("order", [1, 2, 8, 16])
def test_finite_difference_null_vector_and_bound_on_known_analytic_function(order):
    a = finite_difference_vector(order)
    np.testing.assert_allclose(np.linalg.norm(a), 1., atol=2e-16)
    step, frequency = .07, 2.3
    # Closed-form difference of exp(i*w*t), independent of the summing code.
    exact_magnitude = abs(2*np.sin(frequency*step/2))**order/math.sqrt(math.comb(2*order, order))
    bound = finite_difference_bound(order, step, frequency**order)
    assert exact_magnitude <= bound
    actual = abs(a @ np.exp(1j*frequency*step*np.arange(order+1)))
    assert abs(actual-exact_magnitude) < 3e-15


def test_conditional_tree_count_is_not_presented_as_a_kepler_certificate():
    shallow = conditional_tree_counts(16, 8)
    deep = conditional_tree_counts(16, 16)
    assert shallow["total_cells"] < 2*(shallow["root_cells"]+shallow["leaf_cells"])
    assert deep["total_cells"] < 2*(deep["root_cells"]+deep["leaf_cells"])
    assert not deep["chart_cover_certified"]
    assert not deep["interpolation_order_and_constants_included"]
    assert not deep["full_objective_complete"]


@pytest.mark.parametrize("kwargs", [{"depth": -1}, {"depth": True}, {"bandwidth": 0},
                                   {"half_span": float("nan")}])
def test_invalid_count_parameters_are_rejected(kwargs):
    options = dict(depth=4, bandwidth=16)
    options.update(kwargs)
    with pytest.raises(ValueError):
        conditional_tree_counts(**options)


def test_nonphysical_reception_chart_is_rejected():
    chart = reception_chart(0.)
    with pytest.raises(ValueError):
        chart.states(np.array([[.5, C_AU_DAY, 0., 0., 0., 0.]]))
    with pytest.raises(ValueError):
        ReceptionChart([1., 0., 0.], [C_AU_DAY, 0., 0.], *np.eye(3))
