"""Independent physical checks, domain accounting, and adversarial budgets."""

from itertools import product

import numpy as np
import pytest

pytest.importorskip("flint")
pytest.importorskip("finufft")

from magictools.orbits import GM
from research.kepler_transform.chart_atlas import ChartAtlas, scalar_position, third_coefficients
from research.kepler_transform.chart_cover import ChartCover, audit_scores, enclosing_cell, physical_kernel, polynomial_positions
from research.kepler_transform.continuous import verify_partition
from research.kepler_transform.factorization import ARCSEC, observer_path
from research.kepler_transform.kernel import KeplerKernel
from research.kepler_transform.observer_chart import ObserverChart


def fixture(retarded=True, frames=1, span=.2):
    chart = ObserverChart([1, 0, 0], [0, np.sqrt(GM["sun"]), 0],
                          [1, 0, 0], [0, 1, 0], [0, 0, 1], retarded_epoch=retarded)
    center = np.array([.5, 0., 1e-5, 0., 0., -.001])
    widths = np.array([.01, .0001, 2*ARCSEC, 10*ARCSEC, 10*ARCSEC, 2*ARCSEC])
    times = np.linspace(-span/2, span/2, frames)
    state = chart.states(center[None])[0]
    rng = np.random.default_rng(41)
    geometry = KeplerKernel(state[None], times, observer_path(times), np.arange(16) % frames,
                            rng.uniform(-1, 1, (16, 2))/ARCSEC, state)
    data = rng.normal(size=16)+1j*rng.normal(size=16)
    return chart, center, widths, geometry, data


@pytest.mark.parametrize("retarded", [False, True])
def test_chart_round_trip_and_reference_direction(retarded):
    chart, center, widths, geometry, _ = fixture(retarded)
    q = center+np.random.default_rng(42).uniform(-1, 1, (30, 6))*widths
    states = chart.states(q)
    np.testing.assert_allclose(chart.parameters(states), q, atol=3e-14, rtol=1e-11)
    if retarded:
        from research.kepler_transform.universal import apparent
        emitted = apparent(states, np.array([0.]), chart.observer[None])[0][:, 0, :3]
        relative = (emitted-chart.observer) @ chart.axes.T
        np.testing.assert_allclose(relative[:, 1:]/relative[:, :1], q[:, 3:5], atol=3e-14)


def mp_coordinate(chart, parameters, time, observer, normal, axis):
    """Independent 60-digit unscaled universal anomaly/light-time equation."""
    import mpmath as mp
    from research.detection_limit.interval_orbits import MU, C_AU_DAY
    mu, speed = mp.mpf(MU), mp.mpf(C_AU_DAY)
    rho, eta, wy, x, y, wx = parameters
    dot = lambda a, b: sum(i*j for i, j in zip(a, b))
    normal0, east0, north0 = [[mp.mpf(float(v)) for v in row] for row in chart.axes]
    direction = [n+x*e+y*v for n, e, v in zip(normal0, east0, north0)]
    r = [mp.mpf(float(o))+d/rho for o, d in zip(chart.observer, direction)]
    v = [mp.mpf(float(o))+(eta*d+wx*e+wy*n)/rho
         for o, d, e, n in zip(chart.velocity, direction, east0, north0)]
    radius = mp.sqrt(dot(r, r))
    alpha = 2/radius-dot(v, v)/mu
    initial_delay = mp.sqrt(dot(direction, direction))/rho/speed if chart.retarded_epoch else 0

    def position(chi):
        argument = alpha*chi**2
        c = sum((-argument)**k/mp.factorial(2*k+2) for k in range(30))
        s = sum((-argument)**k/mp.factorial(2*k+3) for k in range(30))
        elapsed = (dot(r, v)/mp.sqrt(mu)*chi**2*c+(1-alpha*radius)*chi**3*s+radius*chi)/mp.sqrt(mu)
        f, g = 1-chi**2*c/radius, elapsed-chi**3*s/mp.sqrt(mu)
        return elapsed, [f*a+g*b-mp.mpf(float(o)) for a, b, o in zip(r, v, observer)]

    def equation(chi):
        elapsed, delta = position(chi)
        return elapsed+mp.sqrt(dot(delta, delta))/speed-mp.mpf(time)-initial_delay

    chi = mp.findroot(equation, mp.sqrt(mu)*time/radius)
    _, delta = position(chi)
    return dot(delta, [mp.mpf(float(x)) for x in axis])/dot(delta, [mp.mpf(float(x)) for x in normal])


@pytest.mark.parametrize("retarded,axis", [(False, 0), (True, 0), (True, 1), (True, 2),
                                          (True, 3), (True, 4), (True, 5)])
def test_scaled_parameter_dependent_dynamics_and_light_time_third_derivatives(retarded, axis):
    mp = pytest.importorskip("mpmath")
    from flint import arb, ctx
    chart, center, widths, geometry, _ = fixture(retarded, span=2.)
    with ctx.workprec(128), mp.workdps(60):
        observer, normal, east, north = [getattr(geometry, a)[0] for a in ("observers", "normal", "east", "north")]
        time = float(geometry.times[0])
        enclosed = third_coefficients(chart, center, widths, axis, time, observer, normal, east, north)
        for point in [-.7, .6]:
            def coordinate(z):
                q = [mp.mpf(float(c))+(z if j == axis else mp.mpf(".31"))*mp.mpf(float(w))
                     for j, (c, w) in enumerate(zip(center, widths))]
                return mp_coordinate(chart, q, time, observer, normal, east)
            expected = mp.diff(coordinate, mp.mpf(point), 3)/6
            assert enclosed[0].contains(arb(mp.nstr(expected, 55)))
        q = [arb(float(c)) for c in center]
        position = scalar_position(chart, q, time, observer, normal, east, north)
        expected = mp_coordinate(chart, [mp.mpf(float(c)) for c in center], time, observer, normal, east)
        assert position[0].contains(arb(mp.nstr(expected, 55)))


def test_chart_atlas_certificate_and_arbitrary_queries():
    chart, center, widths, geometry, data = fixture()
    atlas = ChartAtlas(chart, center, widths, geometry)
    certificate = atlas.certify()
    points = np.random.default_rng(43).uniform(-1, 1, (48, 6))
    exact = physical_kernel(geometry, atlas.states(points))
    truth = exact.projected(np.arange(len(points)), np.arange(len(geometry.times)))
    np.testing.assert_array_less(abs(polynomial_positions(atlas, points)-truth),
                                 np.broadcast_to(certificate["position_error_radians"], truth.shape)+2e-14)
    outer, inner = points[0, :3], points[:, 3:]
    plan = atlas.slice(outer)
    same_outer = np.column_stack((np.tile(outer, (len(points), 1)), inner))
    direct = physical_kernel(geometry, atlas.states(same_outer)).direct(data)
    np.testing.assert_allclose(plan.apply(data, inner), direct, atol=1e-7)


def test_outward_boxes_cover_exact_root_even_at_rounding_boundaries():
    from flint import arb, ctx
    center = np.array([.51, .001, 1e-5, .003, .002, -.001])
    widths = np.array([.1, .001, 1e-6, .0001, .0001, .00001])
    bounds = np.array([[-1., 0.], [0., .5], [-.25, .125], [-1., 1.], [-1., 1.], [-1., 1.]])
    c, w = enclosing_cell(center, widths, bounds)
    with ctx.workprec(128):
        for original, radius, limits, middle, half in zip(center, widths, bounds, c, w):
            covered = arb(float(middle), float(half))
            for limit in limits:
                assert covered.contains(arb(float(original))+arb(float(radius))*arb(float(limit)))


def test_cover_refusal_fallback_and_exact_partition():
    chart, center, widths, geometry, data = fixture()
    cover = ChartCover(chart, center, widths, geometry, position_tolerance=1e-25, max_leaves=2)
    assert verify_partition(cover.report())
    assert not cover.report()["all_cells_resolved"]
    q = center+np.array(list(product([-.9, .9], repeat=6)))*widths
    with pytest.raises(ValueError, match="unresolved"):
        cover.apply(data, q)
    actual, report = cover.apply(data, q, allow_direct_fallback=True)
    np.testing.assert_allclose(actual, physical_kernel(geometry, chart.states(q)).direct(data), atol=1e-9)
    assert report["direct_queries"] == len(q)
    assert report["all_queries_evaluated"]
    with pytest.raises(ValueError, match="outside"):
        cover.apply(data, (center+widths*2)[None], allow_direct_fallback=True)
    broken = dict(cover.report(), leaves=cover.report()["leaves"][:-1])
    with pytest.raises(ValueError, match="cover"):
        verify_partition(broken)


def test_resolved_cover_routes_and_complete_arithmetic_audits():
    chart, center, widths, geometry, data = fixture()
    cover = ChartCover(chart, center, widths, geometry, position_tolerance=1e-8, max_leaves=2)
    assert cover.report()["all_cells_resolved"]
    points = np.random.default_rng(44).uniform(-.99, .99, (32, 6))
    points[:, :3] = .21
    q = center+points*widths
    sliced, slices_report = cover.apply(data, q, route="slices")
    polynomial, poly_report = cover.apply(data, q, route="polynomial")
    np.testing.assert_allclose(sliced, polynomial, atol=1e-8)
    assert slices_report["slice_queries"] == len(q)
    assert poly_report["polynomial_queries"] == len(q)
    exact = physical_kernel(geometry, chart.states(q)).direct(data)
    audit = audit_scores(chart, geometry, q[:2], data, sliced[:2])
    assert max(row["absolute_error_upper"] for row in audit) < 1e-6
    for row, truth in zip(audit, exact):
        assert row["real_interval"][0]-1e-10 < truth.real < row["real_interval"][1]+1e-10
    wrong = audit_scores(chart, geometry, q[:1], data, sliced[:1]+1)
    assert wrong[0]["absolute_error_upper"] > .99


def test_invalid_chart_and_cover_inputs():
    with pytest.raises(ValueError):
        ObserverChart([1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1])
    chart, center, widths, geometry, _ = fixture()
    with pytest.raises(ValueError):
        chart.states(np.zeros((1, 6)))
    with pytest.raises(ValueError):
        ChartCover(chart, center, widths, geometry, position_tolerance=0)
    with pytest.raises(ValueError):
        ChartAtlas(chart, center, widths*-1, geometry)


def test_validated_universal_stumpff_tails_and_refusals():
    mp = pytest.importorskip("mpmath")
    from flint import arb, ctx
    from research.kepler_transform.validated_universal import stumpff, retarded_position
    with ctx.workprec(128), mp.workdps(60):
        for value in ["-.49", "0", ".49"]:
            c, s = stumpff(arb(value))
            z = mp.mpf(value)
            expected_c = sum((-z)**k/mp.factorial(2*k+2) for k in range(60))
            expected_s = sum((-z)**k/mp.factorial(2*k+3) for k in range(60))
            assert c.contains(arb(mp.nstr(expected_c, 55)))
            assert s.contains(arb(mp.nstr(expected_s, 55)))
        with pytest.raises(ValueError, match="range"):
            stumpff(arb(1))
        with pytest.raises(ValueError, match="angular momentum"):
            retarded_position([arb(x) for x in [1, 0, 0, 1, 0, 0]], 0,
                              [arb(0)]*3, arb(".001"), arb(173))


@pytest.mark.parametrize("tangent_speed", [.009, .014045, .02])
def test_validated_universal_nodes_across_conic_energies(tangent_speed):
    from flint import arb, ctx
    from research.detection_limit import interval_orbits as flow
    from research.kepler_transform.chart_atlas import ode_scalar_position
    from research.kepler_transform.validated_universal import retarded_position
    chart, center, _, geometry, _ = fixture()
    center[5] = (tangent_speed-chart.velocity[1])*center[0]
    with ctx.workprec(128):
        q = [arb(float(x)) for x in center]
        rho = q[0]
        for time in [-4., 0., 4.]:
            observer = observer_path(np.array([time]))[0]
            direct = retarded_position(chart.scaled_initial(q), arb(time),
                [rho*arb(float(o)) for o in observer], arb(flow.MU)*rho**3,
                arb(flow.C_AU_DAY)*rho, chart.path_offset(q))
            dot = lambda axis: sum((r*arb(float(a)) for r, a in zip(direct, axis)), arb(0))
            expected = ode_scalar_position(chart, q, time, observer,
                                            geometry.normal[0], geometry.east[0], geometry.north[0])
            assert (dot(geometry.east[0])/dot(geometry.normal[0])).overlaps(expected[0])


def test_cover_operator_adjoint_and_regularized_inverse_with_nullspace():
    from research.kepler_transform.chart_operator import ChartOperator
    chart, center, widths, geometry, data = fixture()
    # Exact duplicate columns create a known nullspace.
    geometry.frequencies[-1] = geometry.frequencies[0]
    cover = ChartCover(chart, center, widths, geometry, position_tolerance=1e-8, max_leaves=1)
    rng = np.random.default_rng(45)
    parameters = center+rng.uniform(-.9, .9, (64, 6))*widths
    operator = ChartOperator(cover, parameters)
    matrix = np.exp(1j*np.einsum("nmi,mi->nm", operator.positions[:, geometry.visits], geometry.frequencies))
    values = rng.normal(size=64)+1j*rng.normal(size=64)
    np.testing.assert_allclose(operator.apply(data), matrix @ data, atol=1e-9)
    np.testing.assert_allclose(np.vdot(operator.apply(data), values),
                               np.vdot(data, operator.adjoint(values)), atol=1e-8)
    null = np.zeros(16, complex)
    null[0], null[-1] = 1, -1
    np.testing.assert_allclose(operator.apply(null), 0., atol=1e-11)
    recovered, report = operator.inverse(values, damping=.3, tolerance=1e-11)
    expected = np.linalg.lstsq(np.vstack((matrix, .3*np.eye(16))), np.r_[values, np.zeros(16)], rcond=None)[0]
    np.testing.assert_allclose(recovered, expected, atol=2e-8)
    assert not report["unique_image_reconstruction_claimed"]
    _, limited = operator.inverse(values, max_iterations=1)
    assert limited["iteration_limit_reached"]


def test_finite_detector_matches_nonperiodic_pixels_including_edges_and_masks():
    from research.kepler_transform.detector_fields import finite_pixel_fields, direct_pixel_statistics
    from research.kepler_transform.chart_cover import positions_nufft
    rng = np.random.default_rng(46)
    images = rng.normal(size=(2, 7, 7))
    variances = rng.uniform(.5, 2., images.shape)
    masks = rng.random(images.shape) < .15
    sigma = np.array([[.9, 1.3], [1.2, .85]])
    visits, frequencies, numerator, information, report = finite_pixel_fields(
        images, variances, sigma, ARCSEC, masks=masks, source_half_width_pixels=6)
    chart, center, _, _, _ = fixture(frames=2)
    state = chart.states(center[None])[0]
    geometry = KeplerKernel(state[None], np.array([0., .1]), observer_path(np.array([0., .1])),
                            visits, frequencies, state)
    positions = rng.uniform(-6, 6, (24, 2, 2))
    positions[0] = [[3.49, 3.49], [-3.49, 0.]]  # Partial PSFs at camera edges.
    positions[1] = [[6., 6.], [-6., -6.]]        # Outside; no stamp renormalization.
    actual_n = positions_nufft(geometry, positions*ARCSEC, numerator).real
    actual_i = positions_nufft(geometry, positions*ARCSEC, information).real
    expected_n, expected_i = direct_pixel_statistics(images, variances, sigma, positions, masks=masks)
    np.testing.assert_allclose(actual_n, expected_n, atol=2e-10)
    np.testing.assert_allclose(actual_i, expected_i, atol=2e-10)
    assert not report["periodic_camera_assumed"]
    assert max(x["template_absolute_error_bound"] for x in report["exposures"]) < 1e-16
    # A deliberately short spectrum has a non-negligible, meaningful bound.
    visits, frequencies, numerator, information, coarse = finite_pixel_fields(
        images, variances, sigma, ARCSEC, masks=masks, source_half_width_pixels=6, spectral_sigma=3)
    geometry.visits, geometry.frequencies = visits, frequencies
    approximate_n = positions_nufft(geometry, positions*ARCSEC, numerator).real
    approximate_i = positions_nufft(geometry, positions*ARCSEC, information).real
    n_bound = sum(row["numerator_model_error_bound"] for row in coarse["exposures"])
    i_bound = sum(row["information_model_error_bound"] for row in coarse["exposures"])
    assert max(abs(approximate_n-expected_n)) <= n_bound+2e-10
    assert max(abs(approximate_i-expected_i)) <= i_bound+2e-10
    with pytest.raises(ValueError, match="budget"):
        finite_pixel_fields(images, variances, sigma, ARCSEC, max_modes=10)


def test_chart_search_manifest_round_trip_and_explicit_refusal(tmp_path):
    import json
    from research.kepler_transform.chart_search import run
    chart, center, widths, geometry, data = fixture()
    parameters = center+np.random.default_rng(47).uniform(-.9, .9, (8, 6))*widths
    np.savez(tmp_path/"arrays.npz", times=geometry.times, observers=geometry.observers,
             visits=geometry.visits, frequencies=geometry.frequencies, data=data, parameters=parameters,
             normal=geometry.normal, east=geometry.east, north=geometry.north)
    manifest = {"schema_version": 1, "arrays": "arrays.npz", "chart": chart.to_dict(),
                "center": center.tolist(), "widths": widths.tolist(),
                "position_tolerance_arcsec": .0005, "max_leaves": 1, "audit_count": 1}
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    report = run(path, tmp_path/"success")
    assert report["status"] == "evaluated_all_queries"
    with np.load(tmp_path/"success"/"scores.npz") as output:
        expected = physical_kernel(geometry, chart.states(parameters)).direct(data)
        np.testing.assert_allclose(output["scores"], expected, atol=1e-7)
    manifest["position_tolerance_arcsec"] = 1e-20
    path.write_text(json.dumps(manifest))
    report = run(path, tmp_path/"refused")
    assert report["status"] == "evaluation_refused"
    assert not (tmp_path/"refused"/"scores.npz").exists()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_fifth_variational_implicit_derivative_against_high_precision(axis):
    mp = pytest.importorskip("mpmath")
    from flint import arb, ctx
    from research.kepler_transform.quartic_atlas import fifth_coefficients
    chart, center, widths, geometry, _ = fixture(span=2.)
    with ctx.workprec(128), mp.workdps(70):
        observer, normal, east, north = [getattr(geometry, a)[0] for a in ("observers", "normal", "east", "north")]
        time = float(geometry.times[0])
        enclosed = fifth_coefficients(chart, center, widths, axis, time, observer, normal, east, north)
        for point in [-.3, .5]:
            def coordinate(z):
                q = [mp.mpf(float(c))+(z if j == axis else mp.mpf(".27"))*mp.mpf(float(w))
                     for j, (c, w) in enumerate(zip(center, widths))]
                return mp_coordinate(chart, q, time, observer, normal, east)
            expected = mp.diff(coordinate, mp.mpf(point), 5)/120
            assert enclosed[0].contains(arb(mp.nstr(expected, 60)))


def test_quartic_interpolation_constants_and_certificate():
    from research.kepler_transform.quartic_atlas import QuarticChartAtlas, interpolation_constants
    constants = interpolation_constants()
    assert constants["lebesgue_upper"] < 3
    assert constants["nodal_product_upper"] < .125
    chart, center, widths, geometry, data = fixture()
    atlas = QuarticChartAtlas(chart, center, widths, geometry)
    certificate = atlas.certify()
    rng = np.random.default_rng(48)
    outer, inner = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, (32, 3))
    points = np.column_stack((np.tile(outer, (len(inner), 1)), inner))
    plan = atlas.slice(outer)
    expected_kernel = physical_kernel(geometry, atlas.states(points))
    expected = expected_kernel.projected(np.arange(len(inner)), np.arange(len(geometry.times)))
    from research.kepler_transform.local_transform import monomials
    actual = (monomials(inner) @ plan.position_coefficients.reshape(10, -1)).reshape(expected.shape)
    assert np.all(abs(actual-expected) <= np.asarray(certificate["position_error_radians"])+2e-14)
    assert plan.phase_tail_bound(data) <= atlas.phase_tail_bound(data)*1.001


def test_profiled_background_and_flux_scale_fields_equal_dense_nuisance_projection():
    from research.kepler_transform.detector_fields import finite_pixel_fields, direct_pixel_statistics
    from research.kepler_transform.chart_cover import positions_nufft
    rng = np.random.default_rng(49)
    images = 300+rng.normal(size=(1, 7, 7))
    variance, sigma = rng.uniform(.5, 2, images.shape), [[1., 1.4]]
    positions = rng.uniform(-3, 3, (16, 1, 2))
    visits, frequencies, numerator, information, report = finite_pixel_fields(images, variance, sigma,
        ARCSEC, profile_background=True, flux_scales=[2.3])
    chart, center, _, _, _ = fixture()
    state = chart.states(center[None])[0]
    geometry = KeplerKernel(state[None], np.array([0.]), chart.observer[None], visits, frequencies, state)
    actual_n = positions_nufft(geometry, positions*ARCSEC, numerator).real
    actual_i = positions_nufft(geometry, positions*ARCSEC, information).real
    expected_n, expected_i = direct_pixel_statistics(images, variance, sigma, positions,
        profile_background=True, flux_scales=[2.3])
    np.testing.assert_allclose(actual_n, expected_n, atol=3e-10)
    np.testing.assert_allclose(actual_i, expected_i, atol=3e-10)
    assert report["background"] == "independent profiled constant per exposure"
    shifted_n, shifted_i = direct_pixel_statistics(images+1e4, variance, sigma, positions,
        profile_background=True, flux_scales=[2.3])
    np.testing.assert_allclose(shifted_n, expected_n, atol=3e-10)
    np.testing.assert_allclose(shifted_i, expected_i, atol=3e-10)


@pytest.mark.parametrize("case", ["bound", "mixed_energy", "close_observer"])
def test_compiled_direct_reference_matches_all_conic_oracle(case):
    pytest.importorskip("numba")
    from research.kepler_transform.compiled_reference import projected
    from research.kepler_transform.kernel import physical_problem
    kernel, _, _, _ = physical_problem(48, 16, case=case)
    actual = projected(kernel.states, kernel)
    expected = kernel.projected(np.arange(len(kernel.states)), np.arange(len(kernel.times)))
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=1e-10)


def test_compiled_chart_reference_avoids_epoch_conversion_without_changing_physics():
    pytest.importorskip("numba")
    from research.kepler_transform.compiled_reference import projected_chart
    chart, center, widths, geometry, _ = fixture(frames=3, span=8.)
    q = center+np.random.default_rng(50).uniform(-.99, .99, (32, 6))*widths
    actual = projected_chart(chart, q, geometry)
    expected = physical_kernel(geometry, chart.states(q)).projected(np.arange(len(q)), np.arange(3))
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=1e-10)


def test_uniform_phase_budget_selects_zero_order_and_bounds_unsampled_queries():
    from research.kepler_transform.phase_budget import choose_order
    chart, center, widths, geometry, data = fixture()
    cover = ChartCover(chart, center, widths, geometry, position_tolerance=1e-8,
                       max_leaves=1, phase_tolerance=1e-4)
    atlas = cover.atlases[""]
    policy = choose_order(atlas, 1e-4)
    assert policy["selected_order"] == 0
    rng = np.random.default_rng(51)
    for outer in rng.uniform(-1, 1, (2, 3)):
        inner = rng.uniform(-1, 1, (32, 3))
        plan = atlas.slice(outer, correction_order=0)
        actual = plan.apply(data, inner)
        truth = plan.quadratic_direct(data, inner)
        assert max(abs(actual-truth)) <= policy["operator_bounds_by_order"][0]*sum(abs(data))+1e-8
        parameters = center+np.column_stack((np.tile(outer, (len(inner), 1)), inner))*widths
        result, report = cover.apply(data, parameters, route="slices")
        np.testing.assert_allclose(result, actual, atol=1e-8)
        assert report["phase_orders_used"] == {"0": len(inner)}


@pytest.mark.parametrize("order,axis", [(3, 0), (3, 1), (3, 3), (5, 0), (5, 1), (5, 3)])
def test_translated_variational_flow_against_independent_implicit_derivatives(order, axis):
    mp = pytest.importorskip("mpmath")
    from flint import arb, ctx
    from research.kepler_transform.quartic_atlas import fifth_coefficients
    chart, center, widths, geometry, _ = fixture(span=2.)
    widths[0] *= 5
    function = third_coefficients if order == 3 else fifth_coefficients
    with ctx.workprec(128), mp.workdps(70):
        observer, normal, east, north = [getattr(geometry, a)[0] for a in ("observers", "normal", "east", "north")]
        time = float(geometry.times[0])
        enclosed = function(chart, center, widths, axis, time, observer, normal, east, north, translated=True)
        for point in [-.4, .6]:
            def coordinate(z):
                q = [mp.mpf(float(c))+(z if j == axis else mp.mpf(".23"))*mp.mpf(float(w))
                     for j, (c, w) in enumerate(zip(center, widths))]
                return mp_coordinate(chart, q, time, observer, normal, east)
            expected = mp.diff(coordinate, mp.mpf(point), order)/mp.factorial(order)
            assert enclosed[0].contains(arb(mp.nstr(expected, 60)))


def test_translated_atlas_certifies_unsampled_full_six_dimensional_queries():
    chart, center, widths, geometry, _ = fixture()
    atlas = ChartAtlas(chart, center, widths, geometry, translated=True)
    certificate = atlas.certify()
    points = np.random.default_rng(52).uniform(-.99, .99, (24, 6))
    exact = physical_kernel(geometry, atlas.states(points)).projected(np.arange(len(points)), np.arange(1))
    assert np.all(abs(polynomial_positions(atlas, points)-exact) <=
                  np.asarray(certificate["position_error_radians"])+2e-14)
    assert certificate["translated_certificate"]
