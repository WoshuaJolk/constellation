"""Independent identities, adversarial failures and continuous-domain checks."""

import importlib.util
from itertools import product

import numpy as np
import pytest

from magictools.orbits import GM
from research.kepler_transform.butterfly import Butterfly
from research.kepler_transform.factorization import ARCSEC
from research.kepler_transform.kernel import KeplerKernel, physical_problem
from research.kepler_transform.local_transform import LocalTransform, pixel_fields, monomials


def research_dependencies():
    for name in ("flint", "finufft"):
        if importlib.util.find_spec(name) is None:
            pytest.skip("Install the research extra")


def geometry_fixture(case="bound", frames=2, modes=32):
    source, _, _, data = physical_problem(8, modes, case=case)
    take = np.linspace(0, len(source.times)-1, frames).astype(int)
    radius = 1.02 if case == "close_observer" else 3.
    factor = {"bound": 1., "mixed_energy": np.sqrt(2), "close_observer": 1.8}[case]
    center = np.array([radius, 0., 0., 0., factor*np.sqrt(GM["sun"]/radius), 0.])
    basis = np.zeros((3, 6))
    basis[0, 1] = basis[1, 2] = (radius-1)*10*ARCSEC
    basis[2, 4] = (radius-1)*2*ARCSEC
    kernel = KeplerKernel(center[None], source.times[take], source.observers[take],
                          source.visits % frames, source.frequencies, center)
    return center, basis, kernel, data


def test_butterfly_rectangular_partition_adjoint_and_damped_inverse():
    rng = np.random.default_rng(21)
    x = np.arange(57)/57
    # Duplicate columns create a known nullspace: inverse is not identifiable.
    y = np.tile(np.arange(13), 2)
    matrix = np.exp(2j*np.pi*x[:, None]*y)
    plan = Butterfly(lambda r, c: matrix[np.ix_(r, c)], x[:, None], y[:, None], rank_cap=16, leaf_size=1)
    a, b = rng.normal(size=26)+1j*rng.normal(size=26), rng.normal(size=57)+1j*rng.normal(size=57)
    np.testing.assert_allclose(plan.apply(a), matrix @ a, atol=1e-11)
    np.testing.assert_allclose(np.vdot(plan.apply(a), b), np.vdot(a, plan.apply(b, adjoint=True)), atol=1e-10)
    recovered, report = plan.inverse(matrix @ a, damping=.1, max_iterations=200)
    expected = np.linalg.lstsq(np.vstack((matrix, .1*np.eye(26))), np.r_[matrix @ a, np.zeros(26)], rcond=None)[0]
    np.testing.assert_allclose(recovered, expected, atol=1e-9)
    assert not report["unique_image_reconstruction_claimed"]
    assert not report["iteration_limit_reached"]
    np.testing.assert_allclose(plan.apply(np.r_[np.ones(13), -np.ones(13)]), 0, atol=1e-12)


def test_butterfly_rank_failure_never_claims_certification():
    rng = np.random.default_rng(22)
    matrix = rng.normal(size=(128, 128))+1j*rng.normal(size=(128, 128))
    plan = Butterfly(lambda r, c: matrix[np.ix_(r, c)], rng.normal(size=(128, 3)),
                     rng.normal(size=(128, 2)), rank_cap=2, leaf_size=1)
    assert plan.report["rank_limited_blocks"] > 0
    assert not plan.report["finite_operator_error_certified"]
    assert np.linalg.norm(plan.apply(np.ones(128))-matrix @ np.ones(128)) > 1
    _, report = plan.inverse(np.ones(128), max_iterations=1)
    assert report["iteration_limit_reached"]


@pytest.mark.parametrize("case", ["bound", "mixed_energy", "close_observer"])
def test_local_transform_against_exact_retarded_geometry_and_phase_tail(case):
    research_dependencies()
    center, basis, kernel, data = geometry_fixture(case)
    plan = LocalTransform(center, basis, kernel, correction_order=1)
    z = np.random.default_rng(23).uniform(-1, 1, (80, 3))
    truth_kernel = KeplerKernel(center+z @ basis, kernel.times, kernel.observers,
                               kernel.visits, kernel.frequencies, center)
    truth = truth_kernel.direct(data)
    output = plan.apply(data, z)
    quadratic = plan.quadratic_direct(data, z)
    assert max(abs(output-quadratic)) <= plan.phase_tail_bound(data)+1e-8
    np.testing.assert_allclose(output, truth, atol=2e-7)
    assert plan.report["geometry_state_time_pairs"] == 27*len(kernel.times)


def test_pixel_fields_equal_direct_whitened_templates_including_masks_and_information():
    rng = np.random.default_rng(24)
    n, sigma = 7, 1.1
    images = rng.normal(size=(2, n, n))
    variance = rng.uniform(.5, 3., images.shape)
    mask = rng.random(images.shape) < .2
    visits, frequencies, numerator, information = pixel_fields(images, variance, sigma, 1., masks=mask)
    integer = np.arange(-(n//2), n//2+1)
    yy, xx = np.meshgrid(integer, integer, indexing="ij")
    modes = np.column_stack((xx.ravel(), yy.ravel()))*2*np.pi/n
    psf = np.exp(-.5*sigma**2*np.sum(modes**2, axis=1))*np.prod(np.sinc(modes/(2*np.pi)), axis=1)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    for xy in rng.uniform(-2, 2, (5, 2)):
        template = (np.exp(1j*(pixels-xy) @ modes.T) @ psf/n**2).real.reshape(n, n)
        weight = np.where(mask, 0., 1/variance)
        expected_n = np.sum(images*weight*template)
        expected_i = np.sum(weight*template**2)
        phases = np.exp(1j*frequencies @ xy)
        np.testing.assert_allclose(phases @ numerator, expected_n, atol=1e-12)
        np.testing.assert_allclose(phases @ information, expected_i, atol=1e-12)


def test_local_adjoint_and_inverse_agree_with_dense_linear_algebra():
    research_dependencies()
    center, basis, kernel, data = geometry_fixture(frames=2, modes=8)
    plan = LocalTransform(center, basis, kernel)
    rng = np.random.default_rng(29)
    targets = rng.uniform(-1, 1, (40, 3))
    values = rng.normal(size=40)+1j*rng.normal(size=40)
    np.testing.assert_allclose(np.vdot(plan.apply(data, targets), values),
        np.vdot(data, plan.adjoint(values, targets)), rtol=1e-10, atol=1e-9)
    # Independent explicit matrix for the truncated phase polynomial.
    matrix = np.exp(1j*targets @ plan.linear.T)*(monomials(targets, plan.powers) @ plan.coefficients)
    expected = np.linalg.lstsq(np.vstack((matrix, .3*np.eye(8))), np.r_[values, np.zeros(8)], rcond=None)[0]
    recovered, report = plan.inverse(values, targets, damping=.3, tolerance=1e-11)
    np.testing.assert_allclose(recovered, expected, atol=1e-8)
    assert not report["iteration_limit_reached"]


def test_jet_arithmetic_matches_independent_power_series():
    research_dependencies()
    from flint import arb, ctx
    from research.kepler_transform.jet_geometry import Jet
    with ctx.workprec(128):
        x = Jet([4, 1, 0, 0])
        squared = x.sqrt()*x.sqrt()
        for actual, expected in zip(squared.c, [4, 1, 0, 0]):
            assert actual.contains(expected)
        inverse = x.reciprocal()
        for n, value in enumerate(inverse.c):
            assert value.contains(arb((-1)**n)/4**(n+1))


@pytest.mark.parametrize("case,axis", [("bound", 0), ("mixed_energy", 2), ("close_observer", 0), ("close_observer", 1)])
def test_retarded_third_derivative_encloses_high_precision_implicit_solution(case, axis):
    research_dependencies()
    mp = pytest.importorskip("mpmath")
    from flint import arb, ctx
    from research.detection_limit.interval_orbits import MU, C_AU_DAY
    from research.kepler_transform.jet_geometry import projected_third_coefficients
    center, basis, kernel, _ = geometry_fixture(case)
    time = float(kernel.times[0])
    observer, normal, east = kernel.observers[0], kernel.normal[0], kernel.east[0]

    def coordinate(z):
        mu, speed = mp.mpf(MU), mp.mpf(C_AU_DAY)
        state = [mp.mpf(float(c))+z*mp.mpf(float(b)) for c, b in zip(center, basis[axis])]
        r, v = state[:3], state[3:]
        dot = lambda a, b: sum(x*y for x, y in zip(a, b))
        radius = mp.sqrt(dot(r, r))
        alpha = 2/radius-dot(v, v)/mu

        def position(chi):
            argument = alpha*chi**2
            # Entire Stumpff series; avoids a conic branch near zero energy.
            c = sum((-argument)**k/mp.factorial(2*k+2) for k in range(30))
            s = sum((-argument)**k/mp.factorial(2*k+3) for k in range(30))
            elapsed = (dot(r, v)/mp.sqrt(mu)*chi**2*c+(1-alpha*radius)*chi**3*s+radius*chi)/mp.sqrt(mu)
            f, g = 1-chi**2*c/radius, elapsed-chi**3*s/mp.sqrt(mu)
            return elapsed, [f*x+g*y-mp.mpf(float(o)) for x, y, o in zip(r, v, observer)]

        def equation(chi):
            elapsed, delta = position(chi)
            return elapsed+mp.sqrt(dot(delta, delta))/speed-mp.mpf(time)

        chi = mp.findroot(equation, mp.sqrt(mu)*time/radius)
        _, delta = position(chi)
        return dot(delta, [mp.mpf(float(x)) for x in east])/dot(delta, [mp.mpf(float(x)) for x in normal])

    with ctx.workprec(128), mp.workdps(60):
        box = [arb(float(x), abs(float(b))*1.1) for x, b in zip(center, basis[axis])]
        derivative = projected_third_coefficients(box, basis[axis], time, observer, normal,
                                                 east, kernel.north[0])[0]
        for point in [-.7, 0., .8]:
            expected = mp.diff(coordinate, mp.mpf(point), 3)/6
            assert derivative.contains(arb(mp.nstr(expected, 55)))


def test_continuous_geometry_certificate_bounds_unsampled_inner_points():
    research_dependencies()
    from research.kepler_transform.geometry_certificate import certify_geometry
    center, basis, kernel, _ = geometry_fixture()
    plan = LocalTransform(center, basis, kernel)
    certificate = certify_geometry(plan)
    targets = np.random.default_rng(25).uniform(-1, 1, (30, 3))
    exact = KeplerKernel(center+targets @ basis, kernel.times, kernel.observers,
                         kernel.visits, kernel.frequencies, center)
    projected = exact.projected(np.arange(len(targets)), np.arange(len(kernel.times)))
    approximation = (monomials(targets) @ plan.position_coefficients.reshape(10, -1)).reshape(projected.shape)
    # Float reference has its own rounding; the mathematical certificate does
    # not certify the independent binary64 universal-variable implementation.
    assert np.all(abs(projected-approximation) <= np.array(certificate["position_error_radians"])+2e-14)
    assert certificate["physical_geometry_error_certified"]
    assert not certificate["full_6d_coverage_certified"]
    from research.kepler_transform.geometry_certificate import score_error_bound
    plan.position_coefficients[0, 0, 0] += 1e-6
    with pytest.raises(ValueError, match="different local polynomial"):
        score_error_bound(plan, certificate, np.ones(len(kernel.visits)))


def test_six_dimensional_atlas_off_grid_and_certificate():
    research_dependencies()
    from research.kepler_transform.atlas import GeometryAtlas
    center, inner, kernel, data = geometry_fixture(frames=1)
    basis = np.zeros((6, 6))
    basis[0, 0], basis[1, 3], basis[2, 5] = 1e-4, 1e-6, 4*ARCSEC
    basis[3:] = inner
    atlas = GeometryAtlas(center, basis, kernel)
    certificate = atlas.certify()
    rng = np.random.default_rng(26)
    for outer in rng.uniform(-1, 1, (2, 3)):
        plan = atlas.slice(outer)
        targets = rng.uniform(-1, 1, (20, 3))
        states = center+outer @ basis[:3]+targets @ basis[3:]
        truth = KeplerKernel(states, kernel.times, kernel.observers, kernel.visits,
                             kernel.frequencies, center).direct(data)
        np.testing.assert_allclose(plan.apply(data, targets), truth, atol=1e-6)
        assert plan.phase_tail_bound(data) <= atlas.phase_tail_bound(data)*1.001
        from research.kepler_transform.geometry_certificate import score_error_bound
        assert score_error_bound(plan, certificate, data) > 0
    assert certificate["full_6d_cell_covered"]
    assert not certificate["full_objective_complete"]


def test_same_geometry_fast_baseline_matches_explicit_quadratic_sum():
    research_dependencies()
    from research.kepler_transform.comparison_experiments import polynomial_nufft
    center, basis, kernel, data = geometry_fixture()
    plan = LocalTransform(center, basis, kernel)
    points = np.random.default_rng(30).uniform(-1, 1, (32, 3))
    np.testing.assert_allclose(polynomial_nufft(plan, data, points),
                               plan.quadratic_direct(data, points), atol=1e-9)


def test_continuous_score_budget_preserves_exact_cover_and_failed_cells():
    research_dependencies()
    from research.kepler_transform.continuous import build_atlas, verify_partition
    center, _, kernel, data = geometry_fixture(frames=1, modes=8)
    result = build_atlas(center, np.full(6, 1e-5), kernel, data, tolerance=1e-12, max_cells=4)
    assert verify_partition(result)
    assert not result["all_cells_resolved"]
    assert len(result["leaves"]) == 4
    broken = dict(result, leaves=result["leaves"][:-1])
    with pytest.raises(ValueError, match="cover"):
        verify_partition(broken)
    failed = build_atlas(np.zeros(6), np.ones(6), kernel, data, max_cells=1)
    assert verify_partition(failed)
    assert failed["leaves"][0]["absolute_error_bound"] is None


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        Butterfly(None, [[0]], [[0]], rank_cap=0)
    center, basis, kernel, _ = geometry_fixture()
    with pytest.raises(ValueError):
        LocalTransform(center, basis*0, kernel)
    with pytest.raises(ValueError):
        pixel_fields(np.zeros((1, 4, 4)), 1, 1, 1)
    from research.kepler_transform.local_transform import check_grid_budget
    with pytest.raises(ValueError, match="phase-volume budget"):
        check_grid_budget(np.array([[-1e4]*3, [1e4]*3]), np.array([[-1]*3, [1]*3]))
