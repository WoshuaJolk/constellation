"""Independent inference, probability accounting and enclosure tests."""

from copy import deepcopy
from itertools import product
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, logsumexp
from scipy.stats import multivariate_normal, poisson

pytest.importorskip("flint")
from flint import arb, ctx
from research.fit_sky.arithmetic import Jet, square
from research.fit_sky.graph import infer as graph_infer, brute_force, mix_shared_nuisance
from research.fit_sky.integrate import infer, verify, enclose
from research.fit_sky.scene import Scene
from research.fit_sky.reference import render
from research.fit_sky.sampling import sample


def scalar_scene(flux=(1, 3), count=(.5, .5), data=1.):
    return {"source_types": [{"name": "star", "kind": "star", "parameters": {
        "x": [0, 0], "y": [0, 0], "flux": list(flux)}}],
        "exposures": [{"id": "one", "time": 0, "data": [[data]], "psf_sigma": .6}],
        "count_prior": {"probabilities": list(count)}}


@pytest.mark.parametrize("seed", range(5))
def test_graph_matches_pixel_enumeration(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(9, 7))
    y, prior = rng.normal(size=9), rng.uniform(.05, .9, 7)
    result = graph_infer(y, a, prior)
    z, marginals = brute_force(y, a, prior)
    assert result["log_evidence"] == pytest.approx(z, abs=2e-13)
    assert [q["probability"] for q in result["queries"].values()] == pytest.approx(marginals, abs=2e-13)


def test_joint_blends_not_independent_detections():
    a = np.ones((1, 2))*3
    y = np.array([3.])
    joint = graph_infer(y, a, [.5, .5], queries={"both": {0: 1, 1: 1}, "first": {0: 1}})
    assert joint["queries"]["both"]["probability"] < .01
    assert joint["queries"]["first"]["probability"] == pytest.approx(.5)
    individual = graph_infer(y, a[:, :1], [.5])
    assert individual["queries"]["source_0"]["probability"] > .98


def test_sparse_graph_does_not_enumerate_catalogues():
    # 2**70 catalogue alternatives, but isolated sources factor exactly.
    n = 70
    result = graph_infer(np.ones(n), np.eye(n), np.full(n, .2), max_entries=4)
    assert result["peak_factor_entries"] == 2
    expected = 1/(1+4*np.exp(-.5))
    assert all(q["probability"] == pytest.approx(expected) for q in result["queries"].values())


def test_graph_omitted_interaction_bounds_and_width_failure():
    rng = np.random.default_rng(12)
    a, y = rng.uniform(0, .5, (8, 6)), rng.normal(size=8)
    exact = graph_infer(y, a, np.full(6, .3))
    approximate = graph_infer(y, a, np.full(6, .3), drop_threshold=.4)
    lo, hi = approximate["log_evidence_interaction_bound"]
    assert lo <= exact["log_evidence"] <= hi
    for name, q in exact["queries"].items():
        lo, hi = approximate["queries"][name]["interaction_bound"]
        assert lo <= q["probability"] <= hi
    with pytest.raises(ValueError, match="Interaction width"):
        graph_infer(y, a, np.full(6, .3), max_entries=4)


def test_shared_nuisance_marginalization_uses_joint_evidence():
    a = np.array([[2., 0], [0, 2.]])
    y = np.array([2., 2.])
    reports = [graph_infer(y-b, a, [.2, .2], queries={"both": {0: 1, 1: 1}})
               for b in [0., 2.]]
    mixed = mix_shared_nuisance(reports, [.5, .5])
    terms, event_terms = [], []
    for b in [0., 2.]:
        for state in product([0, 1], repeat=2):
            prior = .5*np.prod([.2 if s else .8 for s in state])
            weight = prior*multivariate_normal.pdf(y, mean=a@state+b, cov=np.eye(2))
            terms.append(weight)
            if all(state):
                event_terms.append(weight)
    assert mixed["log_evidence"] == pytest.approx(np.log(sum(terms)))
    assert mixed["queries"]["both"] == pytest.approx(sum(event_terms)/sum(terms))


@pytest.mark.parametrize("function", ["exp", "erf", "sqrt", "log", "likelihood"])
def test_fourth_derivative_jets_against_mpmath(function):
    import mpmath as mp
    mp.mp.dps = 65
    with ctx.workprec(180):
        x = Jet(arb("1.25"), arb(".3"))
        if function == "likelihood":
            value = (-square(x.erf()-arb(".4"))/2).exp()
            f = lambda t: mp.exp(-(mp.erf(mp.mpf("1.25")+mp.mpf(".3")*t)-mp.mpf(".4"))**2/2)
        else:
            value = getattr(x, function)()
            f = lambda t: getattr(mp, function)(mp.mpf("1.25")+mp.mpf(".3")*t)
        for order in range(5):
            target = mp.diff(f, 0, order)/mp.factorial(order)
            assert abs(float(value.c[order])-float(target)) < 1e-14
            assert value.c[order].contains(arb(mp.nstr(target, 55)))


@pytest.mark.parametrize("method", ["range", "gauss", "hybrid"])
def test_continuous_unknown_count_against_independent_integral(method):
    manifest = scalar_scene()
    report = infer(manifest, method=method, max_leaves=2048, max_seconds=30)
    response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
    one = quad(lambda f: np.exp(-.5*(1-f*response)**2)/2, 1, 3, epsabs=1e-13)[0]
    zero = np.exp(-.5)
    evidence = .5*(one+zero)
    assert report["scaled_evidence"][0] <= evidence <= report["scaled_evidence"][1]
    assert report["queries"]["has:star"][0] <= one/(one+zero) <= report["queries"]["has:star"][1]
    assert report["status"] == "bounded_target_met"
    assert verify(json.loads(json.dumps(report)))["valid"]


def test_gauss_uses_fewer_cells_than_range_and_handles_boundary_query():
    m = scalar_scene()
    ranged = infer(m, method="range", max_seconds=30)
    gauss = infer(m, max_seconds=30)
    assert gauss["work"]["leaves"] < ranged["work"]["leaves"]
    report = infer(m, queries=["any:star:flux:1:2"], max_leaves=2048, max_seconds=30)
    response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
    numerator = .5*quad(lambda f: np.exp(-.5*(1-f*response)**2)/2, 1, 2)[0]
    z = .5*(np.exp(-.5)+quad(lambda f: np.exp(-.5*(1-f*response)**2)/2, 1, 3)[0])
    a, b = report["queries"]["any:star:flux:1:2"]
    assert a <= numerator/z <= b
    assert b-a <= .01


def test_poisson_control_including_empty_catalogue():
    m = scalar_scene(flux=(.3, 4), data=2.)
    m["noise"] = "poisson"
    report = infer(m, max_leaves=512, max_seconds=30)
    response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
    z = .5*quad(lambda f: poisson.pmf(2, f*response)/3.7, .3, 4)[0]
    assert report["log_evidence"][0] <= np.log(z) <= report["log_evidence"][1]
    assert report["queries"]["has:star"][0] > .99


def test_poisson_tail_is_not_renormalized_away():
    m = scalar_scene(flux=(0, 0))
    m["count_prior"] = {"poisson_mean": 2., "max_count": 1}
    report = infer(m, queries=["count=0"], max_leaves=8)
    assert report["status"] == "unresolved"
    lo, hi = report["queries"]["count=0"]
    assert lo <= np.exp(-2) <= hi
    assert report["source_count_tail_prior_upper"] >= poisson.sf(1, 2)


def test_unordered_class_weights_are_multinomial():
    m = scalar_scene(flux=(0, 0), count=(0, 0, 1))
    m["source_types"].append({"name": "other", "kind": "star", "weight": 2.,
                              "parameters": deepcopy(m["source_types"][0]["parameters"])})
    report = infer(m)
    # Two iid draws, P(at least one first class)=1-(2/3)**2.
    a, b = report["queries"]["has:star"]
    assert a <= 5/9 <= b
    assert report["work"]["roots"] == 3


def test_shared_gaussian_linear_nuisance_against_mvn():
    m = scalar_scene(flux=(2., 2.))
    m["exposures"][0]["data"] = [[1., -1.]]
    m["linear_nuisance"] = {"design": [[1.], [1.]], "mean": [.2], "covariance": [[.7]]}
    report = infer(m)
    image = render(m["exposures"], [{"kind": "star", "parameters": {"x": 0., "y": 0., "flux": 2.}}])[0].ravel()
    covariance = np.eye(2)+.7*np.ones((2, 2))
    zero = multivariate_normal.pdf([1, -1], mean=[.2, .2], cov=covariance)
    one = multivariate_normal.pdf([1, -1], mean=image+.2, cov=covariance)
    assert report["log_evidence"][0] <= np.log(.5*(zero+one)) <= report["log_evidence"][1]
    a, b = report["queries"]["has:star"]
    assert a <= one/(zero+one) <= b


def test_motion_parallax_variability_galaxy_and_instrument_forward():
    exposures = [{"id": str(i), "time": t, "instrument": "a" if i < 2 else "b",
                  "data": np.zeros((3, 3)).tolist(), "psf_sigma": .6+.1*i,
                  "flux_scale": 1+.2*i, "band_coordinate": i-1,
                  "parallax_x": np.sin(t), "parallax_y": np.cos(t),
                  "affine": [[1., .1], [-.1, 1.]], "offset": [.1, -.2]}
                 for i, t in enumerate([-1., 0., 2.])]
    concrete = [{"kind": "star", "parameters": {"x": .1, "y": -.1, "vx": .2, "vy": -.3,
                "parallax": .4, "flux": 4., "variability": -.1, "color": .2}},
                {"kind": "galaxy", "parameters": {"x": -.6, "y": .5, "flux": 7.,
                                                  "sigma_x": .8, "sigma_y": .3}}]
    nuisance = {"a_dx": .1, "b_dy": -.2, "a_log_psf": .1, "b_log_gain": -.2}
    truth = render(exposures, concrete, nuisance)
    m = {"exposures": exposures, "source_types": [
        {"name": str(i), "kind": s["kind"], "parameters": {k: [v, v] for k, v in s["parameters"].items()}}
        for i, s in enumerate(concrete)], "nuisance": {k: [v, v] for k, v in nuisance.items()},
        "count_prior": {"probabilities": [0, 0, 1]}}
    scene = Scene(m)
    root = next(r for r in scene.roots() if r["classes"] == [0, 1])
    actual = scene.means(root, [])
    assert np.array(list(map(float, actual))) == pytest.approx(np.concatenate([x.ravel() for x in truth]), abs=2e-13)


def test_kepler_pixel_model_against_independent_ode():
    from research.detection_limit.interval_orbits import MU
    e = {"id": "k", "time": .1, "data": [[0.]], "psf_sigma": 1.,
         "observer": [0., 0., 0.], "center": [1., 0., 0.],
         "east": [0., 1., 0.], "north": [0., 0., 1.], "pixel_scale_arcsec": 100.,
         "light_time_max": .5}
    p = {"rx": 2., "ry": 0., "rz": 0., "vx": 0., "vy": np.sqrt(float(MU)/2), "vz": 0., "flux": 5.}
    m = {"exposures": [e], "source_types": [{"name": "orbit", "kind": "kepler",
         "parameters": {k: [float(v), float(v)] for k, v in p.items()}}],
         "count_prior": {"probabilities": [0, 1]}}
    with ctx.workprec(128):
        scene = Scene(m)
        actual = scene.means(scene.roots()[0], [])[0]
        expected = render([e], [{"kind": "kepler", "parameters": p}])[0][0, 0]
        assert abs(float(actual)-expected) < 1e-10


@pytest.mark.parametrize("damage", ["bound", "partition", "posterior", "manifest", "status"])
def test_certificate_rejects_tampering(damage):
    report = infer(scalar_scene(), method="range", max_leaves=64)
    if damage == "bound":
        report["certificate"][-1]["likelihood"] = [1., 1.]
    elif damage == "partition":
        report["certificate"].pop()
    elif damage == "posterior":
        report["queries"]["has:star"] = [0., 0.]
    elif damage == "manifest":
        report["manifest"]["exposures"][0]["data"] = [[99.]]
    elif damage == "status":
        report["status"] = "bounded_target_met" if report["status"] == "unresolved" else "unresolved"
    with pytest.raises(ValueError):
        verify(report)


def test_posterior_draws_have_unknown_counts_and_continuous_flux():
    report = infer(scalar_scene())
    draws = sample(report, count=2000, seed=19)
    assert draws["complete"]
    counts = [len(d["sources"]) for d in draws["catalogues"]]
    a, b = report["queries"]["has:star"]
    assert a-.04 <= np.mean(counts) <= b+.04
    fluxes = [d["sources"][0]["parameters"]["flux"] for d in draws["catalogues"] if d["sources"]]
    assert min(fluxes) >= 1 and max(fluxes) <= 3
    assert len(set(fluxes)) == len(fluxes)


def test_budget_failure_retains_mass_and_does_not_claim_completion():
    m = scalar_scene(flux=(0., 50.), count=(.2, .3, .5), data=4.)
    m["source_types"][0]["parameters"]["x"] = [-10, 10]
    r = infer(m, max_leaves=3, max_seconds=1)
    assert r["status"] == "unresolved"
    assert not r["full_objective_complete"]
    assert verify(r)["valid"]


def test_duplicate_observation_rejected():
    m = scalar_scene()
    m["exposures"].append(deepcopy(m["exposures"][0]))
    with pytest.raises(ValueError, match="only once"):
        infer(m)


def test_analytic_flux_keeps_two_source_cross_terms():
    from scipy.integrate import dblquad
    m = scalar_scene(flux=(.5, 5), count=(0, 0, 1), data=2.)
    r = infer(m, method="hybrid", max_seconds=30)
    response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
    z = dblquad(lambda a, b: np.exp(-.5*(2-response*(a+b))**2)/4.5**2,
                .5, 5, lambda _: .5, lambda _: 5, epsabs=1e-12)[0]
    assert r["scaled_evidence"][0] <= z <= r["scaled_evidence"][1]
    assert r["status"] == "bounded_target_met"
    assert verify(r)["valid"]


def test_low_rank_shared_calibration_equals_dense_covariance():
    m = scalar_scene(flux=(.1, 6), data=1.)
    m["exposures"][0]["data"] = [[1., -.5, .3]]
    design = np.array([[1., -.3], [1., .2], [1., .8]])
    covariance = np.array([[.7, .1], [.1, .4]])
    m["linear_nuisance"] = {"design": design.tolist(), "mean": [0., 0.],
                            "covariance": covariance.tolist()}
    low_rank = infer(m, method="hybrid")
    dense = deepcopy(m)
    del dense["linear_nuisance"]
    matrix = np.eye(3)+design@covariance@design.T
    dense["covariance"] = ((matrix+matrix.T)/2).tolist()
    direct = infer(dense, method="hybrid")
    assert np.mean(low_rank["log_evidence"]) == pytest.approx(np.mean(direct["log_evidence"]), abs=1e-12)


def test_quadrature_average_is_not_used_as_sampling_envelope():
    # A very sharp one-dimensional likelihood has an average far below its peak.
    # The independent numerical integral supplies the posterior mean of the flux.
    m = scalar_scene(flux=(0., 50.), count=(0, 1), data=5.)
    r = infer(m, method="hybrid")
    assert r["work"]["leaves"] == 1
    draws = sample(r, count=1500, max_proposals=100000, seed=71)
    assert draws["complete"]
    values = [d["sources"][0]["parameters"]["flux"] for d in draws["catalogues"]]
    response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
    density = lambda f: np.exp(-.5*(5-response*f)**2)
    expected = quad(lambda f: f*density(f), 0, 50)[0]/quad(density, 0, 50)[0]
    assert abs(np.mean(values)-expected) < .2


def test_batch_renderer_agrees_with_interval_model():
    from research.fit_sky.experiments import synthetic
    from research.fit_sky.statistical import BatchModel
    m, _ = synthetic("broad")
    with ctx.workprec(100):
        scene = Scene(m)
        root = next(r for r in scene.roots() if r["classes"] == [0, 1, 2])
        bounds = np.array(root["ranges"])
        rng = np.random.default_rng(18)
        x = bounds[:, 0]+rng.random((16, len(bounds)))*(bounds[:, 1]-bounds[:, 0])
        reference = np.array([float(scene.likelihood(root, [arb(float(v)) for v in row])) for row in x])
        value = BatchModel(scene).likelihood(root, x)
        assert value == pytest.approx(reference, rel=1e-10, abs=1e-100)


def test_statistical_bounds_cover_independent_scalar_integral():
    from research.fit_sky.statistical import infer as statistical_infer
    m = scalar_scene()
    cover = infer(m, method="auto")
    report = statistical_infer(m, cover_report=cover, samples=100000, seed=900)
    exact = .607926040537572
    a, b = report["queries"]["has:star"]
    assert a < exact < b
    assert report["simultaneous_confidence"] == .99


def test_complete_prior_sampling_does_not_truncate_poisson_counts():
    from research.fit_sky.statistical import infer as statistical_infer
    m = scalar_scene(flux=(0, 0))
    m["count_prior"] = {"poisson_mean": 2., "max_count": 0}
    r = statistical_infer(m, samples=50000, seed=52, queries=["count=0"])
    assert r["maximum_sampled_source_count"] > 3
    assert r["represented_source_count_tail_upper"] == 0
    a, b = r["queries"]["count=0"]
    assert a < np.exp(-2) < b


def test_irrelevant_nuisance_derivative_is_zero():
    m = scalar_scene(count=(1.,))
    m["nuisance"] = {"default_dx": [-1., 1.]}
    r = infer(m, method="gauss")
    assert r["status"] == "bounded_target_met"
    assert verify(r)["valid"]


@pytest.mark.parametrize("part", ["source", "nuisance"])
def test_unsupported_parameters_cannot_be_silently_ignored(part):
    m = scalar_scene()
    if part == "source":
        m["source_types"][0]["parameters"]["unimplemented_morphology"] = [1, 2]
    else:
        m["nuisance"] = {"camera_that_is_not_used_dx": [0, 1]}
    with pytest.raises(ValueError):
        infer(m)


def test_dual_envelope_covers_continuous_flux_and_position_cells():
    from research.fit_sky.experiments import synthetic
    from research.fit_sky.envelope import bound
    from research.fit_sky.integrate import _box
    from research.fit_sky.statistical import BatchModel
    m, _ = synthetic("confused")
    with ctx.workprec(100):
        scene = Scene(m)
        root = scene.roots()[-1]
        path = [[1, 0], [3, 1], [1, 1], [3, 0]]  # retain continuous x and both fluxes
        ceiling, info = bound(scene, root, path)
        bounds = np.array([[float(a), float(b)] for a, b in _box(root, path)])
        rng = np.random.default_rng(651)
        points = bounds[:, 0]+rng.random((4096, len(bounds)))*(bounds[:, 1]-bounds[:, 0])
        likelihoods = BatchModel(scene).likelihood(root, points)
        assert np.max(likelihoods) <= float(ceiling)*(1+1e-12)
        assert info["status"] == "validated_dual_transfer"


def test_dual_envelope_for_fixed_templates_is_exact_at_interior_optimum():
    from research.fit_sky.envelope import bound
    m = scalar_scene(flux=(0, 10), count=(0, 1), data=1.)
    with ctx.workprec(100):
        scene = Scene(m)
        upper, info = bound(scene, scene.roots()[0], [])
        assert float(upper) == pytest.approx(1.)
        assert info["physical_distance_lower"] == pytest.approx(0.)


def test_dual_envelope_with_shared_covariance_matches_optimized_pixels():
    from research.fit_sky.envelope import bound
    from scipy.optimize import minimize_scalar
    m = scalar_scene(flux=(0, 3), count=(0, 1))
    m["exposures"][0]["data"] = [[3., -3., 3.]]
    m["linear_nuisance"] = {"design": [[1.], [1.], [1.]], "mean": [0.], "covariance": [[.1]]}
    with ctx.workprec(100):
        scene = Scene(m)
        root = scene.roots()[0]
        ceiling, _ = bound(scene, root, [])
        response = render(m["exposures"], [{"kind": "star", "parameters": {"x": 0, "y": 0, "flux": 1}}])[0].ravel()
        precision = np.linalg.inv(np.eye(3)+.1*np.ones((3, 3)))
        def energy(f):
            r = np.array([3., -3., 3.])-f*response
            return r@precision@r
        fit = minimize_scalar(energy, bounds=(0, 3), method="bounded")
        optimum = min(energy(0), energy(3), fit.fun)
        assert float(ceiling) == pytest.approx(np.exp(-optimum/2), rel=1e-10)
        assert float(ceiling) < .01


@pytest.mark.parametrize("invalid", ["negative_background", "covariance"])
def test_poisson_cannot_silently_accept_an_invalid_noise_model(invalid):
    m = scalar_scene(data=1.)
    m["noise"] = "poisson"
    if invalid == "negative_background":
        m["nuisance"] = {"default_background": [-1, 1]}
    else:
        m["covariance"] = [[1.]]
    with pytest.raises(ValueError):
        infer(m)


def test_constant_catalogue_evidence_is_integrated_not_sampled():
    from research.fit_sky.statistical import infer as statistical_infer
    m = scalar_scene()
    cover = infer(m, method="auto")
    r = statistical_infer(m, cover_report=cover, samples=10000, seed=22)
    assert r["analytically_integrated_cells"] == 1
    a, b = r["deterministic_scaled_evidence"]
    assert a <= .5*np.exp(-.5) <= b
    p = r["point_estimates_not_certificates"]["has:star"]
    assert .58 < p < .64
