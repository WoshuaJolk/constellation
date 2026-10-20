"""Independent controls for continuous-template likelihood ceilings."""

import math

import numpy as np
import pytest
pytest.importorskip("flint")
mp = pytest.importorskip("mpmath")
from flint import arb, ctx
from scipy.special import ndtr
from scipy.integrate import quad

from research.fit_sky.scene import Scene
from research.fit_sky.statistical import BatchModel
from research.fit_sky.noise_floor import (
    _profiles, build, build_cover, ceiling, support, tail_ceiling, verify, verify_cover,
)


def scene():
    return {
        "source_types": [{"name": "star", "kind": "star",
                          "parameters": {"x": [-1., 1.], "y": [0., 0.], "flux": [0., 5.]}}],
        "count_prior": {"probabilities": [.4, .6]},
        "exposures": [{"id": "a", "time": 0., "data": [[-2., 3., -1.]],
                       "psf_sigma": .7, "noise_sigma": .5}],
    }


def test_gaussian_profile_curvatures_match_independent_differentiation():
    with ctx.workprec(100), mp.workdps(50):
        x, sigma, c = .21, .83, 1.
        value, dx2, ds2 = _profiles([c], arb(x), arb(sigma))[0]
        def f(x, s):
            return (mp.erf((c+.5-x)/(s*mp.sqrt(2)))-mp.erf((c-.5-x)/(s*mp.sqrt(2))))/2
        expected = [f(mp.mpf(x), mp.mpf(sigma)),
                    mp.diff(lambda q: f(q, mp.mpf(sigma)), mp.mpf(x), 2),
                    mp.diff(lambda q: f(mp.mpf(x), q), mp.mpf(sigma), 2)]
        for actual, ref in zip([value, dx2, ds2], expected):
            assert actual.contains(arb(str(ref)))


def test_continuous_support_includes_width_variation_and_signed_pixels():
    e = {"data": [[0., 0., 0.], [0., 0., 0.]]}
    active = list(np.ndindex(2, 3))
    z = np.array([1., -2., 3., -4., 2., -1.])
    domain = {"ranges": [[-.6, .8], [-.3, .4], [.6, 1.], [.7, 1.1]]}
    with ctx.workprec(100):
        r = support(e, active, list(map(arb, z)), domain, max_leaves=12)
    rng = np.random.default_rng(732)
    b = np.array(domain["ranges"])
    x, y, sx, sy = (b[:, 0]+rng.random((5000, 4))*(b[:, 1]-b[:, 0])).T
    xx = np.array([i-1 for j, i in active])[:, None]
    yy = np.array([j-.5 for j, i in active])[:, None]
    p = (ndtr((xx+.5-x)/sx)-ndtr((xx-.5-x)/sx))*(
        ndtr((yy+.5-y)/sy)-ndtr((yy-.5-y)/sy))
    assert np.max(z@p) <= r["upper"]
    assert len(r["leaves"]) <= 12


def test_noise_floor_covers_all_catalogues_and_replays():
    with ctx.workprec(100):
        s = Scene(scene())
        certificate = build(s, max_leaves=32, grid=5, iterations=100)
        assert verify(s, certificate)["valid"]
        model, rng = BatchModel(s), np.random.default_rng(610)
        for root in s.roots():
            b = np.array(root["ranges"]).reshape(-1, 2)
            points = b[:, 0]+rng.random((4000, len(b)))*(b[:, 1]-b[:, 0])
            actual = model.likelihood(root, points)
            bound = float(ceiling(certificate, root))
            assert actual.max() <= bound*(1+1e-12)
            assert bound < .001  # impossible to fit alternating signs with positive PSFs


@pytest.mark.parametrize("damage", ["partition", "leaf", "baseline", "class", "dual", "manifest"])
def test_noise_floor_rejects_damaged_certificates(damage):
    with ctx.workprec(100):
        s = Scene(scene())
        c = build(s, max_leaves=8, grid=3, iterations=20)
        if damage == "partition":
            c["supports"][0][0]["leaves"].pop()
        elif damage == "leaf":
            c["supports"][0][0]["leaves"][0]["upper"] = -1e100
        elif damage == "baseline":
            c["baseline_energy_lower"] += 10
        elif damage == "class":
            c["class_support_upper"][0] -= 1
        elif damage == "dual":
            c["dual"][0] = float("nan")
        else:
            c["manifest_sha256"] = "wrong"
        with pytest.raises(ValueError):
            verify(s, c)


def test_fixed_template_ceiling_matches_exact_constrained_flux_optimum():
    m = scene()
    m["source_types"][0]["parameters"]["x"] = [0., 0.]
    with ctx.workprec(100):
        s = Scene(m)
        c = build(s, grid=2, iterations=20)
        root = s.roots()[1]
        pixels = np.array([ndtr((i+.5)/.7)-ndtr((i-.5)/.7) for i in [-1, 0, 1]])
        pixels *= ndtr(.5/.7)-ndtr(-.5/.7)
        y = np.array([-2., 3., -1.])
        flux = np.clip(y@pixels/(pixels@pixels), 0, 5)
        exact = np.exp(-.5*np.sum(((y-flux*pixels)/.5)**2))
        assert float(ceiling(c, root)) == pytest.approx(exact, rel=1e-8)


def test_signed_support_retains_positive_flux_lower_bound():
    m = scene()
    m["exposures"][0]["data"] = [[0.]]
    m["source_types"][0]["parameters"].update({"x": [0., 0.], "flux": [5., 10.]})
    with ctx.workprec(100):
        s = Scene(m)
        c = build(s, grid=2, iterations=20)
        assert verify(s, c)["valid"]
        assert c["class_support_upper"][0] < 0
        response = (ndtr(.5/.7)-ndtr(-.5/.7))**2
        exact = np.exp(-.5*(5*response/.5)**2)
        bound = float(ceiling(c, s.roots()[1]))
        assert exact <= bound <= exact*(1+1e-6)


def test_poisson_tail_ceiling_includes_every_omitted_count():
    m = scene()
    m["source_types"][0]["parameters"]["flux"] = [0., 0.]
    m["count_prior"] = {"poisson_mean": 2., "max_count": 2}
    with ctx.workprec(100):
        s = Scene(m)
        c = build(s, grid=2, iterations=10)
        tail = tail_ceiling(s, c)
        expected = math.exp(-28)*(1-math.exp(-2)*5)
        assert float(tail) == pytest.approx(expected, rel=1e-12)
        assert tail < s.tail


def test_shared_background_and_extended_source_bounds_cover_independent_draws():
    m = scene()
    m["source_types"][0]["kind"] = "galaxy"
    m["source_types"][0]["parameters"].update(
        {"sigma_x": [.1, .5], "sigma_y": [.2, .4], "vx": [-.2, .2],
         "parallax": [0., .1], "color": [-.1, .1], "variability": [-.1, .1]})
    m["exposures"][0].update({"time": 1., "parallax_x": .4, "band_coordinate": -.5})
    m["nuisance"] = {"default_background": [-.1, .1], "default_dx": [-.1, .1],
                     "default_log_psf": [-.1, .1], "default_log_gain": [-.1, .1]}
    m["linear_nuisance"] = {"design": [[1.], [1.], [1.]],
                            "mean": [0.], "covariance": [[.1]]}
    with ctx.workprec(100):
        s = Scene(m)
        c = build(s, max_leaves=8, grid=3, iterations=30)
        assert verify(s, c)["valid"]
        root = s.roots()[1]
        b = np.array(root["ranges"])
        rng = np.random.default_rng(991)
        points = b[:, 0]+rng.random((5000, len(b)))*(b[:, 1]-b[:, 0])
        assert BatchModel(s).likelihood(root, points).max() <= float(ceiling(c, root))


def test_conditional_noise_cover_is_bound_to_full_partition():
    from research.fit_sky.integrate import infer, _box
    with ctx.workprec(100):
        m = scene()
        s = Scene(m)
        cover = infer(m, max_leaves=8, max_seconds=5, method="range")
        proof = build_cover(s, cover, max_leaves=4, grid=3, iterations=20)
        ceilings = verify_cover(s, cover, proof)
        rng = np.random.default_rng(32)
        roots = s.roots()
        for leaf, bound in zip(cover["certificate"], ceilings):
            root = roots[leaf["root"]]
            b = np.array([[float(a), float(b)] for a, b in _box(root, leaf["path"])]).reshape(-1, 2)
            points = b[:, 0]+rng.random((100, len(b)))*(b[:, 1]-b[:, 0])
            assert BatchModel(s).likelihood(root, points).max() <= float(bound)*(1+1e-12)
        proof["proofs"].pop()
        with pytest.raises(ValueError):
            verify_cover(s, cover, proof)


def test_vectorized_flux_integral_retains_other_sources_and_shared_covariance():
    m = scene()
    m["count_prior"] = {"probabilities": [0, 0, 1]}
    m["linear_nuisance"] = {"design": [[1.], [1.], [1.]],
                            "mean": [.2], "covariance": [[.3]]}
    with ctx.workprec(100):
        s = Scene(m)
        root = s.roots()[0]
        model = BatchModel(s)
        point = np.array([[2., -.3, 1.5, .4]])
        axis = root["names"].index("s0.flux")
        for restriction in [None, (.5, 2.), (-2., -.1)]:
            got = model.marginal_flux(root, point, axis, [0., 5.], restriction)[0]
            lo, hi = (0., 5.) if restriction is None else (max(0., restriction[0]), min(5., restriction[1]))
            def evaluate(f):
                p = point.copy()
                p[0, axis] = f
                return model.likelihood(root, p)[0]/5
            expected = quad(evaluate, lo, hi, epsabs=1e-20)[0] if hi > lo else 0.
            assert got == pytest.approx(expected, rel=1e-10, abs=1e-30)


def test_flux_integral_accepts_different_cell_bounds_in_one_batch():
    with ctx.workprec(100):
        s = Scene(scene())
        root, model = s.roots()[1], BatchModel(s)
        points = np.array([[.5, -.3], [3., .4]])
        axis = root["names"].index("s0.flux")
        bounds = np.array([[0., 1.], [2., 5.]])
        actual = model.marginal_flux(root, points, axis, bounds.T, (0., 2.5))
        for i, (lo, hi) in enumerate(bounds):
            def f(x):
                p = points[i:i+1].copy()
                p[:, axis] = x
                return model.likelihood(root, p)[0]/(hi-lo)
            expected = quad(f, lo, min(hi, 2.5), epsabs=1e-20)[0]
            assert actual[i] == pytest.approx(expected, rel=1e-10, abs=1e-30)


def test_marginalized_flux_union_query_matches_independent_double_integral():
    from research.fit_sky.statistical import infer
    m = scene()
    m["count_prior"] = {"probabilities": [0, 0, 1]}
    m["source_types"][0]["parameters"]["x"] = [0., 0.]
    m["exposures"][0]["data"] = [[1.]]
    response = (ndtr(.5/.7)-ndtr(-.5/.7))**2
    def integral(a, b, c, d):
        return quad(lambda f: quad(lambda g: np.exp(-.5*((1-(f+g)*response)/.5)**2),
                                   c, d, epsabs=1e-11)[0], a, b, epsabs=1e-11)[0]/25
    evidence = integral(0, 5, 0, 5)
    numerator = evidence-sum(integral(a, b, c, d) for a, b in [(0, 1), (2, 5)]
                             for c, d in [(0, 1), (2, 5)])
    r = infer(m, samples=100000, queries=["any:star:flux:1:2"], seed=499, marginalize_flux=True)
    lo, hi = r["queries"]["any:star:flux:1:2"]
    assert lo < numerator/evidence < hi
    assert r["one_flux_analytically_marginalized"]


def test_reference_quadrature_matches_independent_scalar_integration():
    from research.fit_sky.reference_quadrature import integrate
    m = scene()
    m["source_types"][0]["parameters"]["x"] = [0., 0.]
    m["exposures"][0]["data"] = [[1.]]
    response = (ndtr(.5/.7)-ndtr(-.5/.7))**2
    source = quad(lambda f: np.exp(-.5*((1-f*response)/.5)**2), 0, 5, epsabs=1e-13)[0]/5
    empty = np.exp(-2)
    expected = .6*source/(.4*empty+.6*source)
    result = integrate(m, order=12)
    assert result["queries"]["has:star"] == pytest.approx(expected, rel=1e-12)


def test_fast_catalogue_draws_match_independent_count_and_flux_probabilities():
    from research.fit_sky.integrate import infer
    from research.fit_sky.sampling import sample
    m = scene()
    m["source_types"][0]["parameters"]["x"] = [0., 0.]
    m["exposures"][0]["data"] = [[1.]]
    response = (ndtr(.5/.7)-ndtr(-.5/.7))**2
    integrand = lambda f: np.exp(-.5*((1-f*response)/.5)**2)/5
    source = quad(integrand, 0, 5)[0]
    evidence = .4*np.exp(-2)+.6*source
    expected_count = .6*source/evidence
    expected_event = .6*quad(integrand, 1, 2)[0]/evidence
    with ctx.workprec(100):
        s = Scene(m)
        report = infer(m, method="hybrid")
        floor = build(s, grid=2, iterations=20)
        conditional = build_cover(s, report, grid=2, iterations=20)
        draws = sample(report, 2000, seed=981, max_proposals=30000, fast=True,
                       noise_floor_certificate=floor, conditional_noise_certificate=conditional)
    assert draws["complete"]
    counts = [len(d["sources"]) for d in draws["catalogues"]]
    events = [any(1 <= src["parameters"]["flux"] <= 2 for src in d["sources"])
              for d in draws["catalogues"]]
    assert np.mean(counts) == pytest.approx(expected_count, abs=.04)
    assert np.mean(events) == pytest.approx(expected_event, abs=.04)
    assert draws["evaluated_proposals"] >= draws["proposals"]


def test_adaptive_noise_cover_replays_without_rerunning_optimizer(monkeypatch):
    from research.fit_sky.integrate import infer, verify as verify_integral
    from research.fit_sky import noise_floor
    with ctx.workprec(100):
        report = infer(scene(), method="noise-floor", max_leaves=12, max_seconds=5)
        assert report["work"]["leaves"] == 12
        def forbidden(*args, **kwargs):
            raise AssertionError("A certificate replay must not optimize again")
        monkeypatch.setattr(noise_floor, "propose", forbidden)
        assert verify_integral(report)["valid"]
        leaf = next(r for r in report["certificate"] if isinstance(r["mode"], dict))
        leaf["mode"]["proof"]["baseline_energy_lower"] += 10
        with pytest.raises(ValueError):
            verify_integral(report)


def test_statistical_integration_uses_embedded_pointwise_proofs():
    from research.fit_sky.integrate import infer
    from research.fit_sky.statistical import infer as statistical
    m = scene()
    with ctx.workprec(100):
        cover = infer(m, method="noise-floor", max_leaves=16, max_seconds=5)
        result = statistical(m, cover_report=cover, samples=10000, seed=51, marginalize_flux=True)
    assert result["proposal_envelope"] < 1e-3
    assert result["one_flux_analytically_marginalized"]
