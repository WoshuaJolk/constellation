"""Independent dynamics and joint pixel controls for the orbital adapter."""

from copy import deepcopy
import json

import numpy as np
import pytest
pytest.importorskip("flint")
from flint import ctx
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import multivariate_normal
from scipy.special import xlogy, gammaln

from research.detection_limit.interval_orbits import MU, C_AU_DAY
from research.fit_sky.orbital import OrbitProjection
from research.fit_sky.orbital_experiments import ambiguous_scene, independent_posterior
from research.fit_sky.reference import render
from research.fit_sky.scene import Scene
from research.fit_sky.sequential import BlockModel, SequentialFit, PARAMETER_INDEX


@pytest.mark.parametrize("speed", [.7, 1., 1.3])
def test_elliptic_parabolic_hyperbolic_against_independent_ode(speed):
    state = np.array([2., .1, .05, -.001, speed*np.sqrt(2*float(MU)/np.linalg.norm([2., .1, .05])-.001**2), 0.])
    exposures = []
    for time in [-30., 0., 40.]:
        exposures.append(dict(time=time, observer=[.9, -.2, .03], center=[1., 0., 0.],
                              east=[0., 1., 0.], north=[0., 0., 1.], pixel_scale_arcsec=.2,
                              light_time_max=1.))
    projection = OrbitProjection(exposures)
    def at(t):
        if t == 0:
            return state[:3]
        def rhs(_, y):
            return np.r_[y[3:], -float(MU)*y[:3]/np.linalg.norm(y[:3])**3]
        result = solve_ivp(rhs, [0., t], state, method="DOP853", rtol=3e-13, atol=2e-15)
        assert result.success
        return result.y[:3, -1]
    for i, e in enumerate(exposures):
        delay = brentq(lambda d: d-np.linalg.norm(at(e["time"]-d)-e["observer"])/float(C_AU_DAY),
                       0., 1., xtol=1e-14)
        d = at(e["time"]-delay)-e["observer"]
        expected = d[1:]/d[0]/(e["pixel_scale_arcsec"]*np.pi/648000)
        assert projection.coordinates(state[None], i)[0] == pytest.approx(expected, abs=2e-7)


def continuous_scene():
    m = ambiguous_scene()
    m["count_prior"] = {"probabilities": [0., 0., 1.]}
    for t in m["source_types"]:
        t["parameters"]["flux"] = [2., 7.]
        if t["kind"] == "kepler":
            t["parameters"]["rx"] = [39., 41.]
            t["parameters"]["ry"] = [-.025, .025]
            t["parameters"]["rz"] = [-.01, .01]
            t["parameters"]["vx"] = [-.002, .002]
            t["parameters"]["vy"] = [.002, .0036]
            t["parameters"]["vz"] = [-.0001, .0001]
    m["nuisance"].update(A_log_gain=[-.1, .1], B_log_psf=[-.05, .05], A_background=[-.02, .02])
    return m


@pytest.mark.parametrize("block", [1, 19, 256])
def test_joint_blended_orbits_match_independent_dense_likelihood(block):
    m = continuous_scene()
    model = BlockModel(m, pixel_block=block)
    p = model.prior(6, np.random.default_rng(19))
    p.classes[:, :2] = [[0, 1], [1, 2], [2, 2], [0, 2], [1, 0], [0, 0]]
    actual = model.evaluate(p, [0, 1, 2, 4])
    linear = m["linear_nuisance"]
    for row, catalogue in enumerate(model.catalogues(p)):
        images = render(m["exposures"], catalogue["sources"], catalogue["nuisance"])
        for col, end in enumerate([1, 2, 4], 1):
            data = np.concatenate([np.ravel(e["data"]) for e in m["exposures"][:end]])
            means = np.concatenate([x.ravel() for x in images[:end]])
            design = np.array(linear["design"])[:len(data)]
            covariance = .85**2*np.eye(len(data))+design@linear["covariance"]@design.T
            expected = multivariate_normal.logpdf(data, mean=means+design@linear["mean"], cov=covariance)
            assert actual[row, col] == pytest.approx(expected, abs=2e-9)
    q = p.copy()
    q.classes[:, :2] = q.classes[:, :2][:, ::-1]
    q.unit[:, :2] = q.unit[:, :2][:, ::-1]
    assert model.evaluate(q, [4])[:, 0] == pytest.approx(actual[:, -1], abs=1e-10)
    # A fresh evaluation must not reuse positions from before an orbit update.
    q.unit[:, :, PARAMETER_INDEX["ry"]] = .99
    assert np.max(np.abs(model.evaluate(q, [4])[:, 0]-actual[:, -1])) > 1e-4


def test_interval_model_agrees_with_adapter():
    m = ambiguous_scene()
    m.pop("linear_nuisance")
    m.pop("nuisance")
    m["source_types"] = m["source_types"][1:2]
    m["count_prior"] = {"probabilities": [0., 1.]}
    m["exposures"] = [dict(m["exposures"][0], time=1., data=[[0.]])]
    scene = Scene(m)
    with ctx.workprec(96):
        root = scene.roots()[0]
        intervals = scene.means(root, [])
        model = BlockModel(m)
        p = model.prior(2, np.random.default_rng(3))
        sources, nuisance = model.physical(p)
        actual = model._means(p, sources, nuisance, 0, 1)[0, 0]
        assert abs(float(intervals[0])-actual) < 1e-12


def test_invalid_geometry_aborts_without_excluding_orbits():
    m = ambiguous_scene()
    bad = deepcopy(m)
    bad["exposures"][0]["east"] = [1., 0., 0.]
    with pytest.raises(ValueError, match="orthonormal"):
        BlockModel(bad)
    bad = deepcopy(m)
    bad["exposures"][0]["light_time_max"] = .01
    model = BlockModel(bad)
    p = model.prior(4, np.random.default_rng(3))
    p.counts[:] = 1
    p.classes[:] = -1
    p.classes[:, 0] = 1
    with pytest.raises(ValueError, match="prior mass was not discarded"):
        model.evaluate(p, [1])
    state = np.array([[-40., 0., 0., 0., .003, 0.]])
    with pytest.raises(ValueError, match="tangent horizon"):
        OrbitProjection(m["exposures"]).coordinates(state, 0)


def test_orbital_resume_and_legacy_checkpoint_migration(tmp_path):
    m = ambiguous_scene()
    fit = SequentialFit(m, particles=32, sweeps=1, seed=90)
    fit.assimilate(1)
    path = tmp_path/"orbit.npz"
    fit.save(path)
    resumed = SequentialFit.load(path)
    fit.assimilate(2)
    resumed.assimilate(2)
    assert np.array_equal(fit.population.unit, resumed.population.unit)
    assert fit.log_evidence == resumed.log_evidence
    assert fit.report()["accuracy_target_met"] is False
    # Simulate the actual old ten-coordinate format, preserving its RNG state.
    m["source_types"] = m["source_types"][:1]
    old = SequentialFit(m, particles=4, sweeps=1)
    old.save(path)
    with np.load(path) as saved:
        payload = {k: saved[k].copy() for k in saved.files}
    metadata = json.loads(str(payload["metadata"]))
    metadata["schema"] = "fit-sky-sequential-v1"
    payload["metadata"] = np.array(json.dumps(metadata))
    payload["unit"] = payload["unit"][..., :10]
    np.savez(path, **payload)
    migrated = SequentialFit.load(path)
    assert np.array_equal(migrated.population.unit[..., :10], old.population.unit[..., :10])
    assert migrated.model.evaluate(migrated.population, [4]) == pytest.approx(
        old.model.evaluate(old.population, [4]), abs=0)
    migrated.assimilate(1)


def test_continuous_orbit_moves_flux_and_split_merge_smoke():
    fit = SequentialFit(continuous_scene(), particles=32, sweeps=2, seed=31,
                        gibbs_flux=True, split_merge=True,
                        queries=["has:orbit_a", "any:orbit_a:vy:0.002:0.0028"])
    fit.assimilate(1)
    fit.assimilate(2)
    fit.population.validate(fit.model)
    assert np.isfinite(fit.log_evidence)
    assert all(v["certified_interval"] == [0., 1.] for v in fit.summary()["queries"].values())


def test_ambiguous_reference_quadrature_convergence():
    a, b = [independent_posterior(ambiguous_scene(), order) for order in [4, 6]]
    assert a["log_evidence"] == pytest.approx(b["log_evidence"], abs=2e-6)
    assert a["queries"] == pytest.approx(b["queries"], abs=2e-6)
    assert .05 < b["queries"]["has:orbit_a"] < .95
    assert .05 < b["queries"]["has:orbit_b"] < .95


def test_masked_poisson_orbit_pixels_and_exposure_order():
    m = continuous_scene()
    m.pop("linear_nuisance")
    m["nuisance"].pop("A_background")
    m["noise"] = "poisson"
    for e in m["exposures"]:
        e["background"] = 1.
        e["data"] = np.maximum(0, np.round(e["data"])).tolist()
        mask = np.zeros((5, 7), bool)
        mask[2, 3] = True
        e["mask"] = mask.tolist()
    model = BlockModel(m, pixel_block=9)
    p = model.prior(4, np.random.default_rng(91))
    p.classes[:, :2] = [0, 1]
    ll = model.evaluate(p, [4])[:, 0]
    for i, catalogue in enumerate(model.catalogues(p)):
        images = render(m["exposures"], catalogue["sources"], catalogue["nuisance"])
        expected = 0.
        for e, image in zip(m["exposures"], images):
            valid = ~np.array(e["mask"])
            y = np.array(e["data"])[valid]
            expected += np.sum(xlogy(y, image[valid])-image[valid]-gammaln(y+1))
        assert ll[i] == pytest.approx(expected, abs=2e-9)
    m["exposures"].reverse()
    assert BlockModel(m, pixel_block=13).evaluate(p, [4])[:, 0] == pytest.approx(ll, abs=1e-10)


def test_propagation_reused_across_pixels_but_not_population_updates(monkeypatch):
    model = BlockModel(continuous_scene(), pixel_block=1)
    p = model.prior(4, np.random.default_rng(18))
    p.classes[:, :2] = [1, 2]
    calls = []
    original = model.orbit_projection.coordinates
    def counted(states, exposure):
        calls.append(exposure)
        return original(states, exposure)
    monkeypatch.setattr(model.orbit_projection, "coordinates", counted)
    model.evaluate(p, [4])
    assert len(calls) == 8  # Two slots * four exposures, not 140 pixels.
    p.unit[:, :, PARAMETER_INDEX["vy"]] = .2
    model.evaluate(p, [4])
    assert len(calls) == 16
