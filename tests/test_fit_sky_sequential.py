"""Independent correctness controls for sequential catalogue inference."""

from copy import deepcopy
import json

import numpy as np
import pytest
pytest.importorskip("flint")
from scipy.special import logsumexp, ndtr
from scipy.stats import multivariate_normal, poisson, truncnorm
from scipy.integrate import quad

from research.fit_sky.experiments import scalar, synthetic
from research.fit_sky.reference import render
from research.fit_sky.sequential import (
    BlockModel, SequentialFit, Population, PARAMETERS, propose_counts,
    propose_parameters, reflect, _accept,
    PARAMETER_INDEX,
)


def mixed():
    manifest, _ = synthetic("broad")
    manifest["exposures"][1]["mask"] = [[False, True, False]]*3
    n = sum(np.count_nonzero(~np.broadcast_to(e.get("mask", False), np.shape(e["data"])))
            for e in manifest["exposures"])
    manifest["linear_nuisance"] = {
        "design": np.column_stack([np.ones(n), np.linspace(-1., 1., n)]).tolist(),
        "mean": [.1, -.05], "covariance": [[.2, .03], [.03, .1]],
    }
    return manifest


@pytest.mark.parametrize("block", [1, 7, 128])
def test_prefix_block_likelihood_matches_dense_shared_covariance(block):
    manifest = mixed()
    model = BlockModel(manifest, pixel_block=block)
    population = model.prior(8, np.random.default_rng(997))
    actual = model.evaluate(population, [0, 1, 2, 3])
    linear = manifest["linear_nuisance"]
    for row, catalogue in enumerate(model.catalogues(population)):
        images = render(manifest["exposures"], catalogue["sources"], catalogue["nuisance"])
        data, means, sigmas = [], [], []
        assert actual[row, 0] == 0
        for prefix, (exposure, image) in enumerate(zip(manifest["exposures"], images), 1):
            mask = np.broadcast_to(exposure.get("mask", False), image.shape)
            data.extend(np.asarray(exposure["data"])[~mask])
            means.extend(image[~mask])
            sigmas.extend(np.broadcast_to(exposure["noise_sigma"], image.shape)[~mask])
            design = np.array(linear["design"])[:len(data)]
            covariance = np.diag(np.array(sigmas)**2)+design@np.array(linear["covariance"])@design.T
            expected = multivariate_normal.logpdf(data, mean=np.array(means)+design@linear["mean"],
                                                  cov=covariance)
            assert actual[row, prefix] == pytest.approx(expected, abs=1e-10)


def test_joint_shared_background_is_not_independently_marginalized():
    m = scalar()
    m["source_types"][0]["parameters"]["flux"] = [0., 0.]
    m["exposures"] = [dict(m["exposures"][0], id=str(i), data=[[2.]]) for i in range(2)]
    m["linear_nuisance"] = {"design": [[1.], [1.]], "mean": [0.], "covariance": [[4.]]}
    model = BlockModel(m)
    p = model.prior(4, np.random.default_rng(34))
    actual = model.evaluate(p, [1, 2])[0]
    expected = multivariate_normal.logpdf([2., 2.], cov=[[5., 4.], [4., 5.]])
    assert actual[1] == pytest.approx(expected)
    assert abs(actual[1]-2*actual[0]) > .2


def test_source_permutation_and_exposure_order_invariance():
    model = BlockModel(mixed())
    p = model.prior(20, np.random.default_rng(44))
    q = p.copy()
    for row, k in enumerate(q.counts):
        q.classes[row, :k] = p.classes[row, :k][::-1]
        q.unit[row, :k] = p.unit[row, :k][::-1]
    assert model.evaluate(p, [3]) == pytest.approx(model.evaluate(q, [3]), abs=1e-10)
    m = deepcopy(model.manifest)
    m["exposures"].reverse()
    linear = m["linear_nuisance"]
    design = np.array(linear["design"])
    linear["design"] = np.concatenate([design[model.ends[i]:model.ends[i+1]]
                                       for i in [2, 1, 0]]).tolist()
    reordered = BlockModel(m)
    assert model.evaluate(p, [3]) == pytest.approx(reordered.evaluate(p, [3]), abs=1e-10)


@pytest.mark.parametrize("prior", [{"poisson_mean": 4., "max_count": 1},
                                  {"probabilities": [.2, 0., .3, 0., .5]}])
def test_birth_death_and_class_moves_preserve_complete_prior(prior):
    m, _ = synthetic("broad")
    m["count_prior"] = prior
    m["source_types"][0]["weight"] = 4.
    model, rng = BlockModel(m), np.random.default_rng(95)
    population = model.prior(6000, rng)
    likelihood = np.zeros((len(population.counts), 2))
    for _ in range(18):
        proposal, ratio, _ = propose_counts(model, population, rng)
        _accept(population, proposal, likelihood, likelihood.copy(), ratio, 0., rng)
        proposal, ratio, _ = propose_parameters(model, population, rng, .2)
        _accept(population, proposal, likelihood, likelihood.copy(), ratio, 0., rng)
    population.validate(model)
    if "poisson_mean" in prior:
        assert population.counts.mean() == pytest.approx(4., abs=.12)
        assert population.counts.max() > prior["max_count"]
        assert np.mean(population.counts > 7) == pytest.approx(poisson.sf(7, 4.), abs=.012)
    else:
        for k, expected in enumerate(prior["probabilities"]):
            assert np.mean(population.counts == k) == pytest.approx(expected, abs=.025)
    active = population.classes >= 0
    assert np.mean(population.classes[active] == 0) == pytest.approx(4/6, abs=.02)
    assert population.unit[active].mean() == pytest.approx(.5, abs=.01)
    assert population.nuisance.mean() == pytest.approx(.5, abs=.02)


def test_birth_death_prior_ratio_has_only_one_poisson_factor():
    m = scalar()
    m["count_prior"] = {"poisson_mean": 2., "max_count": 0}
    model, rng = BlockModel(m), np.random.default_rng(129)
    p = model.prior(400, rng)
    q, ratios, _ = propose_counts(model, p, rng)
    # Non-refresh local proposals must use lambda/(K+1), not a second
    # factorial from treating exchangeable labels as another prior penalty.
    expected = model.log_count_prior(q.counts)-model.log_count_prior(p.counts)
    local = ratios != 0
    assert ratios[local] == pytest.approx(expected[local])
    assert np.any(q.counts > 0)


def test_scalar_posterior_and_evidence_against_analytic_integral():
    m = scalar()
    response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
    one = quad(lambda flux: np.exp(-.5*(1-flux*response)**2)/2, 1, 3,
               epsabs=1e-13)[0]
    zero = np.exp(-.5)
    exact_z = .5*(one+zero)/np.sqrt(2*np.pi)
    fit = SequentialFit(m, particles=4000, sweeps=8, seed=177,
                        queries=["count=0", "has:star", "any:star:flux:1:2"])
    summary = fit.assimilate()
    assert np.exp(summary["log_evidence_estimate"]) == pytest.approx(exact_z, rel=.025)
    assert summary["queries"]["has:star"]["estimate"] == pytest.approx(one/(one+zero), abs=.025)
    restricted = quad(lambda flux: np.exp(-.5*(1-flux*response)**2)/2, 1, 2)[0]
    assert summary["queries"]["any:star:flux:1:2"]["estimate"] == pytest.approx(
        restricted/(one+zero), abs=.025)
    assert fit.report()["status"] == "approximate_uncertified"
    assert not fit.report()["accuracy_target_met"]


def test_shared_calibration_posterior_matches_independent_quadrature():
    m = scalar()
    m["source_types"][0]["parameters"]["flux"] = [3., 3.]
    m["nuisance"] = {"default_background": [-1., 1.]}
    m["exposures"] = [dict(m["exposures"][0], id=f"e{i}", data=[[y]])
                      for i, y in enumerate([.2, .9, 1.3])]
    response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
    def likelihood(background, source):
        residual = np.array([.2, .9, 1.3])-background-source*3*response
        return np.exp(-.5*sum(residual**2))/(2*np.pi)**1.5
    evidence = [quad(lambda b: likelihood(b, k)/2, -1, 1)[0] for k in [0, 1]]
    numerator = sum(quad(lambda b: b*likelihood(b, k)/2, -1, 1)[0] for k in [0, 1])
    fit = SequentialFit(m, particles=4000, sweeps=10, seed=205)
    for i in [1, 2, 3]:
        summary = fit.assimilate(i)
    assert np.exp(summary["log_evidence_estimate"]) == pytest.approx(sum(evidence)/2, rel=.035)
    assert summary["queries"]["has:star"]["estimate"] == pytest.approx(
        evidence[1]/sum(evidence), abs=.03)
    assert summary["nuisance_mean_estimate"]["default_background"] == pytest.approx(
        numerator/sum(evidence), abs=.03)


def test_poisson_log_likelihood_matches_independent_count_probability():
    m = scalar()
    m["noise"] = "poisson"
    m["exposures"][0]["data"] = [[3.]]
    m["exposures"][0]["background"] = .2
    model = BlockModel(m)
    p = model.prior(10, np.random.default_rng(13))
    for row, catalogue in enumerate(model.catalogues(p)):
        intensity = render(m["exposures"], catalogue["sources"], catalogue["nuisance"])[0][0, 0]
        assert model.evaluate(p, [1])[row, 0] == pytest.approx(poisson.logpmf(3, intensity))


def test_zero_poisson_likelihood_support_is_counted_in_evidence():
    m = scalar()
    m["noise"] = "poisson"
    m["source_types"][0]["parameters"]["flux"] = [3., 3.]
    m["exposures"][0]["data"] = [[2.]]
    m["exposures"].append(dict(m["exposures"][0], id="second", data=[[1.]]))
    response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
    fit = SequentialFit(m, particles=4000, sweeps=2, seed=243)
    fit.assimilate(1)
    summary = fit.assimilate(2)
    expected = .5*poisson.pmf(2, 3*response)*poisson.pmf(1, 3*response)
    assert np.exp(summary["log_evidence_estimate"]) == pytest.approx(expected, rel=.04)
    assert summary["queries"]["has:star"]["estimate"] == pytest.approx(1.)
    assert fit.stages[0]["kind"] == "zero_likelihood_support_restriction"


def test_thousand_exposure_log_likelihood_does_not_underflow():
    m = scalar()
    m["exposures"] = [dict(m["exposures"][0], id=f"e{i}", data=[[float(i % 3)]])
                      for i in range(1000)]
    model = BlockModel(m, pixel_block=17)
    population = model.prior(20, np.random.default_rng(256))
    actual = model.evaluate(population, [1000])[:, 0]
    for row, cat in enumerate(model.catalogues(population)):
        flux = sum(s["parameters"]["flux"] for s in cat["sources"])
        response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
        expected = -.5*np.sum((np.arange(1000) % 3-flux*response)**2)-500*np.log(2*np.pi)
        assert actual[row] == pytest.approx(expected, abs=1e-9)
    assert np.all(np.isfinite(actual))
    assert np.all(np.exp(actual) == 0)  # Ordinary likelihood products would fail.


def test_tempered_conditional_flux_matches_independent_dense_quadratic():
    m = scalar()
    m["count_prior"] = {"probabilities": [0., 0., 1.]}
    m["exposures"] = [dict(m["exposures"][0], id=f"e{i}", data=[[y]])
                      for i, y in enumerate([.2, .9, 1.3])]
    m["linear_nuisance"] = {"design": [[1.]]*3, "mean": [.1], "covariance": [[.5]]}
    model, rng = BlockModel(m), np.random.default_rng(291)
    population = model.prior(6000, rng)
    population.unit[..., PARAMETER_INDEX["flux"]] = .5  # both initially flux=2
    proposal = model.sample_conditional_flux(population, [1, 3], .37, rng)
    changed = np.abs(proposal.unit[..., PARAMETER_INDEX["flux"]]-.5) > 1e-12
    assert np.all(changed.sum(axis=1) == 1)
    actual = model.physical(proposal)[0][..., PARAMETER_INDEX["flux"]][changed]
    response = (ndtr(.5/.6)-ndtr(-.5/.6))**2
    h, g = 0., 0.
    for count, beta in [(1, .63), (3, .37)]:
        covariance = np.eye(count)+.5*np.ones((count, count))
        a = np.full(count, response)
        residual = np.array([.2, .9, 1.3])[:count]-.1-2*response
        h += beta*(a@np.linalg.solve(covariance, a))
        g += beta*(a@np.linalg.solve(covariance, residual))
    center, scale = g/h, 1/np.sqrt(h)
    expected = truncnorm((1-center)/scale, (3-center)/scale, loc=center, scale=scale)
    assert actual.mean() == pytest.approx(expected.mean(), abs=.025)
    assert actual.var() == pytest.approx(expected.var(), abs=.025)


def test_checkpoint_resume_reproduces_uninterrupted_assimilation(tmp_path):
    m, _ = synthetic("moving")
    fit = SequentialFit(m, particles=80, sweeps=2, seed=199)
    fit.assimilate(1)
    path = tmp_path/"state.npz"
    fit.save(path)
    resumed = SequentialFit.load(path)
    a, b = fit.assimilate(3), resumed.assimilate(3)
    assert a["log_evidence_estimate"] == b["log_evidence_estimate"]
    assert np.array_equal(fit.population.classes, resumed.population.classes)
    assert np.array_equal(fit.population.unit, resumed.population.unit)
    assert fit.stages == resumed.stages
    with pytest.raises(ValueError, match="append"):
        resumed.assimilate(3)


def test_checkpoint_can_append_new_data_but_cannot_change_old_data(tmp_path):
    m = mixed()
    prefix = deepcopy(m)
    m["exposures"].append(dict(deepcopy(m["exposures"][-1]), id="appended", time=2.))
    m["linear_nuisance"]["design"].extend(deepcopy(m["linear_nuisance"]["design"][-9:]))
    fit = SequentialFit(prefix, particles=40, sweeps=1, seed=214)
    fit.assimilate(3)
    path = tmp_path/"state.npz"
    fit.save(path)
    resumed = SequentialFit.load(path, m)
    resumed.assimilate(4)
    assert resumed.assimilated == 4
    bad = deepcopy(m)
    bad["exposures"][0]["data"][0][0] += .1
    with pytest.raises(ValueError, match="observations changed"):
        SequentialFit.load(path, bad)
    bad = deepcopy(m)
    bad["count_prior"]["poisson_mean"] += .1
    with pytest.raises(ValueError, match="prior or instrument"):
        SequentialFit.load(path, bad)
    bad = deepcopy(m)
    bad["linear_nuisance"]["design"][0][0] += .1
    with pytest.raises(ValueError, match="design changed"):
        SequentialFit.load(path, bad)


def test_budget_exhaustion_cannot_be_saved_as_completed(tmp_path):
    m, _ = synthetic("moving")
    fit = SequentialFit(m, particles=40, sweeps=1, max_stages=1)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        fit.assimilate()
    with pytest.raises(RuntimeError, match="interrupted"):
        fit.save(tmp_path/"invalid.npz")
    with pytest.raises(RuntimeError, match="interrupted"):
        fit.report()


def test_reject_invalid_models_checkpoints_and_queries(tmp_path):
    m = scalar()
    m["exposures"].append(deepcopy(m["exposures"][0]))
    with pytest.raises(ValueError, match="once"):
        BlockModel(m)
    m = scalar()
    m["covariance"] = [[1.]]
    with pytest.raises(ValueError, match="diagonal"):
        BlockModel(m)
    with pytest.raises(ValueError, match="absent"):
        SequentialFit(scalar(), queries=["any:star:vy:-1:1"])
    fit = SequentialFit(scalar(), particles=10)
    p = fit.population.copy()
    p.unit[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="values"):
        p.validate(fit.model)
    assert reflect(np.array([-.2, .2, 1.2, 2.2])) == pytest.approx([.2, .2, .8, .2])
