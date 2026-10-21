"""Independent orbital and exhaustive-association controls for solve_itf."""
from copy import deepcopy
from itertools import product
import json
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from magictools import agent_cli
from magictools.orbits import GM, C_AU_DAY
from magictools.solve_itf import validate, log_likelihood_ratios, partition_search, run
from magictools.solve_itf_example import example


def exhaustive(ratios, weights, activity):
    """Independent enumeration via labelled assignments, then deduplication.

    No use of the production partition generator, scores or mass bound.
    """
    n = len(ratios)
    partitions = set()
    for labels in product(range(n), repeat=n):
        groups = tuple(sorted(tuple(i for i, label in enumerate(labels) if label == j)
                              for j in set(labels)))
        partitions.add(groups)
    weighted = {}
    weights = np.asarray(weights)/sum(weights)
    for partition in partitions:
        score = 1.
        for group in partition:
            if len(group) > 1:
                score *= activity*sum(weights[r]*np.prod(ratios[list(group), r])
                                      for r in range(len(weights)))
        weighted[partition] = score
    total = sum(weighted.values())
    return {p: w/total for p, w in weighted.items()}, total


RATIOS = np.array([[8., .2], [5., 4.], [.1, 9.], [2., 1.]])
WEIGHTS = np.array([.35, .65])


@pytest.mark.parametrize("budget", [0, 1, 4, 12, 30, 100000])
def test_frontier_encloses_exhaustive_probabilities(budget):
    truth, total = exhaustive(RATIOS, WEIGHTS, .7)
    result = partition_search(np.log(RATIOS), np.log(WEIGHTS), .7,
                              max_nodes=budget, top_k=100, queries=[(0, 1), (1, 2)])
    assert result["total_assignments"] == 15 == len(truth)
    assert result["completed_assignments"]+result["remaining_assignments"] == len(truth)
    for candidate in result["ranked_candidates"]:
        block = tuple(candidate["indices"])
        exact = sum(p for partition, p in truth.items() if block in partition)
        lower, upper = candidate["probability_interval"]
        assert lower-2e-13 <= exact <= upper+2e-13
    for row in result["queries"]:
        query = set(row["indices"])
        exact = sum(p for partition, p in truth.items()
                    if any(query <= set(block) for block in partition))
        lower, upper = row["probability_interval"]
        assert lower-2e-13 <= exact <= upper+2e-13
    lower, upper = result["log_partition_function_interval"]
    assert lower is None or lower <= math.log(total)+2e-13
    assert upper >= math.log(total)-2e-13
    if budget == 0:
        assert result["unresolved_mass_upper"] == 1
        assert result["queries"][0]["probability_interval"] == [0., 1.]
    if budget == 100000:
        assert result["status"] == "finite_prior_complete"
        assert result["unresolved_mass_upper"] == 0.
        assert lower == pytest.approx(math.log(total))
        assert upper == pytest.approx(lower)
        assert result["returned_assignment_order_resolved"]
        for row in result["ranked_assignments"]:
            expected = truth[tuple(tuple(b) for b in row["blocks"])]
            assert row["probability_interval"] == pytest.approx([expected, expected])


def test_ambiguity_is_global_and_overlapping_links_are_exclusive():
    truth, _ = exhaustive(RATIOS, WEIGHTS, .7)
    result = partition_search(np.log(RATIOS), np.log(WEIGHTS), .7, top_k=100)
    events = {tuple(r["indices"]): r["probability_interval"][0] for r in result["ranked_candidates"]}
    assert events[(0, 1)] > 0 and events[(1, 2)] > 0
    assert events[(0, 1)]+events[(1, 2)] <= 1
    for row in result["ranked_assignments"]:
        flattened = [i for block in row["blocks"] for i in block]
        assert sorted(flattened) == list(range(4))
    # A local orbit evidence normalized over just two candidates is not either
    # event's marginal in the complete assignment model.
    local = np.array([sum(WEIGHTS*RATIOS[0]*RATIOS[1]), sum(WEIGHTS*RATIOS[1]*RATIOS[2])])
    assert abs(events[(0, 1)]-local[0]/sum(local)) > .01


def test_prior_weight_splitting_and_tracklet_permutation_do_not_change_posterior():
    base = partition_search(np.log(RATIOS), np.log(WEIGHTS), .7, queries=[(0, 1)])
    split = partition_search(np.log(RATIOS[:, [0, 0, 1]]),
                             np.log([.1, .25, .65]), .7, queries=[(0, 1)])
    permuted = partition_search(np.log(RATIOS[[2, 0, 3, 1]]), np.log(WEIGHTS), .7,
                                queries=[(1, 3)])
    assert split["queries"][0]["probability_interval"] == pytest.approx(base["queries"][0]["probability_interval"])
    assert permuted["queries"][0]["probability_interval"] == pytest.approx(base["queries"][0]["probability_interval"])


def test_extreme_likelihoods_stay_in_log_space():
    result = partition_search(np.array([[1000., -1000.], [1000., -1000.], [-1000., 1000.]]),
                              np.log([.5, .5]), 1.)
    assert result["status"] == "finite_prior_complete"
    json.dumps(result, allow_nan=False)
    assert result["ranked_candidates"][0]["indices"] == [0, 1]


def test_deadline_never_silently_drops_partition_mass():
    result = partition_search(np.log(RATIOS), np.log(WEIGHTS), 1.,
                              deadline=1., clock=lambda: 2., queries=[(0, 1)])
    assert result["stop_reason"] == "time_budget"
    assert result["remaining_assignments"] == 15
    assert result["unresolved_mass_upper"] == 1


def test_likelihoods_against_independent_dop853_brent():
    value = example()
    states = np.array([[2.5, .1, .03, -.002, .011, .001],
                       [2.8, .2, -.02, .002, .03, -.001]])
    value["orbit_prior"] = [dict(id=str(i), state=s.tolist(), weight=1.) for i, s in enumerate(states)]
    def rhs(t, state):
        return np.r_[state[3:], -GM["sun"]*state[:3]/np.linalg.norm(state[:3])**3]
    solutions = [solve_ivp(rhs, (0., 61.), s, method="DOP853", dense_output=True,
                           rtol=2.3e-14, atol=2e-15) for s in states]
    value["tracklets"] = []
    expected = []
    for i, time in enumerate([1., 20., 60.]):
        rows = []
        reference = np.zeros(2)
        for j, t in enumerate([time, time+.02]):
            angle = np.sqrt(GM["sun"])*t
            observer = np.array([np.cos(angle), np.sin(angle), .00002])
            directions = []
            for sol in solutions:
                emitted = brentq(lambda e: e+np.linalg.norm(sol.sol(e)[:3]-observer)/C_AU_DAY-t,
                                 t-.1, t, xtol=1e-13)
                vector = sol.sol(emitted)[:3]-observer
                directions.append(vector/np.linalg.norm(vector))
            measured = directions[0]
            ra = np.arctan2(measured[1], measured[0])
            dec = np.arctan2(measured[2], np.hypot(*measured[:2]))
            sigma = .5
            rows.append(dict(id=f"{i}-{j}", observatory="independent-test",
                             mjd_tdb=60000.+t, observer_heliocentric_au=observer.tolist(),
                             ra_deg=float(np.rad2deg(ra) % 360), dec_deg=float(np.rad2deg(dec)),
                             sigma_arcsec=sigma))
            kappa = 1/np.deg2rad(sigma/3600)**2
            for k, direction in enumerate(directions):
                predicted_ra = np.arctan2(direction[1], direction[0])
                predicted_dec = np.arctan2(direction[2], np.hypot(*direction[:2]))
                haversine = (np.sin((predicted_dec-dec)/2)**2
                             +np.cos(dec)*np.cos(predicted_dec)*np.sin((predicted_ra-ra)/2)**2)
                reference[k] += np.log(2*kappa)-2*kappa*haversine
        value["tracklets"].append(dict(id=str(i), observations=rows))
        expected.append(reference)
    value["queries"] = []
    actual, count = log_likelihood_ratios(validate(value))
    assert count == 2
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-6)


@pytest.mark.parametrize("change,match", [
    ("duplicate_observation", "Duplicate observation"),
    ("duplicate_tracklet", "Duplicate tracklet"),
    ("bad_query", "unknown tracklet"),
    ("singular_orbit", "radius"),
    ("nonfinite", "finite"),
])
def test_invalid_data_is_rejected_without_pruning(change, match):
    value = example()
    if change == "duplicate_observation":
        value["tracklets"][1]["observations"][0]["id"] = value["tracklets"][0]["observations"][0]["id"]
    elif change == "duplicate_tracklet":
        value["tracklets"][1]["id"] = "A"
    elif change == "bad_query":
        value["queries"] = [["A", "missing"]]
    elif change == "singular_orbit":
        value["orbit_prior"][0]["state"][:3] = [0., 0., 0.]
    else:
        value["orbit_prior"][0]["state"][0] = float("nan")
    with pytest.raises(ValueError, match=match):
        validate(value)


def test_likelihood_timeout_and_failure_are_not_zero_likelihood(monkeypatch):
    import magictools.solve_itf as module
    value = example()
    monkeypatch.setattr(module, "log_likelihood_ratios", lambda *a: (None, 0))
    report = run(value)
    assert report["unresolved_mass_upper"] == 1.
    assert report["status"] == "incomplete"
    assert not report["full_itf_search_complete"]
    def broken(*a):
        raise ArithmeticError("propagation failed")
    monkeypatch.setattr(module, "log_likelihood_ratios", broken)
    with pytest.raises(ArithmeticError, match="propagation failed"):
        run(value)


def test_cli_example_direct_run_and_provenance(tmp_path, capsys):
    def call(*args):
        code = agent_cli.main(list(map(str, args)))
        return code, json.loads(capsys.readouterr().out)
    folder, output = tmp_path/"input", tmp_path/"run"
    code, result = call("example", "solve_itf", "--output", folder)
    assert code == 0, result
    code, result = call("validate", "solve_itf", folder/"input.json")
    assert code == 0, result
    code, result = call("solve_itf", folder/"input.json", "--output", output)
    assert code == 0, result
    report = json.loads((output/"report.json").read_text())
    assert report["status"] == "finite_prior_complete"
    assert report["snapshot_id"] == "synthetic-circular-45-days"
    assert not report["continuous_orbit_coverage_certified"]
    assert not report["full_objective_complete"]
    assert report["excluded_continuous_orbit_mass_upper"] is None
    assert report["ranked_candidates"][0]["tracklets"] == ["A", "B", "C"]
    assert (output/"provenance.json").exists()
    before = (output/"report.json").read_bytes()
    code, result = call("solve_itf", folder/"input.json", "--output", output)
    assert code == 2 and (output/"report.json").read_bytes() == before


@pytest.mark.parametrize("activity", [.001, .5, 1., 2., 100.])
def test_mass_ceiling_across_link_prior_regimes(activity):
    ratios = np.exp(np.random.default_rng(17).normal(0., 3., (4, 3)))
    weights = [.1, .2, .7]
    truth, total = exhaustive(ratios, weights, activity)
    for budget in [3, 8, 15, 21]:
        result = partition_search(np.log(ratios), np.log(weights), activity,
                                  max_nodes=budget, queries=[(0, 1, 2)])
        event = sum(p for partition, p in truth.items()
                    if any({0, 1, 2} <= set(block) for block in partition))
        lower, upper = result["queries"][0]["probability_interval"]
        assert lower-1e-12 <= event <= upper+1e-12
        assert result["log_partition_function_interval"][1] >= math.log(total)-1e-12


def test_nonempty_unresolved_mass_does_not_underflow_to_zero():
    result = partition_search(np.array([[1000.], [1000.]]), np.array([0.]), 1.,
                              max_nodes=3)
    assert result["status"] == "incomplete"
    assert result["remaining_assignments"] == 1
    assert 0 < result["unresolved_mass_upper"] < 1e-300
    assert result["log_unresolved_mass_upper"] == pytest.approx(-2000.)
