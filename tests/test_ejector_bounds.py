import copy
import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

flint = pytest.importorskip('flint')
from flint import arb, ctx

from magictools.ejector_bounds import bound_family, outward_float_bounds, validated_step, refine_group, certify_witness
from magictools.ejector_fit import EncounterProblem
from magictools.ejector_search import family_spec
from magictools.galactic import KMS_TO_PC_MYR
from test_ejector_search import survey_spec


def test_excludes_impossible_groups_without_losing_continuous_injection():
    spec = survey_spec()
    spec['fit']['max_step_myr'] = .13  # Event 2.13 Myr is not a grid point.
    report, bounds = bound_family(spec)
    assert report['excluded_count'] == 3
    assert report['unresolved_count'] == 1
    assert report['unresolved_groups'][0]['star_ids'] == ['ejecta_0', 'ejecta_1', 'ejecta_2']
    assert report['total_mahalanobis_cost_at_most'] == 10.
    assert not report['full_objective_complete']
    assert bounds.shape == (39, 4, 3, 2)


def test_interval_tubes_contain_independent_ode_trajectories():
    spec = survey_spec()
    spec['potential'] = dict(components=[dict(type='plummer', mass_msun=5e9, scale_pc=500),
        dict(type='miyamoto_nagai', mass_msun=6.8e10, a_pc=3000, b_pc=280),
        dict(type='log_halo', speed_kms=180, core_pc=12000)])
    spec['fit']['max_step_myr'] = .2
    report, bounds = bound_family(spec)
    problem = EncounterProblem(family_spec(spec, [0, 1, 2, 3]))
    rng = np.random.default_rng(249)
    z = rng.normal(size=(12, 16))
    z *= np.sqrt(10.)/np.maximum(np.linalg.norm(z, axis=1)[:, None], np.sqrt(10.))
    states = problem.states(np.c_[z, np.ones(len(z))])
    def rhs(t, y):
        s = y.reshape(-1, 6)
        return np.c_[s[:, 3:]*KMS_TO_PC_MYR, problem.potential.acceleration(s[:, :3])/KMS_TO_PC_MYR].ravel()
    solution = solve_ivp(rhs, [0., -5.], states.ravel(), method='DOP853', rtol=1e-12, atol=1e-12, dense_output=True)
    for k, box in enumerate(bounds):
        times = -np.linspace(k*.2, (k+1)*.2, 7)
        positions = solution.sol(times).T.reshape(7, 12, 4, 6)[..., :3]
        assert np.all(positions >= box[..., 0])
        assert np.all(positions <= box[..., 1])


def test_free_flow_exact_solution_and_failed_enclosure():
    state = [arb(v) for v in [10, -3, 40, 2, -5, 7]]
    tube, endpoint = validated_step(state, arb(-2), dict(components=[]))
    exact = [-6+12, 7, 26, 2, -5, 7]
    assert all(a.contains(arb(b)) for a, b in zip(endpoint, exact))
    for i in range(3):
        assert tube[i].contains(state[i])
        assert tube[i].contains(state[i]-2*state[i+3])
    with pytest.raises(ValueError, match='No exclusion certificate'):
        validated_step(state, arb(-2), dict(components=[]), max_iterations=0)


def test_checked_float_endpoints_and_precision_restored():
    ball = arb('1.234567890123456789', '0.000000000000000001')
    lo, hi = outward_float_bounds(ball)
    assert arb(lo) <= ball.lower() and arb(hi) >= ball.upper()
    original = ctx.prec
    spec = survey_spec(); spec['objects'][0]['covariance'][0][1] = 1e-15
    with pytest.raises(ValueError, match='exactly symmetric'): bound_family(spec)
    assert ctx.prec == original


def test_branching_never_excludes_true_encounter_and_cap_is_inconclusive():
    spec = survey_spec()
    spec['fit']['max_step_myr'] = .1
    real = ['ejecta_0', 'ejecta_1', 'ejecta_2']
    result = refine_group(spec, real, max_nodes=30)
    assert not result['excluded']
    assert result['pending_boxes']
    assert result['nodes_visited'] == 30
    false = refine_group(spec, ['ejecta_0', 'unrelated', 'ejecta_2'], max_nodes=30)
    assert false['excluded'] and not false['pending_boxes']
    assert false['spatial_leaves'] >= 1


def test_arb_verifies_positive_witness_and_rejects_changed_observations():
    from magictools.ejector_search import search_family
    spec = survey_spec()
    _, candidates, _ = search_family(spec)
    spec['search']['radius_pc'] = .1
    fitted = candidates[0]['best_fit']
    witness = certify_witness(spec, candidates[0]['star_ids'], fitted)
    assert witness['certified']
    assert witness['centroid_radius_upper_pc'] < 1e-5
    damaged = copy.deepcopy(fitted)
    damaged['fitted_objects'][0]['mean'][3] += 20.
    assert not certify_witness(spec, candidates[0]['star_ids'], damaged)['certified']


def test_failed_enclosure_is_unbounded_not_excluded(monkeypatch):
    import magictools.ejector_bounds as module
    def fail(*args, **kwargs): raise ValueError('test forced enclosure failure')
    monkeypatch.setattr(module, 'validated_step', fail)
    result, boxes = bound_family(survey_spec())
    assert result['excluded_count'] == 0
    assert result['unresolved_count'] == 4
    assert len(result['unbounded_tube_fallbacks']) == 4
    assert np.isneginf(boxes[..., 0]).all() and np.isposinf(boxes[..., 1]).all()
