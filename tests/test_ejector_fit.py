import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from magictools.ejector_fit import EncounterProblem, fit_encounters
from magictools.galactic import KMS_TO_PC_MYR
from magictools.stellar_inputs import icrs_to_galactocentric


DATA = Path(__file__).resolve().parents[1]/'research/ejector_fit/data'


def observation(name, position, velocity):
    d = np.linalg.norm(position)
    ra = np.arctan2(position[1], position[0])
    dec = np.arcsin(position[2]/d)
    east = np.array([-np.sin(ra), np.cos(ra), 0.])
    north = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
    mean = [1000/d, np.dot(velocity, east)/(4.740470463533349*d/1000),
            np.dot(velocity, north)/(4.740470463533349*d/1000), np.dot(velocity, position)/d]
    return dict(id=name, ra_deg=float(np.rad2deg(ra)%360), dec_deg=float(np.rad2deg(dec)),
                mean=mean, covariance=np.diag([.01, .02, .02, .5])**2)


def repeated_spec():
    spec = json.loads((DATA/'hip1997_triple.json').read_text())
    spec['potential'] = dict(components=[])
    p, v = np.array([500., 150, 100]), np.array([3., 10, 0])
    objects = [observation('source', p, v)]
    for name, age, speed in [('a', 1.3, [50., 40, 10]), ('b', 3.1, [-40., 30, 20])]:
        speed = np.array(speed)
        objects.append(observation(name, p+(speed-v)*age*KMS_TO_PC_MYR, speed))
    spec['objects'] = objects
    spec['encounters'] = [dict(id=name, members=['source', name], lookback_bounds_myr=[.1, 5.], radius_pc=0.) for name in ['a', 'b']]
    spec['fit']['starts_myr'] = [[.5, .5], [2., 4.]]
    return spec


def test_repeated_launches_share_one_source_state_and_recover_distinct_ages():
    spec = repeated_spec()
    result = fit_encounters(spec)
    best = result['best_fit']
    assert best['chi2'] < 1e-8
    np.testing.assert_allclose([e['lookback_myr'] for e in best['events']], [1.3, 3.1], atol=1e-7)
    assert len(best['fitted_objects']) == 3
    assert result['discovery_probability'] is None
    # They are not a simultaneous three-star ejection. Failure is inconclusive,
    # but any feasible simultaneous fit must pay a large measurement penalty.
    spec['encounters'] = [dict(id='together', members=['source', 'a', 'b'], lookback_bounds_myr=[.1, 5.], radius_pc=0.)]
    spec['fit']['starts_myr'] = [[1.3], [3.1]]
    same = fit_encounters(spec)
    assert same['best_fit'] is None or same['best_fit']['chi2'] > 100


def test_affine_frame_matches_astropy_and_correlated_mahalanobis():
    spec = repeated_spec()
    spec['objects'][0]['covariance'][0, 1] = spec['objects'][0]['covariance'][1, 0] = .0001
    problem = EncounterProblem(spec)
    x = np.r_[np.linspace(-.5, .5, 12), 1.3, 3.1]
    obs = problem.observations(x)[0]
    samples = np.array([[r['ra_deg'], r['dec_deg'], 1000/o[0], *o[1:]] for r, o in zip(spec['objects'], obs)])
    np.testing.assert_allclose(problem.states(x)[0], icrs_to_galactocentric(samples, spec['frame']), atol=2e-8, rtol=0)
    cost = sum((o-problem.means[i])@np.linalg.solve(spec['objects'][i]['covariance'], o-problem.means[i]) for i, o in enumerate(obs))
    assert cost == pytest.approx(np.dot(x[:12], x[:12]))


def test_real_historical_orion_fit_and_independent_dop853():
    spec = json.loads((DATA/'hip1997_triple.json').read_text())
    spec['fit']['starts_myr'] = [[1.], [4.]]
    result = fit_encounters(spec)
    best = result['best_fit']
    assert best is not None
    assert best['chi2'] == pytest.approx(5.62166, abs=.002)
    age = best['events'][0]['lookback_myr']
    assert age == pytest.approx(2.49325, abs=.001)
    problem = EncounterProblem(spec)
    z = np.concatenate([np.linalg.solve(f, np.array(r['mean'])-m) for r, m, f in zip(best['fitted_objects'], problem.means, problem.factors)])
    states = problem.states(np.r_[z, age])[0]
    def rhs(t, y):
        s = y.reshape(-1, 6)
        return np.c_[s[:, 3:]*KMS_TO_PC_MYR, problem.potential.acceleration(s[:, :3])/KMS_TO_PC_MYR].ravel()
    solution = solve_ivp(rhs, [0, -age], states.ravel(), method='DOP853', rtol=1e-12, atol=1e-12)
    p = solution.y[:, -1].reshape(-1, 6)[:, :3]
    assert solution.success
    assert np.linalg.norm(p-p.mean(axis=0), axis=1).max() < 2e-5


def test_finite_radius_and_failed_optimizer_not_reported_as_discovery():
    spec = repeated_spec()
    # One equality and one finite-radius constraint exercise both Jacobian blocks.
    spec['encounters'][1]['radius_pc'] = 2.
    spec['fit']['starts_myr'] = [[1.3, 3.1]]
    result = fit_encounters(spec)
    assert result['best_fit']['chi2'] < 1e-12
    assert all(e['max_centroid_radius_pc'] <= 2. for e in result['best_fit']['events'])
    spec['fit']['starts_myr'] = [[.1, .1]]
    spec['fit']['max_iterations'] = 1
    result = fit_encounters(spec)
    assert result['best_fit'] is None
    assert result['status'] == 'no_converged_feasible_fit'
    assert not result['full_objective_complete']


@pytest.mark.parametrize('mutation', ['covariance', 'duplicate', 'unused', 'parallax', 'start', 'radius'])
def test_invalid_scientific_inputs_rejected(mutation):
    spec = repeated_spec()
    if mutation == 'covariance': spec['objects'][0]['covariance'][0, 0] = -1
    if mutation == 'duplicate': spec['encounters'][0]['members'] = ['a', 'a']
    if mutation == 'unused': spec['objects'].append(dict(spec['objects'][0], id='unused'))
    if mutation == 'parallax': spec['objects'][0]['mean'][0] = -1
    if mutation == 'start': spec['fit']['starts_myr'] = [[0.]]
    if mutation == 'radius': spec['encounters'][0]['radius_pc'] = -1
    with pytest.raises(ValueError): EncounterProblem(spec)
