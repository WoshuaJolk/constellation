import copy
import json
from math import comb

import numpy as np
import pytest

from magictools import ejector_fit
from magictools.ejector_fit import EncounterProblem
from magictools.ejector_search import (control_rank, match_counterparts, rotated_catalogue,
                                      run_search, search_family, validate_candidate, validate_search)
from magictools.galactic import GalacticPotential, KMS_TO_PC_MYR
from magictools.stellar_inputs import icrs_to_galactocentric
from test_ejector_fit import observation, repeated_spec


def survey_spec():
    template = repeated_spec()
    template.pop('encounters'); template.pop('notes', None)
    birth = np.array([500., 100., 40.])
    age = 2.13
    velocities = [[50, 20, 10], [-30, 40, -10], [20, -40, 20]]
    stars = [observation('ejecta_'+str(i), birth+np.array(v)*age*KMS_TO_PC_MYR, v) for i, v in enumerate(velocities)]
    stars.insert(1, observation('unrelated', [100, -200, 20], [10, -10, 0]))
    for s in stars: s['covariance'] = s['covariance'].tolist()
    template.update(objects=stars, archive_output='archive', catalogue_description='Synthetic blind-membership test',
                    controls=dict(method='proper_motion_rotation', n_controls=2, seed=456),
                    search=dict(group_size=3, lookback_bounds_myr=[.1, 5.], radius_pc=0.,
                                report_chi2_max=10., max_hypotheses=10, workers=1))
    template['fit']['starts_myr'] = [[1.], [3.]]
    template['fit']['max_iterations'] = 60
    return template


def test_all_combinations_and_unknown_injection_recovered():
    spec = survey_spec()
    assert validate_search(spec) == comb(4, 3)
    ledger, candidates, minimum = search_family(spec)
    assert len(ledger) == 4
    assert len({tuple(r['indices']) for r in ledger}) == 4
    assert minimum < 1e-10
    best = candidates[0]
    assert best['star_ids'] == ['ejecta_0', 'ejecta_1', 'ejecta_2']
    assert best['best_fit']['events'][0]['lookback_myr'] == pytest.approx(2.13, abs=1e-7)
    validation = validate_candidate(spec, best)
    assert validation['passed']
    assert validation['relative_rms_speed_kms'] > 30


def test_search_wide_rank_counts_stronger_groups_and_ties():
    # One control has an unrelated stronger group: it must count as an exceedance.
    assert control_rank(3., [2., 3., 9., None]) == 3/5
    assert control_rank(0., [None]*19) == .05
    assert control_rank(4., []) is None


def test_rotation_preserves_covariance_likelihood_and_selected_scalar_speeds():
    spec = survey_spec()
    cov = np.array(spec['objects'][0]['covariance'])
    cov[0, 1] = cov[1, 0] = .00005
    spec['objects'][0]['covariance'] = cov.tolist()
    rotated = rotated_catalogue(spec['objects'], np.random.default_rng(10))
    for a, b in zip(spec['objects'], rotated):
        assert a['mean'][0] == b['mean'][0] and a['mean'][3] == b['mean'][3]
        assert np.linalg.norm(a['mean'][1:3]) == pytest.approx(np.linalg.norm(b['mean'][1:3]))
        np.testing.assert_allclose(np.linalg.eigvalsh(a['covariance']), np.linalg.eigvalsh(b['covariance']), atol=1e-12)
    assert spec['objects'][0]['mean'] != rotated[0]['mean']


def test_cli_archive_controls_and_independent_validation(tmp_path):
    spec = survey_spec()
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(spec))
    result = run_search(path)
    assert result['hypotheses_per_catalogue'] == 4
    assert result['catalogues_searched'] == 3
    assert result['candidates'][0]['search_wide_rotation_rank'] == 1/3
    assert result['completion']['all_declared_combinations_attempted']
    assert not result['full_objective_complete']
    assert result['discovery_probability'] is None
    assert (tmp_path/'archive/complete.json').exists()
    with np.load(tmp_path/'archive/nominal_tracebacks.npz') as archive:
        assert archive['states_pc_kms'].shape == (121, 4, 6)
    for name in ['observed_fits.json', 'control_0000.json', 'control_0001.json']:
        assert len(json.loads((tmp_path/'archive'/name).read_text())) == 4
    with pytest.raises(FileExistsError): run_search(path)


def test_source_is_propagated_to_encounter_not_matched_at_present():
    spec = survey_spec()
    _, candidates, _ = search_family(spec)
    candidate = candidates[0]
    candidate['independent_validation'] = validate_candidate(spec, candidate)
    birth = np.array([500., 100., 40.])
    velocity = np.array([50., -20., 5.])
    source = observation('moving_source', birth+velocity*2.13*KMS_TO_PC_MYR, velocity)
    mean = source['mean']
    icrs = [source['ra_deg'], source['dec_deg'], 1000/mean[0], *mean[1:]]
    states = icrs_to_galactocentric(np.array([icrs]), spec['frame'])
    sources = dict(objects=[dict(id='moving_source', icrs=icrs, match_radius_pc=1.)],
                   excluded=[], coverage='Synthetic complete single-source fixture')
    result = match_counterparts(candidate, sources, states, GalacticPotential(spec['potential']), .02)
    assert result['matches'][0]['separation_pc'] < 1e-5
    assert np.linalg.norm(states[0, :3]-candidate['independent_validation']['origin_pc']) > 100


def test_limits_and_invalid_late_rows_fail_before_archive(tmp_path):
    spec = survey_spec()
    spec['search']['max_hypotheses'] = 3
    with pytest.raises(ValueError, match='no truncated'): validate_search(spec)
    spec['search']['max_hypotheses'] = 10
    spec['objects'][-1]['covariance'][0][0] = -1.
    path = tmp_path/'invalid.json'; path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='positive definite'): run_search(path)
    assert not (tmp_path/'archive').exists()


def test_compiled_integrator_matches_numpy_for_all_components(monkeypatch):
    if ejector_fit.compiled_positions is None:
        pytest.skip('Optional numba accelerator is not installed')
    spec = repeated_spec()
    spec['potential'] = dict(components=[dict(type='plummer', mass_msun=5e9, scale_pc=500),
        dict(type='miyamoto_nagai', mass_msun=6.8e10, a_pc=3000, b_pc=280),
        dict(type='log_halo', speed_kms=180, core_pc=12000)])
    problem = EncounterProblem(spec)
    points = np.c_[np.random.default_rng(45).normal(size=(13, 12)), np.tile([1.3, 3.1], (13, 1))]
    compiled = problem.positions(points, refinement=2)
    monkeypatch.setattr(ejector_fit, 'compiled_positions', None)
    fallback = problem.positions(points, refinement=2)
    for a, b in zip(compiled, fallback): np.testing.assert_allclose(a, b, rtol=0, atol=2e-10)
