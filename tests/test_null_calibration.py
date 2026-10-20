import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta

from magictools.io import run_nulls


def run(tmp_path, manifest):
    path=tmp_path/'nulls.json'; path.write_text(json.dumps(manifest)); return run_nulls(path)


def test_joint_calibration_integral_matches_numerical_quadrature(tmp_path):
    value=dict(schema_version=1,hypotheses=[dict(id='bright',prior_weight=1),dict(id='faint',prior_weight=1)],
               calibrations=[dict(id='selection',recovered=[8,2],injected=[10,10])],
               non_detections=[dict(id=f'e{i}',calibration_id='selection') for i in range(3)],
               candidate_exposures=[dict(id='next',calibration_id='selection')])
    report=run(tmp_path,value)
    likelihood=np.array([quad(lambda p:(1-p)**3*beta.pdf(p,a,b),0,1)[0] for a,b in [(8.5,2.5),(2.5,8.5)]])
    posterior=likelihood/likelihood.sum()
    np.testing.assert_allclose(report['posterior_weights'],posterior,rtol=1e-8)
    expected=posterior @ (np.array([8.5,2.5])/14)
    assert report['ranked_exposures'][0]['detection_probability'] == pytest.approx(expected)
    assert report['calibration_posteriors_conditional_on_hypothesis'][0]['observed_nulls'] == 3


def test_shared_calibration_is_not_treated_as_fixed_efficiency(tmp_path):
    value=dict(schema_version=1,hypotheses=[dict(id='one',prior_weight=1)],
               calibrations=[dict(id='unknown',recovered=[0],injected=[0])],
               non_detections=[dict(id='a',calibration_id='unknown'),dict(id='b',calibration_id='unknown')],
               candidate_exposures=[dict(id='next',calibration_id='unknown')])
    report=run(tmp_path,value)
    assert np.exp(report['log_evidence']) == pytest.approx(3/8)  # Not (1/2)^2.
    assert report['ranked_exposures'][0]['detection_probability'] == pytest.approx(1/6)


def test_orbit_information_marginalizes_brightness_instead_of_optimizing_it(tmp_path):
    value=dict(schema_version=1,hypotheses=[dict(id=f'{oid}-{f}',orbit_id=oid,reference_flux=f,prior_weight=1) for oid in ['a','b'] for f in [1,10]],
               non_detections=[],ranking_target='orbit',candidate_exposures=[
                   dict(id='brightness-only',detection_probability=[0,1,0,1]),
                   dict(id='orbit-informative',detection_probability=[.1,.1,.9,.9])])
    report=run(tmp_path,value)
    assert report['ranked_exposures'][0]['id'] == 'orbit-informative'
    assert report['ranked_exposures'][1]['expected_information_nats'] == pytest.approx(0)
    assert [r['posterior_weight'] for r in report['orbit_posterior']] == pytest.approx([.5,.5])
    value['ranking_target']='hypothesis'
    assert run(tmp_path,value)['ranked_exposures'][0]['id'] == 'brightness-only'


def test_brightness_weights_survive_null_update_and_zero_mass_orbits(tmp_path):
    value=dict(schema_version=1,hypotheses=[dict(id='af',orbit_id='a',prior_weight=1),dict(id='ab',orbit_id='a',prior_weight=3),dict(id='b',orbit_id='b',prior_weight=0)],
               non_detections=[dict(id='e',detection_probability=[0,.9,0])],ranking_target='orbit',candidate_exposures=[])
    report=run(tmp_path,value)
    np.testing.assert_allclose(report['posterior_weights'],[1/1.3,.3/1.3,0])
    assert report['orbit_posterior'][1]['posterior_weight'] == 0


def test_large_calibration_counts_preserve_small_null_log_probability(tmp_path):
    value=dict(schema_version=1,hypotheses=[dict(id='one',prior_weight=1)],
               calibrations=[dict(id='c',recovered=[0],injected=[1000000000])],non_detections=[dict(id='a',calibration_id='c')])
    report=run(tmp_path,value)
    assert report['log_evidence'] == pytest.approx(-np.log1p(.5/1000000000.5),rel=1e-12)


@pytest.mark.parametrize('change', ['unknown','counts','partial_groups'])
def test_bad_calibrations_and_partial_groups_fail(tmp_path,change):
    value=dict(schema_version=1,hypotheses=[dict(id='a',prior_weight=1),dict(id='b',prior_weight=1)],
               calibrations=[dict(id='c',recovered=[1,1],injected=[2,2])],non_detections=[dict(id='e',calibration_id='c')])
    if change=='unknown': value['non_detections'][0]['calibration_id']='missing'
    if change=='counts': value['calibrations'][0]['recovered']=[3,1]
    if change=='partial_groups': value['hypotheses'][0]['orbit_id']='a'
    with pytest.raises(ValueError): run(tmp_path,value)
