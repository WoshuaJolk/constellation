import json
import subprocess
import sys

import numpy as np
import pytest

from magictools.selection import run_selection


def save(tmp_path,value):
    path=tmp_path/'selection.json'; path.write_text(json.dumps(value)); return path


@pytest.fixture
def population():
    return dict(schema_version=1,exposures=[dict(id=f'e{i}',observation_id=f'physical-{i}',surveys=[survey],mjd_utc=60000.8+i//2) for i,survey in enumerate(['a','a','b','b'])],
                objects=[dict(id='bright',weight=1,detection_probability=[.5]*4),dict(id='faint',weight=3,detection_probability=[.1]*4)],
                discovery_rules=[dict(id='rule-a',surveys=['a'],min_visits_per_night=2,min_nights=1),dict(id='rule-b',surveys=['b'],min_visits_per_night=2,min_nights=1),dict(id='cross',surveys=['a','b'],min_visits_per_night=2,min_nights=2)])


def test_exact_cadence_and_weighted_union(population,tmp_path):
    report=run_selection(save(tmp_path,population))
    first=report['objects'][0]['probabilities']
    assert first['rule-a'] == pytest.approx(.25)
    assert first['cross'] == pytest.approx(1/16)
    assert first['union'] == pytest.approx(1-.75**2)
    assert report['population_probabilities']['union'] == pytest.approx(.25*(1-.75**2)+.75*(1-.99**2))
    assert report['union_mc_interval'] is None


def test_shared_exposure_is_not_double_counted_between_surveys(tmp_path):
    value=dict(schema_version=1,exposures=[dict(id='shared',observation_id='same',surveys=['a','b'],mjd_utc=60000)],
               objects=[dict(id='object',weight=1,detection_probability=[.5])],
               discovery_rules=[dict(id=s,surveys=[s],min_visits_per_night=1,min_nights=1) for s in ['a','b']])
    result=run_selection(save(tmp_path,value))
    assert result['population_probabilities'] == pytest.approx(dict(a=.5,b=.5,union=.5))


def test_night_boundary_and_span_are_applied(tmp_path):
    value=dict(schema_version=1,exposures=[dict(id=f'e{i}',observation_id=f'o{i}',surveys=['a'],mjd_utc=t) for i,t in enumerate([60000.49,60000.51,60003.6])],
               objects=[dict(id='one',weight=1,detection_probability=[1,1,1])],
               discovery_rules=[dict(id='rule',surveys=['a'],min_visits_per_night=1,min_nights=2,max_night_span=1)])
    assert run_selection(save(tmp_path,value))['population_probabilities']['union'] == 1
    value['discovery_rules'][0]['max_night_span']=0
    assert run_selection(save(tmp_path,value))['population_probabilities']['union'] == 0
    value['night_boundary_utc_hours']=0
    value['discovery_rules'][0]['max_night_span']=1
    assert run_selection(save(tmp_path,value))['population_probabilities']['union'] == 0


def test_monte_carlo_agrees_with_exact_and_is_reproducible(population,tmp_path):
    exact=run_selection(save(tmp_path,population))
    population.update(draws_per_object=40000,seed=123,confidence=.99)
    result=run_selection(save(tmp_path,population),monte_carlo=True)
    assert result == run_selection(save(tmp_path,population),monte_carlo=True)
    low,high=result['union_mc_interval']
    assert low < exact['population_probabilities']['union'] < high
    for e,m in zip(exact['objects'],result['objects']):
        assert m['probabilities']['union'] == pytest.approx(e['probabilities']['union'],abs=.007)


def test_poisson_count_model_and_impossible_zero_efficiency(population,tmp_path):
    population.update(population_size=100,observed_count=3)
    result=run_selection(save(tmp_path,population))
    mean=result['population_probabilities']['union']*100
    assert result['population_count_model']['expected_selected_count'] == pytest.approx(mean)
    from scipy.stats import poisson
    assert result['population_count_model']['log_count_likelihood'] == pytest.approx(poisson.logpmf(3,mean))
    for obj in population['objects']: obj['detection_probability']=[0]*4
    result=run_selection(save(tmp_path,population))
    assert result['population_count_model']['zero_probability']
    assert result['population_count_model']['log_count_likelihood'] is None


@pytest.mark.parametrize('bad',['physical_duplicate','probability','exact_limit','unknown_survey'])
def test_reject_bad_or_infeasible_inputs(population,tmp_path,bad):
    if bad=='physical_duplicate': population['exposures'][1]['observation_id']='physical-0'
    if bad=='probability': population['objects'][0]['detection_probability']=[1.1]*4
    if bad=='unknown_survey': population['discovery_rules'][0]['surveys']=['unknown']
    if bad=='exact_limit':
        population['exposures']=[dict(id=str(i),observation_id=str(i),surveys=['a','b'],mjd_utc=60000+i) for i in range(21)]
        for row in population['objects']: row['detection_probability']=[.5]*21
    with pytest.raises(ValueError): run_selection(save(tmp_path,population))


def test_selection_cli(population,tmp_path):
    path=save(tmp_path,population)
    result=subprocess.run([sys.executable,'-m','magictools','map_complete',str(path)],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout)['algorithm']['outcomes_per_object'] == 16
