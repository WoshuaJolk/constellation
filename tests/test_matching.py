import copy
import itertools
import json
import subprocess
import sys

import numpy as np
import pytest

from magictools.matching import run_match, solve_assignment, verify_certificate


def trap():
    return dict(detections=[dict(id='a',unassigned_score=0),dict(id='b',unassigned_score=0)],candidates=[
        dict(id='greedy',detection_ids=['a','b'],score=10),dict(id='left',detection_ids=['a'],score=6),dict(id='right',detection_ids=['b'],score=6)])


def brute_force(problem):
    best=sum(d['unassigned_score'] for d in problem['detections'])
    for select in itertools.product([False,True],repeat=len(problem['candidates'])):
        used=set();score=0;valid=True
        for candidate,take in zip(problem['candidates'],select):
            if take:
                if used & set(candidate['detection_ids']): valid=False;break
                used.update(candidate['detection_ids']); score+=candidate['score']
        if valid:
            score+=sum(d['unassigned_score'] for d in problem['detections'] if d['id'] not in used)
            best=max(best,score)
    return best


def test_beats_greedy_and_emits_verifiable_optimum():
    result=solve_assignment(trap())
    assert result['selected_candidate_ids'] == ['left','right']
    assert result['status'] == 'optimal'
    assert result['objective'] == result['upper_bound'] == 12
    assert verify_certificate(result['certificate'])['gap'] == 0


def test_budget_exhaustion_returns_honest_gap():
    result=solve_assignment(trap(),node_limit=1)
    assert result['status'] == 'bounded'
    assert result['objective'] == 10
    assert result['upper_bound'] == 12
    assert verify_certificate(result['certificate'])['gap'] == 2


def test_exact_integer_scores_larger_than_float_precision():
    problem=trap(); value=2**100
    problem['candidates'][0]['score']=2*value+1
    problem['candidates'][1]['score']=value+1
    problem['candidates'][2]['score']=value+1
    result=solve_assignment(problem)
    assert result['objective'] == 2*value+2
    assert result['selected_candidate_ids'] == ['left','right']


def test_background_scores_and_nonpositive_benefits():
    problem=dict(detections=[dict(id='a',unassigned_score=-10),dict(id='b',unassigned_score=5)],candidates=[dict(id='useful',detection_ids=['a'],score=-3),dict(id='bad',detection_ids=['b'],score=2)])
    result=solve_assignment(problem)
    assert result['objective'] == 2
    assert result['selected_candidate_ids'] == ['useful']
    assert result['unassigned_detection_ids'] == ['b']


def test_many_random_problems_match_independent_exhaustive_search():
    rng=np.random.default_rng(42)
    for _ in range(24):
        problem=dict(detections=[dict(id=str(i),unassigned_score=int(rng.integers(-3,4))) for i in range(6)],candidates=[])
        for i in range(9):
            ids=rng.choice(6,size=int(rng.integers(1,4)),replace=False)
            problem['candidates'].append(dict(id=str(i),detection_ids=[str(j) for j in ids],score=int(rng.integers(-3,15))))
        result=solve_assignment(problem)
        assert result['status'] == 'optimal'
        assert result['objective'] == brute_force(problem)


def test_empty_candidate_set_and_ties():
    problem=trap(); problem['candidates']=[]
    result=solve_assignment(problem)
    assert result['status'] == 'optimal' and result['objective'] == 0
    problem=trap();problem['candidates'][0]['score']=12
    assert solve_assignment(problem)['objective'] == 12  # No uniqueness claim.


def test_tampered_certificate_omissions_and_false_cliques_rejected():
    certificate=solve_assignment(trap())['certificate']
    bad=copy.deepcopy(certificate);bad['tree']={'cliques':[]}
    with pytest.raises(ValueError,match='omitted'): verify_certificate(bad)
    bad=copy.deepcopy(certificate);bad['tree']={'cliques':[['greedy','left','right']]}
    with pytest.raises(ValueError,match='compatible'): verify_certificate(bad)
    bad=copy.deepcopy(certificate);bad['selected_candidate_ids']=['greedy','left']
    with pytest.raises(ValueError,match='more than once'): verify_certificate(bad)


def test_float_scores_are_rejected_instead_of_silently_rounded():
    problem=trap();problem['candidates'][0]['score']=10.1
    with pytest.raises(ValueError,match='exact integers'): solve_assignment(problem)


def test_match_and_verifier_cli(tmp_path):
    path=tmp_path/'input.json';path.write_text(json.dumps(dict(schema_version=1,**trap())))
    result=run_match(path)
    report=tmp_path/'report.json';report.write_text(json.dumps(result))
    output=subprocess.run([sys.executable,'-m','magictools','verify_match',str(report)],capture_output=True,text=True)
    assert output.returncode == 0,output.stderr
    assert json.loads(output.stdout)['valid']
    result['objective']+=1;report.write_text(json.dumps(result))
    assert subprocess.run([sys.executable,'-m','magictools','verify_match',str(report)],capture_output=True).returncode == 2
