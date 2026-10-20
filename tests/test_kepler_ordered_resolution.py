"""Stage-specific certificates must preserve exact child coordinate identities."""
import json
from pathlib import Path

import numpy as np
import pytest
from flint import ctx,arb

from research.kepler_transform.cascade_domain import solve_hybrid
from research.kepler_transform.constructive_experiments import domain
from research.kepler_transform.factorization import ARCSEC
from research.kepler_transform.ordered.resolution import local_grid,stage_phase,PERMUTATION
from research.kepler_transform.ordered.hierarchy import OrderedHierarchyPlan
from research.kepler_transform.reception_flow import CircularObserver


@pytest.fixture(autouse=True)
def precision():
    old=ctx.prec;ctx.prec=100
    yield
    ctx.prec=old


@pytest.fixture(scope='module')
def proof():
    return json.loads((Path(__file__).parents[1]/'research/kepler_transform/constructive_results/local.json').read_text())['certificates'][0]['transfer_certificate']


def test_instantaneous_fields_need_no_hidden_grid(proof):
    plan=local_grid(proof,CircularObserver(),[.125,.125],[1/ARCSEC]*2)
    assert plan['permutation']==list(PERMUTATION)
    for i in (0,1,2,5):
        assert plan['cells'][i]==plan['orders'][i]==1
        assert plan['certificates'][i]['kind']=='constant_instantaneous_stage'
        assert plan['certificates'][i]['phase_derivative_bound']==0
        assert plan['axes'][i]['error_bound']<=np.nextafter(0.,1.)
    assert plan['nodes']<2000
    assert plan['error_bound']<=1e-8


def test_native_first_angle_has_no_spurious_inverse_factor(proof):
    certificate=stage_phase(proof,CircularObserver(),3,32.,[.125,.125],[1/ARCSEC]*2)
    assert certificate['kind']=='native_child_field'
    expected=arb(1/ARCSEC)*proof['parent_widths'][3]
    assert arb(certificate['phase_derivative_bound'])>=expected
    assert certificate['phase_derivative_bound']<1.00000001*float(expected)


def test_internal_stages_cover_independent_derivative_witnesses(proof):
    observer=CircularObserver()
    c,w=np.array(proof['parent_center']),np.array(proof['parent_widths'])
    cc,cw=np.array(proof['child_center']),np.array(proof['child_widths'])
    rng=np.random.default_rng(811)
    rebase=lambda q:observer.chart(0.).rebase(q,.125,observer.chart(.125))
    for axis in PERMUTATION:
        selected=PERMUTATION[:PERMUTATION.index(axis)]
        certificate=stage_phase(proof,observer,axis,1.2,[.125,.375],[1/ARCSEC]*2)
        assert certificate['phase_derivative_bound']>0
        h=cc+rng.uniform(-.2,.2,(4,6))*cw
        for j in selected:
            h[:,j]=c[j]+rng.uniform(-.2,.2,4)*w[j]
        eps=1e-4
        hp,hm=h.copy(),h.copy()
        hp[:,axis]+=eps*w[axis];hm[:,axis]-=eps*w[axis]
        xp,_=solve_hybrid(proof,hp,selected,rebase,tolerance=1e-13)
        xm,_=solve_hybrid(proof,hm,selected,rebase,tolerance=1e-13)
        for t in [.125,.23,.375]:
            ap=observer.chart(0.).rebase(xp,t,observer.chart(t))[:,[3,4]]
            am=observer.chart(0.).rebase(xm,t,observer.chart(t))[:,[3,4]]
            derivative=np.sum(abs(ap-am),axis=1)/(2*eps*ARCSEC)
            assert max(derivative)<=certificate['phase_derivative_bound']*(1+1e-5)+1e-8


def test_ordered_graph_budget_execution_and_adjoint():
    *_,c,w=domain('near')
    observer=CircularObserver()
    plan=OrderedHierarchyPlan(observer,c,w,[-1e-7,1e-7],[.001/ARCSEC]*2,
                              tolerance=1e-5,neighborhood_ellipses=(2.,8.,32.))
    with pytest.raises(MemoryError):
        plan.materialize([[[0.,0.]]]*2,max_nodes=1)
    with pytest.raises(ValueError):
        plan.materialize([[[0.,0.]]]*2,max_nodes=float('nan'))
    frequencies=[np.array([[.001/ARCSEC,-.001/ARCSEC]])]*2
    operator=plan.materialize(frequencies)
    data=np.array([1+2j,.3-.7j])
    values=operator.transform(data)
    rng=np.random.default_rng(8)
    queries=c+rng.uniform(-.9,.9,(12,6))*w
    exact=sum(np.exp(1j*observer.chart(0.).rebase(queries,t,observer.chart(t))[:,[3,4]] @ frequencies[i][0])*data[i]
              for i,t in enumerate(plan.times))
    assert max(abs(operator.evaluate(values,queries)-exact))/sum(abs(data))<=plan.error+1e-12
    test=rng.normal(size=len(values))+1j*rng.normal(size=len(values))
    assert abs(np.vdot(test,values)-np.vdot(operator.adjoint(test),data))<1e-10
    assert operator.stats['actual_transfer_entries']==plan.stats['total_transfer_entries']


def test_replay_refuses_wrong_cascade_order_and_false_constant_stage(proof):
    from research.kepler_transform.ordered.verify import replay_grid
    observer=CircularObserver()
    plan=local_grid(proof,observer,[.125,.125],[1/ARCSEC]*2,neighborhood_ellipses=(2.,8.))
    plan['permutation']=[3,4,5,2,1,0]
    with pytest.raises(ValueError,match='wrong stage order'):
        replay_grid(plan,proof,observer)
    plan['permutation']=list(PERMUTATION)
    plan['time_range']=[.125,.375]
    with pytest.raises(ValueError,match='Stage certificate changed'):
        replay_grid(plan,proof,observer)
