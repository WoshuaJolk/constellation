import sys
from pathlib import Path
import numpy as np
import pytest
from numba import cuda
from test_solve_itf import injected_catalogue
from magictools.itf.fit import Geometry,fit_chain
from magictools.itf.search import SearchConfig,solve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from gpu_pipeline_adapter import GPUBatch,cpu_batch


def test_batch_hook_preserves_cpu_search(injected_catalogue):
    cat,sites,_=injected_catalogue
    config=SearchConfig(workers=1,max_seeds=4,seconds=120)
    a=solve(cat,sites,config,progress=lambda _:None)
    b=solve(cat,sites,config,progress=lambda _:None,batch_fitter=cpu_batch)
    assert a['status']==b['status']=='bounded_search_complete'
    assert a['candidates']==b['candidates']
    for k in ['fits_attempted','fits_rejected','successful_extensions','triple_seeds_completed']:
        assert a['counts'][k]==b['counts'][k]


@pytest.mark.skipif(not cuda.is_available(),reason='CUDA or explicit simulator required')
def test_gpu_plan_replay_preserves_membership_and_acceptance(injected_catalogue):
    cat,sites,_=injected_catalogue;g=Geometry(cat,sites);g.prepare(range(6))
    state=np.array([2.4,.7,.3,-.004,.009,.002]);epoch=float(g.get([0,1,2,3,4,5])[0].mean())
    settings=dict(start_policy='provided',extra_seeds=[(state,epoch)],max_nfev=5)
    expected=fit_chain(g,[0,1,2],**settings)
    adapter=GPUBatch(allow_simulator=True);actual=adapter(g,[([0,1,2],settings)])[0]
    assert (expected is None)==(actual is None)
    if expected:
        assert actual['tracklet_indices']==expected['tracklet_indices']
        assert actual['observations']==expected['observations']
        assert actual['rms_arcsec']==pytest.approx(expected['rms_arcsec'],abs=1e-7)
    assert adapter.receipts[0]['starts']==1


def test_compiled_verifier_keeps_checks_and_replays_invalid_cost():
    from gpu_pipeline_adapter import checked_cpu_replay
    from test_itf_gpu import fixture_bank
    from magictools.itf.derivatives import solve_lm,residual_jacobian
    b=fixture_bank();bounds=np.r_[0,np.cumsum(np.bincount(b['case_index']))]
    states=[];flags=[];costs=[]
    for c in range(len(b['states'])):
        lo,hi=bounds[c:c+2];args=[b[k][lo:hi] for k in ('offsets','observers','observed','east','north')]
        x,r,j,ok=solve_lm(b['states'][c],*args,float(b['sigma']),5)
        states.append(x);flags.append(ok);costs.append(r@r)
    out=dict(states=np.array(states),success=np.array(flags),cost=np.array(costs));out['cost'][0]=np.nan
    actual,mask=checked_cpu_replay(b,out,5,threads=2)
    assert mask[0] and mask[-1]
    for c,(x,r,j,ok) in enumerate(actual):
        lo,hi=bounds[c:c+2];args=[b[k][lo:hi] for k in ('offsets','observers','observed','east','north')]
        expected=solve_lm(b['states'][c],*args,float(b['sigma']),5) if mask[c] else (out['states'][c],*residual_jacobian(out['states'][c],*args,float(b['sigma'])),out['success'][c])
        for a,e in zip((x,r,j,ok),expected):np.testing.assert_array_equal(a,e)
