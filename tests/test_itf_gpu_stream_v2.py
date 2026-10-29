import sys
from pathlib import Path
import numpy as np
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from test_itf_gpu import fixture_bank
from test_solve_itf import injected_catalogue
from magictools.itf.gpu import cpu_reference
from magictools.itf.fit import Geometry,fit_chain
from gpu_pipeline_adapter import checked_cpu_arrays
from gpu_stream_adapter_v2 import checked
from gpu_batch_host import prepare_bank,finish_group


def test_verification_views_and_gathers_exact():
    original=fixture_bank()
    for reverse in [False,True]:
        bank=dict(original,geometry_index=np.arange(len(original['offsets']),dtype=np.int64))
        if reverse:
            bank['geometry_index']=bank['geometry_index'][::-1].copy()
            for k in ['observers','observed','east','north']:bank[k]=np.ascontiguousarray(bank[k][::-1])
        r,_,_=cpu_reference(bank);cost=np.array([np.sum(r[bank['case_index']==c]**2) for c in range(len(bank['states']))])
        for mismatch in [False,True]:
            out=dict(states=bank['states'].copy(),success=np.ones(len(cost),dtype=bool),cost=cost.copy())
            if mismatch:out['cost'][0]+=1e8
            expected=checked_cpu_arrays(bank,out,5,threads=2);actual=checked(bank,out,5,threads=2)
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_omitted_transport_seed_list_does_not_change_result(injected_catalogue):
    cat,sites,_=injected_catalogue;g=Geometry(cat,sites);g.prepare(range(6));key=[0,1,2]
    fit=fit_chain(g,key);assert fit is not None
    mode=fit['modes'][0];settings=dict(start_policy='provided',extra_seeds=[(np.array(mode['state_au_au_per_day']),mode['epoch_mjd_tdb'])])
    bank,plans,groups,epochs,bounds,limit=prepare_bank(g,[(key,settings)],compact_geometry=True)
    r,j,_=cpu_reference(bank);flags=np.ones(len(bank['states']),dtype=bool)
    args=(settings,groups[0],epochs,bounds,bank['states'],flags,r.ravel(),j.reshape(-1,6))
    expected=finish_group(plans[0],*args)
    actual=finish_group(replace(plans[0],seeds=()),*args)
    assert actual is not None and actual==expected
