import numpy as np
import pytest
from numba import cuda
from test_itf_gpu import fixture_bank
from test_solve_itf import injected_catalogue
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research/solve_itf"))
from magictools.itf.gpu import ResidentBank
from magictools.itf.gpu_fit import ResidentWorkspace
from magictools.itf.gpu_indexed import ResidentGeometry


@pytest.mark.skipif(not cuda.is_available(),reason='CUDA or explicit simulator required')
def test_indexed_arrays_match_copied_arrays_and_lm():
    bank=fixture_bank();names=('observers','observed','east','north')
    index=np.arange(len(bank['offsets'])-1,-1,-1,dtype=np.int64)
    geometry=ResidentGeometry({k:np.ascontiguousarray(bank[k][::-1]) for k in names},allow_simulator=True)
    bank['observation_index']=index
    expected=ResidentBank(bank,allow_simulator=True).evaluate()
    actual=ResidentBank(bank,allow_simulator=True,resident_geometry=geometry).evaluate()
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    a=ResidentWorkspace(bank,allow_simulator=True,coalesced=True).solve(bank['states'],5,parallel_normal=True,parallel_copy=True)
    b=ResidentWorkspace(bank,allow_simulator=True,coalesced=True,resident_geometry=geometry).solve(bank['states'],5,parallel_normal=True,parallel_copy=True,completion_interval=16,register_normal=True)
    for k in ['states','success','cost','reasons','iterations','evaluations']:np.testing.assert_array_equal(a[k],b[k])
    assert geometry.bindings==2 and geometry.avoided_upload_bytes>0
    bank['observation_index']=np.roll(index,1)
    with pytest.raises(ValueError,match='differs'):
        ResidentBank(bank,allow_simulator=True,resident_geometry=geometry)


@pytest.mark.skipif(not cuda.is_available(),reason='CUDA or explicit simulator required')
def test_atlas_template_is_exact_for_selected_fit(injected_catalogue,tmp_path):
    from magictools.itf.campaign_geometry import prepare_atlas
    from magictools.itf.campaign_geometry_fast import FastAtlasGeometry,prepare_observation_ids
    from gpu_batch_host import prepare_bank
    c,sites,sitefile=injected_catalogue
    prepare_atlas(c.path,sitefile,tmp_path/'atlas',lambda _:None)
    prepare_observation_ids(c.path,tmp_path/'ids')
    g=FastAtlasGeometry(c.path,sitefile,tmp_path/'atlas',tmp_path/'ids')
    geometry=ResidentGeometry.from_atlas(g,allow_simulator=True)
    bank,*_=prepare_bank(g,[([0,2,5],dict(max_nfev=5))])
    a=ResidentBank(bank,allow_simulator=True).evaluate()
    b=ResidentBank(bank,allow_simulator=True,resident_geometry=geometry).evaluate()
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
