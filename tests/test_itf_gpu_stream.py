import sys
from pathlib import Path
import numpy as np
import pytest
from numba import cuda
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from test_solve_itf import injected_catalogue
from test_itf_gpu import fixture_bank
from magictools.itf.fit import Geometry
from magictools.itf.gpu import ResidentBank, validate_bank
from magictools.itf.gpu_fit import ResidentWorkspace
from gpu_batch_host import prepare_bank
from gpu_stream_adapter import DeviceBufferPool, store_prepared, load_prepared
from gpu_pipeline_adapter import checked_cpu_arrays


def test_compact_preparation_exact(injected_catalogue,tmp_path):
    cat,sites,_=injected_catalogue;g=Geometry(cat,sites);g.prepare(range(6))
    jobs=[([0,1,2],{}),([1,2,3],{'broad_starts':True})]
    old=prepare_bank(g,jobs);new=prepare_bank(g,jobs,compact_geometry=True)
    a,b=old[0],new[0]
    for k in ['states','offsets','case_index','sigma']:
        np.testing.assert_array_equal(a[k].view(np.uint64),b[k].view(np.uint64))
    for k in ['observers','observed','east','north']:
        np.testing.assert_array_equal(a[k].view(np.uint64),b[k][b['geometry_index']].view(np.uint64))
        assert b[k].nbytes<a[k].nbytes
    validate_bank(b)
    from magictools.itf.gpu import cpu_reference
    for x,y in zip(cpu_reference(a),cpu_reference(b)):np.testing.assert_array_equal(x,y)
    receipt=store_prepared(new,tmp_path);loaded=load_prepared(receipt)
    for k in b:
        np.testing.assert_array_equal(b[k],loaded[0][k])
        assert b[k].shape==loaded[0][k].shape
    assert loaded[1][0].indices==new[1][0].indices
    assert not loaded[0]['states'].flags.writeable
    assert len(receipt['metadata'])<receipt['bytes']
    broken=dict(b,geometry_index=b['geometry_index'].copy());broken['geometry_index'][0]=-1
    with pytest.raises(ValueError,match='compact'):validate_bank(broken)


@pytest.mark.skipif(not cuda.is_available(),reason='CUDA or simulator required')
def test_compact_reused_workspace_matches_fresh_and_cpu_checks():
    original=fixture_bank();pool=DeviceBufferPool()
    for scale in [1.,1.001,1.]:
        bank={k:np.copy(v) for k,v in original.items()}
        bank['states'][:,0]*=scale
        compact=dict(bank,geometry_index=np.arange(len(bank['offsets'])-1,-1,-1,dtype=np.int64))
        for k in ['observers','observed','east','north']:compact[k]=np.ascontiguousarray(bank[k][::-1])
        expected=ResidentWorkspace(bank,allow_simulator=True,coalesced=True).solve(bank['states'],5,parallel_normal=True,parallel_copy=True)
        pool.begin();w=ResidentWorkspace(compact,allow_simulator=True,coalesced=True,allocator=pool)
        actual=w.solve(compact['states'],5,parallel_normal=True,parallel_copy=True)
        for k in ['states','cost','success','reasons','iterations','evaluations']:
            np.testing.assert_array_equal(actual[k],expected[k])
        a=checked_cpu_arrays(bank,expected,5,threads=2)
        b=checked_cpu_arrays(compact,actual,5,threads=2)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    assert pool.reuses>0


@pytest.mark.skipif(not cuda.is_available(),reason='CUDA or simulator required')
def test_pool_changed_shapes_and_reset():
    pool=DeviceBufferPool()
    for shape in [(4,6),(2,6),(8,6),(1,6)]:
        pool.begin();a=pool.empty(shape,np.float64,order='F')
        x=np.asfortranarray(np.arange(np.prod(shape),dtype=float).reshape(shape))
        a.copy_to_device(x);np.testing.assert_array_equal(a.copy_to_host(),x)
    assert pool.allocations==2 and pool.reuses==2


def test_stream_order_overlap_and_cleanup(monkeypatch,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import gpu_stream_adapter as module
    started=Event();second=Event();seen=[]
    bank=fixture_bank(); prepared=(bank,[],[],None,None,1)
    monkeypatch.setattr(module,'_prepare',lambda jobs:(store_prepared(prepared,tmp_path),0.))
    class Workspace:
        setup_seconds=0
        def __init__(self,*a,**k):pass
        def solve(self,*a,**k):
            seen.append('gpu')
            if seen.count('gpu')==2:
                assert started.wait(2);second.set()
            return {'profile':{}}
    monkeypatch.setattr(module,'ResidentWorkspace',Workspace)
    class Adapter(module.StreamGPUBatch):
        prepare_task=staticmethod(module._prepare)
        def start(self,g):
            if self.pool:return
            self.pool=ThreadPoolExecutor(max_workers=2)
            self.finish_pool=ThreadPoolExecutor(max_workers=1)
        def _finish(self,p,j,o,r,*args):
            try:
                if j==[0]:started.set();assert second.wait(2)
                return j,{'finished':j}
            finally:Path(r['path']).unlink()
    with Adapter() as adapter:
        answer=list(adapter.iter_batches(None,iter([[0],[1],[2]])))
        assert answer==[([0],[0]),([1],[1]),([2],[2])]
        assert adapter.pipeline['overlapped_batches']==2
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not cuda.is_available(),reason='CUDA or simulator required')
def test_pool_three_dimensional_fortran_views():
    pool=DeviceBufferPool()
    for shape in [(4,6,6),(2,6,6)]:
        pool.begin();a=pool.empty(shape,np.float64,order='F')
        x=np.asfortranarray(np.arange(np.prod(shape),dtype=float).reshape(shape))
        a.copy_to_device(x);np.testing.assert_array_equal(a.copy_to_host(),x)
    assert pool.allocations==1 and pool.reuses==1


def test_shared_bank_uses_one_mapping_for_many_arrays(tmp_path):
    import os,gc
    arrays=[np.arange(i%7+1,dtype=np.float64) for i in range(4096)]
    receipt=store_prepared(arrays,tmp_path)
    before=len(os.listdir('/proc/self/fd'))
    loaded=load_prepared(receipt)
    after=len(os.listdir('/proc/self/fd'))
    assert after-before<=2
    for a,b in zip(arrays,loaded):
        np.testing.assert_array_equal(a,b)
        assert not b.flags.writeable and b.flags.aligned
    Path(receipt['path']).unlink()
    assert loaded[-1][0]==0 # Mapping survives unlink until last view released.
    del loaded,b;gc.collect()
    assert len(os.listdir('/proc/self/fd'))<=before
