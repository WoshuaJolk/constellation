import time
import numpy as np
from magictools.itf.catalogue import TRACK_DTYPE
from magictools.itf.search import SearchConfig
from magictools.itf.broad import _curved_scan
from magictools.itf.sparse_curved import sparse_scan


def sparse_example(nights=12):
    rng=np.random.default_rng(91);n=nights*3
    tracks=np.zeros(n,TRACK_DTYPE);tracks['t']=60000+np.repeat(np.arange(nights),3)+rng.uniform(.1,.15,n)
    tracks['id']=[f'B{i:05d}'.encode() for i in range(n)];tracks['id'][4]=tracks['id'][0]
    tracks['u']=np.column_stack((np.ones(n),rng.normal(0,.002,n),rng.normal(0,.001,n)))
    tracks['u']/=np.linalg.norm(tracks['u'],axis=1)[:,None]
    tracks['v']=rng.normal(0,.0001,(n,3));tracks['count']=3;tracks['span']=.03
    days=np.arange(59998,60000+nights+3,dtype=float);earth=np.tile([0.,1.,0.],(len(days),1));ev=np.zeros_like(earth)
    positions=np.tile([0.,1.,0.],(n,1));velocities=np.zeros_like(positions)
    return tracks,positions,velocities,days,earth,ev


def test_sparse_distance_proposals_match_tree_reference_including_pool_caps():
    data=sparse_example()
    for cap in (100000,20):
        config=SearchConfig(max_gap_days=4,neighbors_per_day=2,max_edges=cap)
        expected,ec=_curved_scan(*data,config,np.inf)
        actual,ac=sparse_scan(*data,config,np.inf)
        expected=expected[np.lexsort((expected[:,1],expected[:,0]))]
        actual=actual[np.lexsort((actual[:,1],actual[:,0]))]
        np.testing.assert_array_equal(actual[:,:2],expected[:,:2])
        np.testing.assert_allclose(actual[:,2],expected[:,2],rtol=1e-10,atol=1e-10)
        assert ac==ec


def test_sparse_empty_and_partition_accounting():
    data=sparse_example();config=SearchConfig(max_gap_days=3)
    a,ac=sparse_scan(*data,config,np.inf,partition=(0,2))
    b,bc=_curved_scan(*data,config,np.inf,partition=(0,2))
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10);assert ac==bc


def test_capped_equal_distance_ties_fall_back_to_original_tree(monkeypatch):
    from magictools.itf import sparse_curved
    data=sparse_example();tracks=data[0]
    tracks['t']=60000+np.repeat(np.arange(12),3)+.1
    tracks['u']=[1.,0,0];tracks['v']=0
    calls=[]
    original=sparse_curved.reference_scan
    def reference(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(sparse_curved,'reference_scan',reference)
    config=SearchConfig(max_gap_days=3,neighbors_per_day=1)
    actual,ac=sparse_curved.sparse_scan(*data,config,np.inf)
    expected,ec=_curved_scan(*data,config,np.inf)
    assert calls
    np.testing.assert_array_equal(actual,expected);assert ac==ec


def test_larger_control_nights_preserve_tree_results_with_opt_in_limit():
    data=list(sparse_example(nights=36));tracks=data[0]
    tracks['t']=60000+np.repeat(np.arange(3),36)+np.tile(np.linspace(.1,.15,36),3)
    config=SearchConfig(max_gap_days=3,neighbors_per_day=8)
    expected,ec=_curved_scan(*data,config,np.inf)
    actual,ac=sparse_scan(*data,config,np.inf,max_sparse_night=256)
    expected=expected[np.lexsort((expected[:,1],expected[:,0]))]
    actual=actual[np.lexsort((actual[:,1],actual[:,0]))]
    np.testing.assert_array_equal(actual[:,:2],expected[:,:2])
    np.testing.assert_allclose(actual[:,2],expected[:,2],rtol=1e-10,atol=1e-10)
    assert ac==ec
