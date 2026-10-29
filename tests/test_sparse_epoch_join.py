import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from benchmark_sparse_epoch_join import nightly,shared


def test_sparse_join_same_edges_scores_and_top16():
    rng=np.random.default_rng(329);p=rng.normal(size=(180,3));p[:80]=np.array([1.,0.,0.])+rng.normal(size=(80,3))*.002;p/=np.linalg.norm(p,axis=1)[:,None]
    t=np.repeat(np.array([0.,1.,30.,366.,731.,1097.]),30)+np.tile(np.arange(30)*.001,6)
    noise=np.ones(len(t));ids=np.arange(len(t));ids[30]=ids[0]
    a,_=nightly(p,t,noise,ids);b,r=shared(p,t,noise,ids)
    np.testing.assert_array_equal(a[:,:2],b[:,:2]);np.testing.assert_allclose(a[:,2],b[:,2],rtol=0,atol=1e-12)
    assert r['queries']==2*len(t)
    with pytest.raises(MemoryError):shared(p,t,noise,ids,max_raw_neighbors=1)


def test_no_later_nights_empty_graph():
    p=np.array([[1.,0,0],[1.,0,0]])
    a,_=nightly(p,np.zeros(2),np.ones(2),np.arange(2));b,_=shared(p,np.zeros(2),np.ones(2),np.arange(2))
    assert a.shape==b.shape==(0,3)


def test_strict_radius_boundary_and_top16_tie_policy():
    p=np.array([[1.,0.,0.],[np.cos(.0001),np.sin(.0001),0.]])
    r=float(np.linalg.norm(p[0]-p[1]));args=(p,np.array([0.,1.]),np.ones(2),np.arange(2))
    a,_=nightly(*args,radius=r);b,_=shared(*args,radius=r)
    np.testing.assert_array_equal(a,b)
    p=np.tile(np.array([[1.,0.,0.]]),(40,1));t=np.r_[0.,np.ones(39)]
    a,_=nightly(p,t,np.ones(40),np.arange(40));b,counts=shared(p,t,np.ones(40),np.arange(40))
    assert counts['tie_fallbacks']==1
    np.testing.assert_array_equal(a,b)
