import sys
from pathlib import Path
import numpy as np
import pytest
from numba import cuda
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from gpu_night_query import NightQueryBank
from magictools.itf.night_query import query_nights

pytestmark = pytest.mark.skipif(not cuda.is_available(),reason='CUDA hardware or explicit simulator required')


def test_exact_hits_order_empty_boundary_duplicates_and_overflow():
    rng=np.random.default_rng(29)
    p=rng.normal(size=(131,3));p/=np.linalg.norm(p,axis=1)[:,None]
    p[1]=p[0]; starts=np.array([0,51,51],np.int64);ends=np.array([51,51,131],np.int64)
    for lo,hi in zip(starts,ends):p[lo:hi]=p[lo:hi][np.argsort(p[lo:hi,0],kind='stable')]
    nights=np.array([0,1,2,0,2,0],np.int64)
    predictions=np.ascontiguousarray(p[[0,0,51,1,52,3]])
    radii=np.array([0.,2.,0.,2.,.5,np.linalg.norm(p[3]-p[4])])
    bank=NightQueryBank(p,starts,ends,allow_simulator=True)
    got,receipt=bank.query(nights,predictions,radii)
    expected=query_nights(p,starts,ends,nights,predictions,radii)
    for a,b in zip(got[:2],expected[:2]):np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(got[2],expected[2],rtol=0,atol=1e-15)
    assert receipt['hits']==len(got[0])
    empty,_=bank.query(nights[:0],predictions[:0],radii[:0]);assert not len(empty[0])
    with pytest.raises(MemoryError):bank.query(nights,predictions,radii,max_output_bytes=1)
    with pytest.raises(ValueError):bank.query(np.array([-1],np.int64),predictions[:1],radii[:1])
    with pytest.raises(ValueError):bank.query(nights,predictions.astype('f4'),radii)


def test_no_nearest_neighbor_limit():
    p=np.tile(np.array([[1.,0,0]]),(300,1))
    bank=NightQueryBank(p,np.array([0]),np.array([300]),allow_simulator=True)
    result,receipt=bank.query(np.array([0]),p[:1],np.array([0.]))
    assert receipt['hits']==300
    np.testing.assert_array_equal(result[0],np.arange(300))
