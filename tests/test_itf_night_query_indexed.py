import numpy as np
import pytest
from magictools.itf.night_query import query_nights
from magictools.itf.night_query_indexed import build_axis_orders,query_indexed_nights


def test_exact_query_parity_clustered_poles_wrap_and_boundary():
    rng=np.random.default_rng(2513)
    points=rng.normal(size=(300,3));points/=np.linalg.norm(points,axis=1)[:,None]
    points=np.vstack([points,[[1.,0,0],[1.,0,0],[0,1.,0],[0,0,1.],[-1.,0,0]]])
    parts=[points[:100],points[100:200],points[200:]]
    units=np.concatenate([p[np.argsort(p[:,0],kind='stable')] for p in parts])
    starts=np.array([0,100,200]);ends=np.array([100,200,len(units)])
    orders=build_axis_orders(units,starts,ends)
    nights=np.repeat(np.arange(3),20)
    pred=rng.normal(size=(60,3));pred/=np.linalg.norm(pred,axis=1)[:,None]
    pred[-5:]=[[1.,0,0],[1.,0,0],[0,1.,0],[0,0,1.],[-1.,0,0]]
    radii=np.tile([0.,1e-10,.01,.1,1.,2.,3.,.2,.3,1.5],6)
    for a,b in zip(query_nights(units,starts,ends,nights,pred,radii),query_indexed_nights(units,orders,starts,ends,nights,pred,radii)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='x-sorted'):
        build_axis_orders(units[::-1].copy(),starts,ends)
