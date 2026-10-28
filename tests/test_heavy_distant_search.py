import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
import numpy as np
from heavy_distant_search import select_new_seeds

def test_frozen_seed_exclusion_quota_and_missing_photometry():
    keys=[(0,1,i) for i in range(2,30)];times=np.arange(30)*20.;mag=np.full(30,np.nan);mag[:2]=21.
    chosen,count=select_new_seeds(keys,[(0,1,2)],times,mag,14)
    assert (0,1,2) not in chosen
    assert len(chosen)==len(set(chosen))==14
    assert count['bright']==3 and count['long_gap']==3 and count['ordinary']==8
    assert chosen==select_new_seeds(keys,[(0,1,2)],times,mag,14)[0]
    chosen,count=select_new_seeds(keys,[],times,np.full(30,np.nan),14)
    assert count['bright']==0 and len(chosen)==14

def test_empty_and_exhausted_pool():
    assert select_new_seeds([],[],np.zeros(3),np.zeros(3),14)[0]==[]
    assert select_new_seeds([(0,1,2)],[(0,1,2)],np.zeros(3),np.zeros(3),14)[0]==[]

def test_exact_projection_against_independent_integrator():
    from distant_epoch_graph import projected
    from magictools.itf.kernels import MU,ARCSEC
    from scipy.integrate import solve_ivp
    for radius in (30.,100.,1000.):
        radial=.5*np.sqrt(2*MU/radius)
        u=np.array([[1.,0,0]]);du=np.array([[0,np.sqrt(MU/radius)/radius,0]])
        pos=np.zeros((1,3));vel=pos.copy();ref=365.;distance=radius+radial*ref
        actual=projected(np.array([0.]),u,du,pos,vel,ref,np.zeros(3),distance,radial)[0]
        state=np.array([radius,0,0,radial,np.sqrt(MU/radius),0])
        def rhs(t,x):return np.r_[x[3:],-MU*x[:3]/np.linalg.norm(x[:3])**3]
        integrated=solve_ivp(rhs,[0,ref],state,method='DOP853',rtol=2e-13,atol=1e-14).y[:3,-1]
        integrated/=np.linalg.norm(integrated)
        assert np.linalg.norm(np.cross(actual,integrated))*ARCSEC<1e-5

def test_epoch_windows_cover_year_pairs():
    anchors=np.arange(-360,1081,180)
    for start in np.arange(0,360,17.):
        for gap in (1,120,240,365):
            assert np.any((np.abs(anchors-start)<=365)&(np.abs(anchors-start-gap)<=365))
