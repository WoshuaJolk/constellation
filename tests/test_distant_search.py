import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
import numpy as np
from magictools.itf.catalogue import TRACK_DTYPE
from magictools.itf.kernels import ARCSEC
from distant_search import distant_eligible
from distant_followup import lane_tags,select

def test_rate_gate_and_invalid_tracklets():
    t=np.zeros(5,TRACK_DTYPE);t['count']=3;t['span']=.1;t['v'][:,0]=[0,599,601,20,20]/np.array(ARCSEC)
    t['span'][3]=0;t['count'][4]=1
    assert distant_eligible(t).tolist()==[True,True,False,False,False]

def test_other_conics_retained_and_duplicate_selection():
    mode=dict(range_au=50.,perihelion_au=.5,semimajor_axis_au=-20.,eccentricity=2.,inclination_deg=150.,rms_arcsec=.2,epoch_mjd_tdb=60000.,state_au_au_per_day=[1,2,3,4,5,6])
    assert set(lane_tags(mode))=={'distant','NEO_mode','retrograde','hyperbolic_mode'}
    row=dict(tracklet_indices=[1,2,3],modes=[mode],span_days=5.)
    chosen,stats=select([row]);assert len(chosen)==1;assert len(chosen[0]['roles'])==5

def test_partition_matches_existing_projection_proposals(tmp_path):
    from distant_search import scan_partition,DISTANCES
    from magictools.itf.broad import _curved_scan
    from magictools.itf.search import SearchConfig
    import time
    t=np.zeros(6,TRACK_DTYPE);t['count']=3;t['span']=.03;t['t']=60000+np.repeat([0.,2.,5.],2);t['id']=[f'X{i}'.encode() for i in range(6)]
    angles=np.tile([0.,.01],3)+np.repeat([0.,-.0001,-.00025],2)
    t['u'][:,0]=np.cos(angles);t['u'][:,1]=np.sin(angles);t['v'][:,1]=-.00005
    days=np.arange(59998,60130,dtype=float);earth=np.column_stack((np.ones(len(days)),np.zeros((len(days),2))));vel=np.zeros_like(earth)
    arrays=(t,np.tile([1.,0,0],(6,1)),np.zeros((6,3)),days,earth,vel)
    for name,a in zip(('tracks','positions','velocities','days','earth','earth_velocity'),arrays):np.save(tmp_path/(name+'.npy'),a)
    r=scan_partition((str(tmp_path),0,1,30));assert r['complete']
    cfg=SearchConfig(max_gap_days=120,neighbors_per_day=16,max_edges=150000,curvature_gate_arcsec_per_day2=.01)
    expected,_=_curved_scan(*arrays,cfg,time.monotonic()+30,hypotheses=[(d,0.,120) for d in DISTANCES])
    np.testing.assert_array_equal(np.load(tmp_path/'part-0.npy'),expected)
