"""Independent retrieval, precision, and deferred-work checks for fast growth."""
import json,sqlite3
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial import cKDTree
from magictools.itf.night_query import query_nights
from magictools.itf.nbody import Trajectory,fit_nbody,FitDeferred
from magictools.itf.longarc import prepare_screening,screen_extension,export
from test_solve_itf_nbody import CircularJupiter,independent_state
from magictools.itf.kernels import C


def test_packed_queries_equal_independent_spherical_trees():
    rng=np.random.default_rng(419)
    units=rng.normal(size=(4000,3));units/=np.linalg.norm(units,axis=1)[:,None]
    starts=np.arange(0,len(units),200);ends=starts+200
    for start,end in zip(starts,ends):units[start:end]=units[start:end][np.argsort(units[start:end,0])]
    nights=rng.integers(0,len(starts),500)
    predictions=rng.normal(size=(500,3));predictions/=np.linalg.norm(predictions,axis=1)[:,None]
    radii=rng.uniform(0,2,500)
    points,offsets,_=query_nights(units,starts,ends,nights,predictions,radii)
    for k,(night,position,radius) in enumerate(zip(nights,predictions,radii)):
        expected=cKDTree(units[starts[night]:ends[night]]).query_ball_point(position,radius)
        assert set(points[offsets==k])==set(np.asarray(expected,dtype=int)+starts[night])


def test_batched_screening_matches_individual_precise_predictions():
    ep=CircularJupiter();state=np.array([2.2,.2,.15,-.002,.0105,.001])
    times=np.array([365.,365.02,365.04,730.,730.02,730.04])
    observers=ep.earth(times)
    precise=Trajectory(state,0.,-1.,740.,ep)
    observed=precise.apparent(times,observers)[0]
    labels=np.repeat([1,2],3)
    class Geometry:
        def get(self,indices):
            use=np.isin(labels,indices)
            return times[use],observed[use],observers[use],labels[use]
    seed=dict(state_au_au_per_day=state,covariance_factor=np.diag([1e-5]*3+[1e-7]*3))
    fast=Trajectory(state,0.,-1.,740.,ep,rtol=2e-8,max_step=256.)
    prepare_screening(fast,Geometry(),{1,2})
    for candidate in (1,2,[1,2]):
        expected,chi2=screen_extension(seed,precise,Geometry(),candidate)
        actual,actual_chi2=screen_extension(seed,fast,Geometry(),candidate)
        np.testing.assert_allclose(actual,expected,atol=1e-8)
        assert abs(actual_chi2-chi2)<1e-6
    assert fast.screening_trajectory.rtol==2e-11
    assert fast.screening_trajectory.max_step==10.


def test_fast_initializer_preserves_year_link_and_defers_bad_pilot():
    ep=CircularJupiter();state=np.array([2.2,.2,.15,-.002,.0105,.001])
    times=np.array([-365.,-364.98,0.,.02,2.,2.02,5.,5.02,370.,370.02])
    observers=ep.earth(times);emitted=times.copy()
    for _ in range(6):
        vector=independent_state(state,emitted,ep)[:,:3]-observers
        emitted=times-np.linalg.norm(vector,axis=1)/C
    observed=vector/np.linalg.norm(vector,axis=1)[:,None];labels=np.repeat(np.arange(5),2)
    class Geometry:
        ephemeris=ep
        def get(self,indices):return times,observed,observers,labels
    initial=state+np.array([1e-4,-2e-4,1e-4,1e-6,-1e-6,1e-6])
    result=fit_nbody(Geometry(),list(range(5)),initial,0.,fast_start=True,pilot_gate_arcsec=3.)
    assert result is not None and result['rms_arcsec']<.01 and result['span_days']>730
    predicted=Trajectory(result['state_au_au_per_day'],result['epoch_mjd_tdb'],-366.,371.,ep).apparent(times,observers)[0]
    np.testing.assert_allclose(predicted,observed,atol=1e-8)
    observed[-2:,0]+=.01;observed[-2:]/=np.linalg.norm(observed[-2:],axis=1)[:,None]
    with pytest.raises(FitDeferred):
        fit_nbody(Geometry(),list(range(5)),initial,0.,fast_start=True,pilot_gate_arcsec=3.)


def test_export_distinguishes_deferred_fits_from_complete_and_retryable(tmp_path):
    db=sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE seeds (key TEXT,status TEXT,counts TEXT,attempts INTEGER); CREATE TABLE candidates (key TEXT,body TEXT);')
    db.executemany('INSERT INTO seeds VALUES (?,?,?,1)', [('a','searched','{}'),('b','searched_deferred_fits','{}')])
    result=export(db,tmp_path,SimpleNamespace(manifest={}),{},dict(eligible_seed_modes=2),'deferred_full_fits_remaining',1.)
    counts=result['counts']
    assert counts['seeds_completed']==1 and counts['seeds_first_pass_completed']==2
    assert counts['seeds_retryable']==0 and counts['seeds_with_deferred_fits']==1
    assert counts['first_pass_scan_complete'] and not counts['full_catalogue_scan_complete']


def test_deferred_replay_deduplicates_inputs_and_preserves_completed_jobs(tmp_path):
    from magictools.itf.replay_growth import import_jobs
    path=tmp_path/'deferred-fits';path.mkdir()
    job=dict(indices=[1,2,3,4],initial=[2.,0.,0.,0.,.01,0.],epoch=60000.,sigma=.3,seed_key='1:0')
    (path/'1.jsonl').write_text(json.dumps(job)+'\n'+json.dumps(dict(job,seed_key='2:0'))+'\n')
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE jobs (key TEXT PRIMARY KEY,body TEXT,status TEXT,attempts INTEGER,error TEXT)')
    import_jobs(db,[tmp_path])
    assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==1
    db.execute("UPDATE jobs SET status='accepted',attempts=1");db.commit()
    import_jobs(db,[tmp_path])
    assert db.execute('SELECT status,attempts FROM jobs').fetchone()==('accepted',1)
