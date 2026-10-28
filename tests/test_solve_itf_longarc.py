from types import SimpleNamespace

import numpy as np

from magictools.itf.catalogue import TRACK_DTYPE
from magictools.itf.longarc import NightIndex, screen_extension
from magictools.itf.nbody import Trajectory
from test_solve_itf_nbody import CircularJupiter


def test_full_night_index_retrieves_both_year_directions_and_exposes_caps():
    ephemeris = CircularJupiter()
    state = np.array([2.2, .2, .15, -.002, .0105, .001])
    trajectory = Trajectory(state, 0, -430, 750, ephemeris)
    times = np.array([-365.1, .1, 365.1, 740.1])
    position = trajectory.apparent(times, ephemeris.earth(times))[0]
    dt = .01
    rate = (trajectory.apparent(times+dt, ephemeris.earth(times+dt))[0] -
            trajectory.apparent(times-dt, ephemeris.earth(times-dt))[0])/(2*dt)
    tracks = np.zeros(5, dtype=TRACK_DTYPE)
    tracks['id'] = [b'BEFORE', b'SEED', b'AFTER', b'AFTER2', b'UNRELATED']
    tracks['site'] = b'500'
    tracks['count'] = 2
    tracks['t'][:4], tracks['u'][:4], tracks['v'][:4] = times, position, rate
    tracks['t'][4], tracks['u'][4] = times[2], -position[2]
    index = NightIndex(SimpleNamespace(tracks=tracks), {b'500': None}, ephemeris)
    fitted = dict(state_au_au_per_day=state, epoch_mjd_tdb=0.,
                  covariance_factor=np.diag([1e-7]*3+[1e-9]*3), tracklet_indices=[1])
    proposals, counts = index.proposals(fitted, ephemeris, minimum_gap=300)
    assert set(proposals) == {0, 2, 3}
    assert index.counts['indexed_tracklets'] == 5
    assert counts['raw_proposals'] == 3
    limited, counts = index.proposals(fitted, ephemeris, minimum_gap=300, max_candidates=1)
    assert len(limited) == 1 and counts['proposals_discarded_at_cap'] == 2
    bounded_index = NightIndex(SimpleNamespace(tracks=tracks), {b'500': None}, ephemeris)
    bounded, counts = bounded_index.proposals(fitted, ephemeris, minimum_gap=100, maximum_gap=500)
    assert set(bounded) == {0, 2}
    assert counts['outside_time_window_nights'] == 1
    assert bounded_index.last_trajectory.end < 500
    fitted['covariance_factor'] = np.eye(6)
    proposals, counts = index.proposals(fitted, ephemeris, minimum_gap=300)
    assert proposals == [] and counts['uncertainty_gated_nights'] == 3
    assert counts['excluded_window_uncertainty_gates'] == 1


def test_joint_innovation_preserves_truth_and_rejects_wrong_motion():
    ephemeris = CircularJupiter()
    state = np.array([2.2, .2, .15, -.002, .0105, .001])
    trajectory = Trajectory(state, 0, -10, 380, ephemeris)
    times = np.array([365., 365.02, 365.04])
    observers = ephemeris.earth(times)
    observed = trajectory.apparent(times, observers)[0]

    class Geometry:
        def get(self, indices):
            return times, observed, observers, np.zeros(3)

    seed = dict(state_au_au_per_day=state, covariance_factor=np.diag([1e-7]*3+[1e-9]*3))
    initial, chi2 = screen_extension(seed, trajectory, Geometry(), 0)
    np.testing.assert_allclose(initial, state, atol=1e-14)
    assert chi2 < 1e-14
    observed[-1, 0] += .001
    observed[-1] /= np.linalg.norm(observed[-1])
    initial, chi2 = screen_extension(seed, trajectory, Geometry(), 0)
    assert initial is None and chi2 > 150


def test_whole_arc_proposals_preserve_alternatives_and_skip_contained_sets():
    from magictools.itf.longarc import arc_lookup, proposed_arcs
    lookup = arc_lookup([[1, 2, 3], [4, 5, 6], [4, 7, 8], [2, 4, 5], [4, 5, 6]])
    proposals, count = proposed_arcs([4, 5, 7], [1, 2, 3], lookup, limit=10)
    assert set(proposals) == {(4, 5, 6), (4, 7, 8), (4, 5)}
    assert count == 3
    limited, count = proposed_arcs([4, 5, 7], [1, 2, 3], lookup, limit=1)
    assert limited == [(4, 5, 6)] and count == 3


def test_short_gap_whole_arc_growth_refits_all_observations(monkeypatch):
    from magictools.itf import longarc
    from magictools.itf.kernels import C
    from test_solve_itf_nbody import independent_state

    ephem = CircularJupiter()
    state = np.array([2.2, .2, .15, -.002, .0105, .001])
    times = (np.array([0., 2., 5., 60., 62., 65., 60., 62., 65.])[:, None]
             + np.array([0., .02, .04])).ravel()
    labels = np.repeat(np.arange(9), 3)
    observers = ephem.earth(times)
    emitted = times.copy()
    for _ in range(6):
        position = independent_state(state, emitted, ephem)[:, :3] - observers
        emitted = times - np.linalg.norm(position, axis=1)/C
    observed = position / np.linalg.norm(position, axis=1)[:, None]
    # A competing arc shares a plausible anchor but has incompatible later motion.
    observed[labels == 7, 0] += .001
    observed /= np.linalg.norm(observed, axis=1)[:, None]

    class Geometry:
        ephemeris = ephem

        def get(self, indices):
            selected = np.isin(labels, indices)
            return times[selected], observed[selected], observers[selected], labels[selected]

    class Index:
        utc = times
        arc_lookup = longarc.arc_lookup([[3, 4, 5], [6, 7, 8]])
        last_trajectory = Trajectory(state, 0., -65., 70., ephem)

        def proposals(self, fitted, ephemeris, **kwargs):
            return [3, 6], {}

    # Only metadata serialization is replaced; prediction, screening and fits are real.
    monkeypatch.setattr(longarc, 'candidate_row', lambda fitted, *args: dict(fitted))
    args = SimpleNamespace(sigma_arcsec=.3, min_gap_days=1., max_gap_days=120.,
                           max_cone_deg=2., max_candidates=32)
    mode = dict(state_au_au_per_day=state.tolist(), epoch_mjd_tdb=0.)
    seed = (0., -5., 1, 0, dict(tracklet_indices=[0, 1, 2]), mode)
    _, status, counts, accepted = longarc.recover_seed(
        seed, Geometry(), Index(), ephem, args, np.inf)
    assert status == 'searched'
    merged = [row for row in accepted if row.get('extension_kind') == 'whole_arc_joint_fit']
    assert len(merged) == 1
    assert merged[0]['tracklet_indices'] == [0, 1, 2, 3, 4, 5]
    assert merged[0]['observations'] == 18 and merged[0]['rms_arcsec'] < .01
    assert counts['arc_merge_innovation_rejections'] == 1


def test_motion_ranking_recovers_match_behind_positional_interlopers():
    ephem = CircularJupiter()
    state = np.array([2.2, .2, .15, -.002, .0105, .001])
    orbit = Trajectory(state, 0., -1., 370., ephem)
    t = 365.5
    position = orbit.apparent([t], ephem.earth([t]))[0][0]
    velocity = (orbit.apparent([t+.01], ephem.earth([t+.01]))[0][0] -
                orbit.apparent([t-.01], ephem.earth([t-.01]))[0][0])/.02
    tracks = np.zeros(10, dtype=TRACK_DTYPE)
    tracks['site'] = b'500'; tracks['count'] = 3; tracks['span'] = .04
    tracks['t'] = t; tracks['u'] = position; tracks['v'] = velocity
    east = np.array([-position[1], position[0], 0.]); east /= np.linalg.norm(east)
    tracks['v'][1:9] += .05*east
    tracks['u'][9] += 40/206264.806*east
    tracks['u'] /= np.linalg.norm(tracks['u'],axis=1)[:,None]
    index = NightIndex(SimpleNamespace(tracks=tracks), {b'500':None}, ephem)
    fitted = dict(state_au_au_per_day=state, epoch_mjd_tdb=0.,
                  covariance_factor=np.diag([1e-7]*3+[1e-9]*3),tracklet_indices=[0])
    old, _ = index.proposals(fitted,ephem,minimum_gap=1,max_candidates=2)
    new, counts = index.proposals(fitted,ephem,minimum_gap=1,max_candidates=2,
                                  retrieval='motion',position_reserve=1)
    assert 9 not in old and 9 in new
    assert new[0] == old[0] and counts['proposals_discarded_at_cap'] == 7


def test_export_does_not_count_failed_or_interrupted_modes_as_complete(tmp_path):
    import sqlite3,json
    from magictools.itf.longarc import export
    db=sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE seeds (key TEXT,status TEXT,counts TEXT,attempts INTEGER); '
                     'CREATE TABLE candidates (key TEXT,body TEXT);')
    statuses=['searched','seed_refit_rejected','numerical_failure','partial_failure','seed_budget_exhausted','budget_expired']
    db.executemany('INSERT INTO seeds VALUES (?,?,?,?)',[(str(k),v,'{}',1) for k,v in enumerate(statuses)])
    report=export(db,tmp_path,SimpleNamespace(manifest={}),{},dict(eligible_seed_modes=7),'running',1.)
    assert report['counts']['seeds_completed']==2
    assert report['counts']['seeds_searched']==1
    assert report['counts']['seeds_retryable']==4
    assert report['counts']['seeds_unattempted']==1
    assert report['counts']['seeds_remaining']==5
    assert report['counts']['full_catalogue_scan_complete'] is False


def test_iterative_growth_requeries_refitted_state_and_covariance(monkeypatch):
    from magictools.itf import longarc
    from magictools.itf.kernels import C
    from test_solve_itf_nbody import independent_state
    ephem = CircularJupiter()
    state = np.array([2.2,.2,.15,-.002,.0105,.001])
    times = (np.array([0.,2.,5.,60.,400.])[:,None]+np.array([0.,.02,.04])).ravel()
    observers=ephem.earth(times); emitted=times.copy(); labels=np.repeat(np.arange(5),3)
    for _ in range(6):
        ray=independent_state(state,emitted,ephem)[:,:3]-observers
        emitted=times-np.linalg.norm(ray,axis=1)/C
    observed=ray/np.linalg.norm(ray,axis=1)[:,None]
    class Geometry:
        ephemeris=ephem
        def get(self,indices):
            use=np.isin(labels,indices)
            return times[use],observed[use],observers[use],labels[use]
    class Index:
        utc=times; arc_lookup={}; seen=[]
        last_trajectory=Trajectory(state,0.,-1.,410.,ephem)
        def proposals(self,fitted,ephemeris,**kwargs):
            self.seen.append(fitted)
            # Controlled retrieval frontier: the next detection is available only
            # after the intermediate arc is fitted; all orbit fits remain real.
            return ([3] if len(fitted['tracklet_indices'])==3 else [4]),{}
    monkeypatch.setattr(longarc,'candidate_row',lambda fitted,*args:dict(fitted))
    args=SimpleNamespace(sigma_arcsec=.3,min_gap_days=1.,max_gap_days=0.,max_cone_deg=2.,
                         max_candidates=32,iterations=2,seed_seconds=15.,fit_seconds=5.)
    seed=(0.,-5.,1,0,dict(tracklet_indices=[0,1,2]),dict(state_au_au_per_day=state.tolist(),epoch_mjd_tdb=0.))
    index=Index();_,status,counts,accepted=longarc.recover_seed(seed,Geometry(),index,ephem,args,np.inf)
    assert status=='searched' and counts['requery_hypotheses']==1
    assert len(index.seen)==2 and len(index.seen[1]['tracklet_indices'])==4
    assert np.linalg.norm(index.seen[1]['covariance_factor']) < np.linalg.norm(index.seen[0]['covariance_factor'])
    assert any(row['tracklet_indices']==list(range(5)) and row['rms_arcsec']<.01 for row in accepted)


def test_weak_modes_are_explicit_opt_in_not_silently_dropped():
    from magictools.itf.longarc import seed_bank
    report={'candidates':[{'maximal':True,'rank':1,'span_days':3.,'modes':[
        {'range_au':10.,'local_uncertainty':{'range_sigma_au':2.,'local_linearization_weak':True}},
        {'range_au':10.,'local_uncertainty':{'range_sigma_au':.1,'local_linearization_weak':False}}]}]}
    assert len(seed_bank(report))==1
    assert len(seed_bank(report,.3,allow_weak=True))==2
    assert len(seed_bank(report,.05,allow_weak=True))==1


def test_direct_arc_lane_survives_tracklet_cap_and_clears_empty_windows():
    from magictools.itf.longarc import attach_arcs
    ephem=CircularJupiter();state=np.array([2.2,.2,.15,-.002,.0105,.001])
    orbit=Trajectory(state,0.,-1.,370.,ephem)
    times=np.array([365.5]*5);positions=orbit.apparent(times,ephem.earth(times))[0]
    tracks=np.zeros(5,dtype=TRACK_DTYPE);tracks['site']=b'500';tracks['count']=3
    tracks['t']=times;tracks['u']=positions;tracks['span']=.04
    c=SimpleNamespace(tracks=tracks)
    index=NightIndex(c,{b'500':None},ephem);attach_arcs(index,[[3,4]],c)
    fitted=dict(state_au_au_per_day=state,epoch_mjd_tdb=0.,covariance_factor=np.diag([1e-7]*3+[1e-9]*3),tracklet_indices=[])
    proposals,_=index.proposals(fitted,ephem,max_candidates=1)
    assert proposals==[0] and index.arc_candidates==[3,4]
    index.proposals(fitted,ephem,maximum_gap=10)
    assert index.arc_candidates==[]


def test_bridge_first_requeries_improved_child_before_parent_far_window(monkeypatch):
    from magictools.itf import longarc
    calls=[]
    state=np.array([2.2,.2,.15,-.002,.0105,.001])
    class Index:
        last_trajectory=None
        arc_lookup={}
        def proposals(self,fitted,ephemeris,**kwargs):
            ids=tuple(fitted['tracklet_indices']);calls.append((ids,kwargs['minimum_gap']))
            return ([3] if ids==(0,1,2) and kwargs['minimum_gap']==0 else []),{}
    def fit(geometry,ids,initial,epoch,**kwargs):
        return dict(tracklet_indices=list(ids),state_au_au_per_day=state,epoch_mjd_tdb=0.,
                    covariance_factor=np.eye(6),span_days=float(len(ids)),chi2=0.)
    monkeypatch.setattr(longarc,'fit_nbody',fit)
    monkeypatch.setattr(longarc,'screen_extension',lambda *args:(state,0.))
    monkeypatch.setattr(longarc,'candidate_row',lambda fitted,*args:dict(fitted))
    args=SimpleNamespace(sigma_arcsec=.3,staged=True,bridge_first=True,iterations=2,branches=2,
        max_cone_deg=2.,max_candidates=8,min_gap_days=365.,max_gap_days=0.)
    row={'tracklet_indices':[0,1,2]};mode={'state_au_au_per_day':state,'epoch_mjd_tdb':0.}
    result=longarc.recover_seed((.01,-3,1,0,row,mode),object(),Index(),object(),args,np.inf)
    assert result[1]=='searched'
    assert calls.index(((0,1,2,3),0.)) < calls.index(((0,1,2),14.))
    assert ((0,1,2),730.) in calls


def test_nonlinear_probe_lane_retrieves_beyond_central_gate():
    from magictools.itf.recovery_explore import retrieve
    base=dict(state_au_au_per_day=np.zeros(6),covariance_factor=np.eye(6))
    class Index:
        arc_candidates=[]
        last_trajectory='central'
        def proposals(self,trial,ephemeris,**kwargs):
            shifted=np.any(trial['state_au_au_per_day'])
            self.last_trajectory='probe' if shifted else 'central'
            return ([9] if shifted else [1]),dict(uncertainty_gated_nights=1,raw_proposals=1)
    args=SimpleNamespace(max_cone_deg=2.,max_candidates=2,uncertainty_samples=2)
    index=Index()
    ids,counts,hints,arcs=retrieve(index,base,None,args,0.,30.,np.inf)
    assert ids==[1,9] and counts['uncertainty_probes_attempted']==2
    assert hints[9][1]=='probe' and hints[1][0] is base
    assert index.last_trajectory=='central'


def test_recovery_retains_higher_chi2_alternative_modes_on_same_chain():
    from magictools.itf.longarc import merge_candidate_rows
    mode1=dict(epoch_mjd_tdb=60000.,state_au_au_per_day=[2,0,0,0,.01,0],chi2=1.,conic='elliptic')
    mode2=dict(epoch_mjd_tdb=60000.,state_au_au_per_day=[2,0,0,0,.05,0],chi2=2.,conic='hyperbolic')
    a=dict(tracklet_indices=[0,1,2,3],chi2=1.,rms_arcsec=.1,modes=[mode1])
    b=dict(tracklet_indices=[0,1,2,3],chi2=2.,rms_arcsec=.2,modes=[mode2])
    result=merge_candidate_rows(a,b)
    assert result['chi2']==1. and result['rms_arcsec']==.1
    assert result['modes']==[mode1,mode2]
    assert merge_candidate_rows(result,b)['modes']==result['modes']
    assert a['modes']==[mode1] and b['modes']==[mode2]
    assert 'not a normalized' in result['mode_weights_kind']
