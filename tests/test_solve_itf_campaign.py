from itertools import combinations
import numpy as np
from magictools.itf.campaign_graph import csr_graph, enumerate_triples, distinct_triples


def test_compact_lazy_wedges_match_exhaustive_with_crowding_and_aliases():
    rng=np.random.default_rng(42);n=80
    edges=np.array([(i,j,float(rng.random())) for i,j in combinations(range(n),2) if rng.random()<.15])
    offsets,neighbors,scores=csr_graph(edges,n)
    times=np.floor(rng.uniform(0,12,n));ids=np.arange(n)//2
    t,s,o,visited,capped=enumerate_triples(offsets,neighbors,scores,times,ids,0,n,5,100000)
    expected=[]
    for mid in range(n):
        links=list(zip(scores[offsets[mid]:offsets[mid+1]],neighbors[offsets[mid]:offsets[mid+1]]))
        pairs=[]
        for (sa,a),(sb,b) in combinations(links,2):
            if min(abs(times[a]-times[mid]),abs(times[b]-times[mid]),abs(times[a]-times[b]))<.3:continue
            if len({ids[a],ids[mid],ids[b]})<3:continue
            pairs.append((sa+sb,tuple(sorted((a,mid,b)))))
        expected.extend((triple,score,k) for k,(score,triple) in enumerate(sorted(pairs)[:5]))
    actual=sorted((tuple(a),float(b),int(c)) for a,b,c in zip(t,s,o))
    assert actual==sorted(expected)
    dt,ds,do=distinct_triples(t,s,o)
    assert len({tuple(x) for x in dt})==len(dt)
    for triple,score,ordinal in zip(dt,ds,do):
        options=[(int(k),float(v)) for x,v,k in zip(t,s,o) if np.array_equal(x,triple)]
        assert (ordinal,score)==min(options)


def test_visit_limit_and_empty_graph_are_explicit():
    e=np.array([[0,1,1.],[0,2,2.],[0,3,3.]])
    offsets,neighbors,scores=csr_graph(e,4)
    t,s,o,visited,capped=enumerate_triples(offsets,neighbors,scores,np.zeros(4),np.arange(4),0,4,10,1)
    assert len(t)==0 and visited==1 and capped==1
    offsets,neighbors,scores=csr_graph(np.empty((0,3)),4)
    t,s,o,visited,capped=enumerate_triples(offsets,neighbors,scores,np.arange(4.),np.arange(4),0,4,10,1)
    assert len(t)==visited==capped==0


def test_transactional_seed_checkpoint_counts_restart_and_idempotence(tmp_path):
    import zlib,json
    from magictools.itf.campaign_fit import open_checkpoint,commit_batch,checkpoint_counts
    ident={'source':'test','settings':{'x':1}}
    path=tmp_path/'fit.sqlite'
    db=open_checkpoint(path,ident)
    body=zlib.compress(b'{"tracklet_indices":[0,1,2]}')
    result=(0,0,4,bytes([1,2,3,0]),[(1,body)],[(2,'test','numerical')],1.)
    commit_batch(db,result);commit_batch(db,result)
    assert db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]==1
    assert checkpoint_counts(db,8)==dict(seed_target=8,seeds_attempted=3,seeds_completed=2,
        seeds_without_accepted_fit=1,accepted_seed_fits=1,seeds_retryable=1,seeds_unattempted=5)
    db.close();db=open_checkpoint(path,ident)
    # An interrupted replay must not erase terminal outcomes from the prior pass.
    commit_batch(db,(0,0,4,bytes([0,0,1,2]),[(3,body)],[],2.))
    assert checkpoint_counts(db,8)['seeds_completed']==4
    assert checkpoint_counts(db,8)['accepted_seed_fits']==2
    db.close()
    import pytest
    with pytest.raises(ValueError,match='changed'):open_checkpoint(path,{'source':'different'})


from test_solve_itf import injected_catalogue

def test_exact_shared_geometry_matches_fit_and_resumes(injected_catalogue,tmp_path):
    from magictools.itf.campaign_geometry import prepare_atlas,AtlasGeometry
    from magictools.itf.fit import Geometry,fit_chain
    c,sites,sitefile=injected_catalogue
    destination=tmp_path/'atlas'
    prepare_atlas(c.path,sitefile,destination,lambda _:None)
    shared=AtlasGeometry(c.path,sitefile,destination)
    reference=Geometry(c,sites);reference.prepare(range(len(c.tracks)))
    for actual,expected in zip(shared.get([0,2,5]),reference.get([0,2,5])):
        np.testing.assert_array_equal(actual,expected)
    expected=fit_chain(reference,[0,2,5],broad_starts=True)
    actual=fit_chain(shared,[0,2,5],broad_starts=True)
    assert expected['rms_arcsec']==actual['rms_arcsec']
    assert len(expected['modes'])==len(actual['modes'])
    # Simulate loss of the commit marker and one unfinished tracklet.
    (destination/'complete.json').unlink()
    ready=np.load(destination/'ready.npy',mmap_mode='r+');ready[2]=0;ready.flush()
    prepare_atlas(c.path,sitefile,destination,lambda _:None)
    resumed=AtlasGeometry(c.path,sitefile,destination)
    for actual,expected in zip(resumed.get([0,2,5]),reference.get([0,2,5])):
        np.testing.assert_array_equal(actual,expected)


def test_annual_parallax_lane_retrieves_independent_distant_motion():
    from scipy.integrate import solve_ivp
    from magictools.itf.catalogue import TRACK_DTYPE
    from magictools.itf.kernels import MU
    from magictools.itf.parallax_pairs import slow_pairs
    state=np.array([60.,20.,5.,-.001,.002,.0001])
    def rhs(t,y):return np.r_[y[3:],-MU*y[:3]/np.linalg.norm(y[:3])**3]
    sol=solve_ivp(rhs,[-1.,740.],state,method='DOP853',rtol=1e-12,atol=1e-13,dense_output=True).sol
    def earth(t):
        a=np.asarray(t)*2*np.pi/365.25
        return np.column_stack((np.cos(a),np.sin(a),np.zeros_like(a)))
    def earth_v(t):
        a=np.asarray(t)*2*np.pi/365.25;w=2*np.pi/365.25
        return np.column_stack((-w*np.sin(a),w*np.cos(a),np.zeros_like(a)))
    def observed(t):
        delta=sol(t)[:3].T-earth(t)
        return delta/np.linalg.norm(delta,axis=1)[:,None]
    times=np.array([.25,365.25,730.25]);tracks=np.zeros(3,dtype=TRACK_DTYPE)
    tracks['id']=[b'A',b'B',b'C'];tracks['count']=4;tracks['span']=.05
    tracks['t']=times;tracks['u']=observed(times)
    tracks['v']=(observed(times+.001)-observed(times-.001))/.002
    days=np.arange(-2.,733.)
    edges,counts=slow_pairs(tracks,earth(times),earth_v(times),days,earth(days),earth_v(days),
        gaps=(365,730),slack=1,ranges=(60.,),progress=lambda _:None)
    assert {(int(a),int(b)) for a,b,_ in edges}=={(0,1),(0,2),(1,2)}
    assert counts['slow_tracklets']==3


def test_stream_export_preserves_modes_and_distinguishes_pairs(injected_catalogue,tmp_path):
    import json,zlib
    from types import SimpleNamespace
    from magictools.itf.campaign_fit import open_checkpoint,commit_batch
    from magictools.itf.campaign_export import export
    from magictools.itf.campaign_graph import save_json
    from magictools.itf.fit import Geometry,fit_chain
    c,sites,sitefile=injected_catalogue
    output=tmp_path/'campaign';output.mkdir()
    row=fit_chain(Geometry(c,sites),[0,2,5],broad_starts=True)
    save_json(output/'seeds.json',dict(supplied=1,seed_sha256='test'))
    db=open_checkpoint(output/'fitting-full.sqlite',dict(seed_sha256='test',geometry={'input':c.manifest}))
    commit_batch(db,(0,0,1,bytes([2]),[(0,zlib.compress(json.dumps(row).encode()))],[],.1));db.close()
    export(SimpleNamespace(output=output,cache=c.path,start_policy='full'))
    report=json.loads((output/'report-full.json').read_text())
    actual=report['candidates'][0]
    assert actual['modes']==row['modes']
    assert actual['hypothesis_only'] is False and actual['tracklet_indices']==[0,2,5]
    assert report['counts']['seeds_completed']==1


def test_recovery_bank_preserves_full_modes_on_shared_scout_chains():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'research/solve_itf/make_recovery_bank.py'
    spec=importlib.util.spec_from_file_location('recovery_bank_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m1={'epoch_mjd_tdb':1.,'state_au_au_per_day':[1,0,0,0,1,0]}
    m2={'epoch_mjd_tdb':1.,'state_au_au_per_day':[2,0,0,0,1,0]}
    row=dict(tracklet_indices=[0,1,2],rank=1,maximal=True,modes=[m1])
    base=dict(input={'source':'same'},candidates=[row])
    other=dict(input=base['input'],candidates=[dict(row,modes=[m1,m2])])
    merged=module.combine(base,[(other,'digest')])
    assert merged['candidates'][0]['modes']==[m1,m2]
    assert merged['additional_modes_on_shared_chains']==1
    assert len(base['candidates'][0]['modes'])==1


def test_pair_budget_zero_retains_all_supplemental_pairs(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import magictools.itf.campaign_pairs as module
    root=tmp_path/'run';graph=root/'graph';graph.mkdir(parents=True)
    np.save(graph/'offsets.npy',np.array([0,1,2,3,4]))
    edges=np.array([[0,1,1.],[2,3,2.]])
    np.save(graph/'part-000.npy',edges);np.save(graph/'parallax.npy',np.array([[1,2,3.]]))
    (graph/'graph-identity.json').write_text('{}')
    tracks=np.zeros(4,dtype=[('count','i4')]);tracks['count']=3
    monkeypatch.setattr(module,'Catalogue',lambda _:SimpleNamespace(tracks=tracks))
    args=SimpleNamespace(output=root,cache=tmp_path,pair_seeds=0)
    directory=module.prepare(args)
    assert len(np.load(directory/'seeds.npy'))==3
    assert json.loads((directory/'seeds.json').read_text())['deferred_pairs']==0
    assert module.prepare(args)==directory
    import pytest
    with pytest.raises(ValueError,match='Pair budget changed'):
        module.prepare(SimpleNamespace(output=root,cache=tmp_path,pair_seeds=1))
