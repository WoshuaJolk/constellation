import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from ambitious_distant_graph import plan_frames,cap_edges,DISTANCES,LONG_DISTANCES,RADIAL
from ambitious_distant_seeds import prioritize
from benchmark_sparse_epoch_join import shared,nightly

def test_plan_complete_unique_and_both_windows_early():
    p=plan_frames(np.array([50000.,56000.]));assert p==plan_frames(np.array([50000.,56000.]))
    assert {r['gap'] for r in p[:2]}=={365.,1095.}
    tasks=[(r['gap'],r['epoch'],*h) for r in p for h in r['hypotheses']]
    assert len(tasks)==len(set(tasks))
    for gap,ds in [(365.,DISTANCES),(1095.,LONG_DISTANCES)]:
        for epoch in {r['epoch'] for r in p if r['gap']==gap}:
            assert {(x[2],x[3]) for x in tasks if x[:2]==(gap,epoch)}=={(d,f) for d in ds for f in RADIAL}

def test_three_year_join_parity():
    rng=np.random.default_rng(72);p=rng.normal(size=(90,3))*.001+np.array([1.,0.,0.]);p/=np.linalg.norm(p,axis=1)[:,None]
    t=np.repeat([0.,366.,730.,1096.,2000.],18);args=(p,t,np.ones(90),np.arange(90))
    a,_=nightly(*args,max_gap_days=1095);b,_=shared(*args,max_gap_days=1095)
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    assert np.max(np.abs(t[a[:,0].astype(int)]-t[a[:,1].astype(int)]))>365

def test_cap_preserves_bright_and_missing_in_ordinary():
    e=np.array([[0,1,.1],[1,2,.2],[2,3,.3],[4,5,9.],[5,6,10.]])
    m=np.array([np.nan,np.nan,24,24,15,15,15]);b=np.array([b'',b'',b'r',b'r',b'g',b'g',b'g'])
    r=cap_edges(e,4,m,b,100.);assert len(r)==4;assert (0,1) in set(map(tuple,r[:,:2]));assert (4,5) in set(map(tuple,r[:,:2]))
    np.testing.assert_array_equal(cap_edges(e,10,m,b,100),e)

def test_priority_retains_every_seed():
    a=np.array([[0,1,2],[1,2,3],[2,3,4]]);m=np.array([25,25,25,12,12.]);b=np.array([b'r']*5)
    r=prioritize(a,m,b,100);assert set(map(tuple,a))==set(map(tuple,r))

def test_additional_starts_preserve_existing(monkeypatch):
    import ambitious_distant_adapter as mod
    class G:
        def get(self,k):return np.arange(3.),np.ones((3,3)),np.zeros((3,3)),np.zeros(3)
    old=('old',);new=('new',)
    monkeypatch.setattr(mod,'distant_seeds',lambda *a,**k:([new],{'distances':[{'starts':1},{'starts':0}]}))
    result,r=mod.augment_jobs(G(),[((1,2,3),{'extra_seeds':(old,),'max_nfev':80})])
    assert result[0][1]['extra_seeds']==(old,new);assert result[0][1]['max_nfev']==80;assert r==dict(extra_distant_starts=1,unsupported_chord_distances=1)
