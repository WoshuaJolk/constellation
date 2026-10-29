import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
import numpy as np
from magictools.itf.catalogue import TRACK_DTYPE
from magictools.itf.search import make_seeds,SearchConfig
from ranked_seed_stream import make_ranked_seeds

def test_exact_rank_and_ties_against_original():
    rng=np.random.default_rng(3245)
    for trial in range(30):
        n=25;t=np.zeros(n,TRACK_DTYPE);t['t']=rng.integers(0,20,n)+rng.random(n)/10;t['id']=[f'X{x//2}'.encode() for x in range(n)]
        a,b=np.triu_indices(n,1);keep=rng.random(len(a))<.25;e=np.column_stack((a[keep],b[keep],rng.integers(0,4,keep.sum()).astype(float)))
        for k in (1,7,100,10000):
            expected,adj=make_seeds(e,t,SearchConfig(max_seeds=k,seed_neighbors=0),{},time.monotonic()+60)
            actual,other,counts=make_ranked_seeds(e,t,k)
            assert actual==expected
            assert all(sorted(other[i])==sorted(v) for i,v in adj.items())

def test_empty_edges():
    t=np.zeros(2,TRACK_DTYPE);keys,adj,c=make_ranked_seeds(np.empty((0,3)),t,10)
    assert keys==[] and adj=={} and c['seed_stream_exhausted']
