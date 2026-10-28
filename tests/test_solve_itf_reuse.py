import json,sqlite3
from types import SimpleNamespace
import numpy as np
import pytest
from magictools.itf.reuse import ObserverCache,NightPredictionCache,ExactFitCache,fit_key
from magictools.itf.deferred_queue import import_incremental,pending_jobs,WarmBank,key_for


def test_observers_batch_misses_and_bound_memory():
    calls=[]
    def earth(t):calls.append(t.copy());return np.column_stack((t,t*2,t*3))
    cache=ObserverCache(4);ep=SimpleNamespace(earth=earth)
    np.testing.assert_array_equal(cache.earth(ep,[1,2,1]),[[1,2,3],[2,4,6],[1,2,3]])
    cache.earth(ep,[2,3]);assert calls[-1].tolist()==[3]
    cache.earth(ep,[4,5,6]);assert len(cache.values)==4


def test_prediction_cache_preserves_distinct_starts_and_domains():
    from magictools.itf.nbody import TrajectoryDomainError
    calls=[]
    def apparent(t,o):
        calls.append(t.tolist())
        if np.any(t==3):raise TrajectoryDomainError(t==3)
        return np.column_stack((t,t,t)),t+1,np.tile(np.eye(3,6),(len(t),1,1))
    cache=NightPredictionCache(8);tr=SimpleNamespace(apparent=apparent)
    cache.apparent('a',tr,np.array([0,1]),np.array([0,1]),np.zeros((2,3)))
    with pytest.raises(TrajectoryDomainError) as error:cache.apparent('a',tr,np.array([0,3]),np.array([0,3]),np.zeros((2,3)))
    assert error.value.invalid.tolist()==[False,True]
    cache.apparent('b',tr,np.array([0,1]),np.array([0,1]),np.zeros((2,3)))
    assert calls==[[0,1],[3],[0,1]]


def test_exact_fit_cache_distinguishes_states_options_and_versions(tmp_path):
    state=[2.,0,0,0,.01,0];k=fit_key([1,2,3],state,60000,{'sigma':.3})
    assert k==fit_key([3,1,2],state,60000,{'sigma':.3})
    changed=state.copy();changed[0]=np.nextafter(changed[0],3.)
    assert k!=fit_key([1,2,3],changed,60000,{'sigma':.3})
    assert k!=fit_key([1,2,3],state,60000,{'sigma':.4})
    c=ExactFitCache(tmp_path/'cache.sqlite',{'force':'a'});c.put(k,dict(kind='fit',fit={'state':state}))
    c.get(k)['fit']['state'][0]=9
    assert c.get(k)['fit']['state'][0]==2
    d=ExactFitCache(tmp_path/'cache.sqlite',{'force':'a'});assert d.get(k)['fit']['state'][0]==2
    with pytest.raises(ValueError):ExactFitCache(tmp_path/'cache.sqlite',{'force':'b'})


def database():
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE jobs(key TEXT PRIMARY KEY,body TEXT,status TEXT,attempts INTEGER,error TEXT)');return db


def record():return dict(indices=[1,2,3,4],initial=[2.,0,0,0,.01,0],epoch=60000.,sigma=.3,seed_key='1:0')


def test_incremental_import_handles_duplicates_partial_lines_and_truncation(tmp_path):
    directory=tmp_path/'deferred-fits';directory.mkdir();path=directory/'a.jsonl';r=record();other=dict(r,epoch=60001.)
    path.write_text(json.dumps(r)+'\n'+json.dumps(dict(r,indices=[4,3,2,1]))+'\n'+json.dumps(other)[:15]);db=database()
    first=import_incremental(db,[tmp_path]);assert first['new_jobs']==1 and first['duplicate_inputs']==1
    second=import_incremental(db,[tmp_path]);assert second.get('bytes_read',0)==0
    with path.open('a') as f:f.write(json.dumps(other)[15:]+'\n')
    third=import_incremental(db,[tmp_path]);assert third['new_jobs']==1
    db.execute("UPDATE jobs SET status='accepted'");db.commit()
    path.write_text(json.dumps(r)+'\n');fourth=import_incremental(db,[tmp_path]);assert fourth['reset_sources']==1
    assert db.execute("SELECT count(*) FROM jobs WHERE status='accepted'").fetchone()[0]==2


def test_warm_start_does_not_retire_original_or_merge_starting_modes():
    db=database();r=record();key=key_for(r);db.execute('INSERT INTO jobs VALUES (?,?,?,0,NULL)',(key,json.dumps(r),'retryable'));bank=WarmBank(db)
    bank.add(dict(tracklet_indices=[1,2,3,5],modes=[dict(state_au_au_per_day=[3.,0,0,0,.01,0],epoch_mjd_tdb=60000.)]))
    child=bank.suggest(key,r);assert child and child[0]!=key
    assert db.execute('SELECT status FROM jobs WHERE key=?',(key,)).fetchone()[0]=='retryable'
    assert bank.suggest(key,r) is None
    assert len(list(pending_jobs(db,batch=1)))==2


def test_empty_log_fingerprint_grows_and_detects_rewrite(tmp_path):
    directory=tmp_path/'deferred-fits';directory.mkdir();path=directory/'a.jsonl';path.touch();db=database()
    import_incremental(db,[tmp_path]);path.write_text(json.dumps(record())+'\n')
    import_incremental(db,[tmp_path])
    changed=dict(record(),epoch=60001.);path.write_text(json.dumps(changed)+'\n')
    result=import_incremental(db,[tmp_path])
    assert result['reset_sources']==1 and result['new_jobs']==1


def test_completed_rejection_is_distinct_from_cache_miss(tmp_path):
    cache=ExactFitCache(tmp_path/'cache.sqlite',{'version':1})
    assert cache.get('exact_start') is None
    cache.put('exact_start',dict(kind='completed_rejection',fit=None))
    assert cache.get('exact_start')==dict(kind='completed_rejection',fit=None)
    assert cache.get('different_start') is None




def test_indexed_observers_invalidate_changed_times_and_step():
    from magictools.itf.reuse import IndexedObserverCache
    calls=[]
    def earth(t):calls.append(t.copy());return np.column_stack((t,2*t,3*t))
    cache=IndexedObserverCache(10);ep=SimpleNamespace(earth=earth)
    cache.pair(ep,np.array([1,2]),np.array([10.,20.]),.01)
    cache.pair(ep,np.array([2]),np.array([20.]),.01);assert len(calls)==1
    p,m=cache.pair(ep,np.array([2]),np.array([21.]),.01);assert len(calls)==2 and p[0,0]==21.01
    cache.pair(ep,np.array([1]),np.array([10.]),.02);assert len(calls)==3
