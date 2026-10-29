import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from test_solve_itf import injected_catalogue
from magictools.itf.fit import Geometry, BudgetExpired, fit_chain
from magictools.itf.search import SearchConfig,solve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from gpu_prefetch_adapter import ordered_prepared
from gpu_batch_host import prepare_bank


def receipt():return dict(submitted_batches=0,returned_batches=0,cancelled_batches=0,unused_prepared_batches=0,peak_pending_batches=0,drain_seconds=0.,drain_errors=[])


def test_preparation_is_bounded_ordered_and_drained():
    r=receipt()
    def work(n):
        time.sleep(.005*(4-n%4));return n
    with ThreadPoolExecutor(max_workers=3) as pool:
        it=ordered_prepared(range(12),pool,work,3,r)
        assert next(it)==(0,0)
        assert r['submitted_batches']==4
        assert [x for x in it]==[(n,n) for n in range(1,12)]
    assert r['peak_pending_batches']==3 and r['returned_batches']==12


def test_close_cancels_or_drains_without_replacement():
    r=receipt()
    with ThreadPoolExecutor(max_workers=1) as pool:
        it=ordered_prepared(range(100),pool,lambda x:x,2,r)
        assert next(it)==(0,0)
        it.close()
    assert r['submitted_batches']==3
    assert r['cancelled_batches']+r['unused_prepared_batches']==2


def test_pipeline_hook_preserves_full_search(injected_catalogue):
    c,sites,_=injected_catalogue;config=SearchConfig(workers=1,max_seeds=4,seconds=120)
    expected=solve(c,sites,config,progress=lambda _:None)
    class Adapter:
        batch_size=1
        def iter_batches(self,g,batches):
            for batch in batches:
                yield batch,[fit_chain(g,key,**settings) for key,settings in batch]
    actual=solve(c,sites,config,progress=lambda _:None,batch_fitter=Adapter())
    assert actual['status']==expected['status']=='bounded_search_complete'
    assert actual['candidates']==expected['candidates']
    for key in ['fits_attempted','fits_rejected','triple_seeds_completed','successful_extensions']:
        assert actual['counts'][key]==expected['counts'][key]


def test_pipeline_cannot_omit_or_reorder_batches(injected_catalogue):
    c,sites,_=injected_catalogue;config=SearchConfig(workers=1,max_seeds=4,seconds=120)
    class Broken:
        batch_size=1
        def iter_batches(self,g,batches):
            batch=next(batches)
            yield [(tuple(reversed(batch[0][0])),batch[0][1])],[None]
    with pytest.raises(ValueError,match='order or membership'):
        solve(c,sites,config,progress=lambda _:None,batch_fitter=Broken())


def test_expired_prefetch_remains_uncomputed_not_rejected(injected_catalogue):
    c,sites,_=injected_catalogue
    class Expired:
        batch_size=1
        def iter_batches(self,g,batches):
            next(batches);next(batches)
            raise BudgetExpired()
            yield
    r=solve(c,sites,SearchConfig(workers=1,max_seeds=4,seconds=120),progress=lambda _:None,batch_fitter=Expired())
    assert r['status']=='time_budget_exhausted'
    assert r['counts']['fits_rejected']==r['counts']['fits_attempted']==0
    assert r['counts']['batch_submitted_not_returned']==2
    assert r['counts']['triple_seeds_remaining']==r['counts']['seeds_retained']


def test_exact_indices_match_fitting_vectors_and_stale_cache_refused(injected_catalogue,tmp_path):
    from magictools.itf.campaign_geometry import prepare_atlas
    from magictools.itf.campaign_geometry_fast import FastAtlasGeometry,prepare_observation_ids
    c,sites,sitefile=injected_catalogue
    prepare_atlas(c.path,sitefile,tmp_path/'atlas',lambda _:None)
    prepare_observation_ids(c.path,tmp_path/'ids')
    g=FastAtlasGeometry(c.path,sitefile,tmp_path/'atlas',tmp_path/'ids')
    reference=Geometry(c,sites);reference.prepare(range(len(c.tracks)))
    keys=[0,2,5]
    for a,b in zip(g.get(keys),reference.get(keys)):np.testing.assert_array_equal(a,b)
    bank,*_=prepare_bank(g,[(keys,dict(max_nfev=5))]);ix=bank['observation_index']
    np.testing.assert_array_equal(g.unit[ix],bank['observed'])
    np.testing.assert_array_equal(g.observers[ix],bank['observers'])
    changed=tmp_path/'changed-sites';changed.write_text(sitefile.read_text()+'\n')
    with pytest.raises(ValueError,match='differs'):
        FastAtlasGeometry(c.path,changed,tmp_path/'atlas',tmp_path/'ids')
