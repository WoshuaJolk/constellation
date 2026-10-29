"""Compare the frozen previous fitter and compiled bookkeeping on identical data."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from test_solve_itf import injected_catalogue
from magictools.itf.fit import Geometry,fit_chain,prepare_fit,fit_prepared,local_uncertainty,orbital_elements


def reference():
    path=Path(__file__).resolve().parents[1]/'research/solve_itf/gpu-bookkeeping-receipts/reference_fit.py'
    spec=importlib.util.spec_from_file_location('magictools.itf._test_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10)
    else:assert a==b


def test_all_modes_and_uncertainty_preserved(injected_catalogue):
    cat,sites,_=injected_catalogue;geometry=Geometry(cat,sites);geometry.prepare(range(6))
    settings=dict(broad_starts=True)
    a=reference().fit_chain(geometry,[0,1,2],**settings)
    b=fit_chain(geometry,[0,1,2],**settings)
    assert a is not None
    compare(a,b)
    for am,bm in zip(a['modes'],b['modes']):
        for key in ['state_au_au_per_day','chi2','rms_arcsec','relative_mode_likelihood','jacobian_rank']:
            assert am[key]==bm[key]


def test_prepare_reuse_does_not_touch_geometry_again(injected_catalogue):
    cat,sites,_=injected_catalogue;g=Geometry(cat,sites);g.prepare(range(6))
    plan=prepare_fit(g,[0,1,2],start_policy='scout')
    first=fit_prepared(plan,start_policy='scout')
    g.get=lambda _:(_ for _ in ()).throw(AssertionError('unexpected second preparation'))
    second=fit_prepared(plan)
    assert first==second
    assert second is None or second['orbit_start_policy']=='scout'
    with pytest.raises(ValueError,match='cannot be relabelled'):
        fit_prepared(plan,start_policy='full')


def test_uncertainty_rank_and_elements_edge_cases():
    ref=reference();state=np.array([2.4,.7,.3,-.004,.009,.002]);observer=np.array([1.,0,0])
    rng=np.random.default_rng(104)
    for j in [rng.normal(size=(12,6)),np.zeros((12,6)),np.ones((12,6))]:
        compare(ref.local_uncertainty(state,j,1.,observer),local_uncertainty(state,j,1.,observer))
    for x in [state,state*np.array([1,1,1,10,10,10]),np.array([2.,0,0,.01,0,0])]:
        compare(ref.orbital_elements(x),orbital_elements(x))
