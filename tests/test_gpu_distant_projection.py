import sys
from pathlib import Path
import numpy as np
import pytest
from numba import cuda
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from distant_coverage_plan import coverage_plan


def test_coverage_is_complete_unique_and_spans_early():
    epochs = np.arange(100.)*180
    distances = np.geomspace(10, 2000, 16)
    plan = coverage_plan(epochs, distances, [-.5, 0, .5])
    assert len(plan) == len({x['task_id'] for x in plan}) == 100*16*3
    assert len({x['epoch'] for x in plan[:100]}) == 100
    assert len({x['distance_au'] for x in plan[:16]}) == 16
    assert len({(x['distance_au'], x['radial_fraction']) for x in plan[:48]}) == 48
    assert plan == coverage_plan(epochs, distances, [-.5, 0, .5])
    assert {tuple(x[k] for k in ('epoch','distance_au','radial_fraction')) for x in plan} == {
        (e,d,r) for e in epochs for d in distances for r in (-.5,0,.5)}


def test_plan_invalid_and_empty():
    assert coverage_plan([], [40], [0]) == []
    with pytest.raises(ValueError): coverage_plan([0, 0], [40], [0])
    with pytest.raises(ValueError): coverage_plan([0], [-1], [0])
    with pytest.raises(ValueError): coverage_plan([np.nan], [40], [0])


@pytest.mark.skipif(not cuda.is_available(), reason='CUDA hardware or explicit simulator required')
def test_resident_projection_cpu_parity_and_reuse():
    from gpu_distant_projection import ProjectionBank
    from distant_epoch_graph import projected, DISTANCES
    from magictools.itf.kernels import MU, ARCSEC
    rng = np.random.default_rng(20261029)
    t = np.array([-366., -365., -150., 0., 100., 365., 366.])
    u = rng.normal(size=(len(t),3));u /= np.linalg.norm(u,axis=1)[:,None]
    du = rng.normal(size=u.shape)*1e-5; du -= (du*u).sum(axis=1)[:,None]*u
    obs = rng.normal(size=u.shape)*.5
    vel = rng.normal(size=u.shape)*.01
    bank = ProjectionBank(t,u,du,obs,vel,allow_simulator=True)
    h = np.array([(d,f) for d in DISTANCES for f in (-.5,0,.5)])
    for ref in (0., 37.):
        earth = np.array([.3,-.8,.1])
        actual, receipt = bank.project(ref,earth,h)
        assert receipt['output_bytes'] == actual.nbytes
        for j,(d,f) in enumerate(h):
            expected = projected(t,u,du,obs,vel,ref,earth,d,f*np.sqrt(2*MU/d))
            expected[np.abs(ref-t)>365] = np.nan
            np.testing.assert_array_equal(np.isfinite(actual[j]),np.isfinite(expected))
            good = np.isfinite(expected).all(axis=1)
            error = np.linalg.norm(np.cross(expected[good],actual[j,good]),axis=1)*ARCSEC
            assert np.max(error, initial=0) < 1e-5
            np.testing.assert_allclose(actual[j,good],expected[good],atol=1e-10,rtol=0)
    with pytest.raises(MemoryError): bank.project(0.,earth,h,max_output_bytes=1)
    with pytest.raises(ValueError): bank.project(0.,earth,np.array([[-1.,0.]]))
    with pytest.raises(ValueError): ProjectionBank(t.astype('f4'),u,du,obs,vel,allow_simulator=True)
