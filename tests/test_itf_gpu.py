"""Run with NUMBA_ENABLE_CUDASIM=1; these are not hardware benchmarks."""
import numpy as np
import pytest
from numba import cuda
from magictools.itf.gpu import ResidentBank, cpu_reference, validate_bank
from magictools.itf.derivatives import universal_scalars
from magictools.itf.kernels import MU


def fixture_bank():
    states = np.array([
        [2., .1, .2, -.001, .011, .002],
        [50., .1, 1., 0., .0024, .0002],
        [4., 0., .1, 0., -.009, .001],
        [1., 0., 0., 0., np.sqrt(2*MU), 0.],
        [1., 0., 0., .01, .04, -.003],
        [1., 0., .1, -.01, .002, .003],
        [0., 0., 0., 0., 0., 0.],
    ], dtype=np.float64)
    case = np.repeat(np.arange(len(states), dtype=np.int64), 5)
    offsets = np.tile(np.array([-30., -1., 0., 1., 30.]), len(states))
    observers = np.tile([.8, .3, .01], (len(case), 1))
    u = []
    for c, dt, ob in zip(case, offsets, observers):
        p = np.array(universal_scalars(states[c], dt)[:3])-ob
        if not np.isfinite(p).all():
            p = np.array([1., .2, .1])
        u.append(p/np.linalg.norm(p))
    u = np.array(u)
    east = np.column_stack((-u[:, 1], u[:, 0], np.zeros(len(u))))
    east /= np.linalg.norm(east, axis=1)[:, None]
    return dict(states=states, case_index=case, offsets=offsets,
                observers=observers, observed=u, east=east,
                north=np.cross(u, east), sigma=.3)


@pytest.mark.skipif(not cuda.is_available(), reason='CUDA or explicit simulator required')
def test_all_conics_and_failure_parity():
    bank = fixture_bank()
    ref = cpu_reference(bank)
    actual = ResidentBank(bank, allow_simulator=True).evaluate()
    np.testing.assert_array_equal(actual[2], ref[2])
    good = ref[2].astype(bool)
    np.testing.assert_allclose(actual[0][good], ref[0][good], atol=1e-6, rtol=1e-8)
    np.testing.assert_allclose(actual[1][good], ref[1][good], atol=1e-5, rtol=1e-8)
    assert not actual[2][-5:].any()
    assert np.isnan(actual[0][-5:]).all()


def test_refuse_single_precision_and_bad_indices():
    bank = fixture_bank()
    bank['states'] = bank['states'].astype(np.float32)
    with pytest.raises(ValueError, match='float64'):
        validate_bank(bank)
    bank = fixture_bank()
    bank['case_index'][0] = -1
    with pytest.raises(ValueError, match='invalid case'):
        validate_bank(bank)


def test_refuse_wrong_tangent_basis():
    bank = fixture_bank()
    bank['north'] *= -1
    with pytest.raises(ValueError, match='basis'):
        validate_bank(bank)


@pytest.mark.skipif(not cuda.config.ENABLE_CUDASIM, reason='simulator guard')
def test_simulator_cannot_be_mislabeled_as_gpu():
    with pytest.raises(RuntimeError, match='not a GPU benchmark'):
        ResidentBank(fixture_bank())
