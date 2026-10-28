import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
from test_solve_itf import injected_catalogue
from magictools.itf.fit import Geometry, prepare_fit, fit_prepared
from magictools.itf.derivatives import solve_lm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'research/solve_itf'))
from gpu_batch_host import prepare_bank, finish_group


def reference():
    name = 'magictools.itf._host_batch_reference'
    spec = importlib.util.spec_from_file_location(name, ROOT/'research/solve_itf/gpu-host-receipts/reference_fit.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('policy,broad', [('full', True), ('full', False), ('scout', True), ('provided', False)])
def test_preparation_matches_frozen_seed_bits(injected_catalogue, policy, broad):
    cat, sites, _ = injected_catalogue
    g = Geometry(cat, sites); g.prepare(range(6))
    extra = [(np.array([2., .1, 0., 0., .01, 0.]), 60000.25), (np.full(6, np.nan), 60000.)]
    settings = dict(start_policy=policy, broad_starts=broad, extra_seeds=extra)
    old = reference().prepare_fit(g, [0, 1, 2], **settings)
    new = prepare_fit(g, [0, 1, 2], **settings)
    assert len(old.seeds) == len(new.seeds)
    for a, b in zip(old.seeds, new.seeds):
        np.testing.assert_array_equal(a[0], b[0]); assert a[1] == b[1]
    bank, plans, groups, epochs, bounds, limit = prepare_bank(g, [([0, 1, 2], settings)])
    expected = [(s, e) for s, e in old.seeds if np.isfinite(s).all()]
    assert len(bank['states']) == len(expected)
    for i, (state, epoch) in enumerate(expected):
        lo, hi = bounds[i:i+2]
        np.testing.assert_array_equal(bank['states'][i], state)
        np.testing.assert_array_equal(bank['offsets'][lo:hi], old.t-epoch)
        for name, value in [('observers',old.observers),('observed',old.u),('east',old.east),('north',old.north)]:
            np.testing.assert_array_equal(bank[name][lo:hi], value)
        assert np.all(bank['case_index'][lo:hi] == i)


def test_compiled_selection_matches_frozen_fit(injected_catalogue):
    cat, sites, _ = injected_catalogue
    g = Geometry(cat, sites); g.prepare(range(6))
    settings = dict(broad_starts=True, sigma=.3, rms_limit=.6, max_nfev=80)
    saved = []
    def capture(*args):
        result = solve_lm(*args); saved.append(result); return result
    old = reference().fit_chain(g, [0, 1, 2], optimizer=capture, **settings)
    bank, plans, groups, epochs, bounds, _ = prepare_bank(g, [([0, 1, 2], settings)])
    states = np.stack([r[0] for r in saved]); flags = np.array([r[3] for r in saved])
    residuals = np.concatenate([r[1] for r in saved]); jac = np.concatenate([r[2] for r in saved])
    new = finish_group(plans[0], settings, groups[0], epochs, bounds, states, flags, residuals, jac)
    assert old == new
    # A numerically successful optimizer with poor residuals remains rejected.
    residuals[:] = 1e6
    assert finish_group(plans[0], settings, groups[0], epochs, bounds, states, flags, residuals, jac) is None


def test_no_starts_and_mixed_policy_rejection(injected_catalogue):
    cat, sites, _ = injected_catalogue
    g = Geometry(cat, sites); g.prepare(range(6))
    bank, plans, groups, epochs, bounds, _ = prepare_bank(g, [([0, 1, 2], {'start_policy':'provided'})])
    assert bank['states'].shape == (0, 6)
    assert groups == [(0, 0)]
    with pytest.raises(ValueError, match='one sigma'):
        prepare_bank(g, [([0, 1, 2], {'sigma': .3}), ([0, 1, 2], {'sigma': .4})])
    with pytest.raises(ValueError, match='integral'):
        prepare_bank(g, [([0, 1, 2], {'max_nfev':80.5})])
    with pytest.raises(ValueError, match='already be float64'):
        prepare_bank(g, [([0, 1, 2], {'start_policy':'provided', 'extra_seeds':[(np.ones(6,dtype=np.float32),60000.)]})])
    if np.dtype(np.longdouble).itemsize > 8:
        with pytest.raises(ValueError, match='epochs'):
            prepare_bank(g, [([0, 1, 2], {'start_policy':'provided', 'extra_seeds':[(np.ones(6),np.longdouble(60000.))]})])


def test_ordered_duplicate_replacement_and_ties(injected_catalogue):
    cat, sites, _ = injected_catalogue
    g = Geometry(cat, sites); g.prepare(range(6))
    old = reference()
    best = old.fit_chain(g, [0, 1, 2], broad_starts=True)['modes'][0]
    state, epoch = np.array(best['state_au_au_per_day']), best['epoch_mjd_tdb']
    settings = dict(start_policy='provided', extra_seeds=[(state, epoch)]*4)
    bank, plans, groups, epochs, bounds, _ = prepare_bank(g, [([0, 1, 2], settings)])
    from magictools.itf.derivatives import residual_jacobian
    plan = plans[0]
    residual, jac = residual_jacobian(state, plan.t-epoch, plan.observers, plan.u, plan.east, plan.north, .3)
    answers = [(state, residual*scale, jac, True) for scale in [1.1, 1., 1., 1.2]]
    iterator = iter(answers)
    expected = old.fit_chain(g, [0, 1, 2], optimizer=lambda *args:next(iterator), **settings)
    actual = finish_group(plan, settings, groups[0], epochs, bounds, np.stack([r[0] for r in answers]),
        np.ones(4,dtype=bool), np.concatenate([r[1] for r in answers]), np.concatenate([r[2] for r in answers]))
    assert actual == expected
    assert len(actual['modes']) == 1


def test_gpu_compact_matches_legacy_adapter(injected_catalogue):
    from numba import cuda
    if not cuda.is_available():
        pytest.skip('CUDA or explicit simulator required')
    from gpu_pipeline_adapter import GPUBatch
    from gpu_compact_adapter import CompactGPUBatch
    cat, sites, _ = injected_catalogue
    g = Geometry(cat, sites); g.prepare(range(6))
    state = np.array([2.4, .7, .3, -.004, .009, .002])
    epoch = float(g.get([0, 1, 2, 3, 4, 5])[0].mean())
    settings = dict(start_policy='provided', extra_seeds=[(state, epoch)], max_nfev=5)
    jobs = [([0, 1, 2], settings), ([1, 2, 3], settings)]
    expected = GPUBatch(allow_simulator=True)(g, jobs)
    actual = CompactGPUBatch(allow_simulator=True)(g, jobs)
    assert actual == expected
