import numpy as np
import pytest
from numba import cuda
from magictools.itf.batched_fit import CPUEvaluator, solve_batch
from magictools.itf.derivatives import solve_lm
from magictools.itf.gpu import ResidentBank, cpu_reference
from test_itf_gpu import fixture_bank


@pytest.mark.parametrize('iterations', [0, 1, 12])
def test_batch_matches_original_solver_including_invalid_and_exhausted(iterations):
    bank = fixture_bank()
    bank['states'] = bank['states'].copy()
    bank['states'][:-1, 0] += .001  # genuine optimization, same observations
    evaluator = CPUEvaluator(bank)
    result = solve_batch(bank, evaluator, iterations)
    for c in range(len(bank['states'])):
        lo, hi = evaluator.bounds[c:c+2]
        args = [bank[name][lo:hi] for name in ('offsets','observers','observed','east','north')]
        state, r, j, ok = solve_lm(bank['states'][c], *args, float(bank['sigma']), iterations)
        np.testing.assert_array_equal(result['states'][c], state)
        np.testing.assert_array_equal(result['residual'][2*lo:2*hi], r)
        assert result['success'][c] == ok
    assert not result['success'][-1]
    if iterations:
        assert result['reasons'][-1] == 2


@pytest.mark.skipif(not cuda.is_available(), reason='CUDA or explicit simulator required')
def test_resident_state_update_and_inactive_cases_preserve_results():
    bank = fixture_bank()
    resident = ResidentBank(bank, allow_simulator=True)
    before = resident.evaluate()
    new_states = bank['states'].copy(); new_states[0, 0] += .001
    mask = np.zeros(len(new_states), np.uint8);mask[0] = 1
    after = resident.evaluate(states=new_states, active=mask)
    reference = cpu_reference(dict(bank, states=new_states))
    ix = bank['case_index'] == 0
    np.testing.assert_allclose(after[0][ix], reference[0][ix], atol=1e-6, rtol=1e-8)
    for first, second in zip(before, after):
        np.testing.assert_array_equal(first[~ix], second[~ix])
    with pytest.raises(ValueError, match='mask'):
        resident.evaluate(active=np.ones(1, np.uint8))

@pytest.mark.skipif(not cuda.is_available(), reason='CUDA or explicit simulator required')
@pytest.mark.parametrize('iterations', [0, 1, 12])
def test_resident_solver_preserves_policy_and_invalid_cases(iterations):
    from magictools.itf.gpu_fit import solve_resident
    bank = fixture_bank()
    bank['states'] = bank['states'].copy();bank['states'][:-1,0] += .001
    ref = solve_batch(bank,CPUEvaluator(bank),iterations)
    result = solve_resident(bank,iterations,allow_simulator=True)
    np.testing.assert_array_equal(result['success'],ref['success'])
    np.testing.assert_allclose(result['states'],ref['states'],atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(result['cost'],ref['cost'],atol=1e-6,rtol=1e-6)
    assert not result['success'][-1]
    assert result['reasons'][-1] == (2 if iterations else 4)
    assert result['profile']['per_iteration_full_array_transfers']==0
