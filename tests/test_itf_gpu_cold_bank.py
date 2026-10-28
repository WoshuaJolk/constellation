"""Seed expansion must match production fit_chain, not a favorable invented grid."""
import importlib.util
from pathlib import Path
import numpy as np
from magictools.itf import fit
from test_itf_gpu import fixture_bank
spec=importlib.util.spec_from_file_location('cold_bank',Path(__file__).resolve().parents[1]/'research/solve_itf/gpu_cold_bank.py')
cold=importlib.util.module_from_spec(spec);spec.loader.exec_module(cold)


def test_grid_exactly_matches_production_range_radial_starts(monkeypatch):
    original=fixture_bank();bank={k:original[k][:5].copy() for k in ('offsets','observers','observed','east','north')}
    for k in ('observers','observed','east','north'):bank[k]=np.vstack([bank[k],bank[k][-1]])
    bank['offsets']=np.r_[bank['offsets'],40.]
    bank.update(states=original['states'][:1].copy(),case_index=np.zeros(6,np.int64),sigma=.3)
    captured=[]
    def fake_solver(state,*args):
        captured.append(state.copy())
        return state,np.full(12,1e12),np.zeros((12,6)),False
    class Geometry:
        def get(self,indices):
            return bank['offsets'],bank['observed'],bank['observers'],np.repeat(np.arange(3),2)
    monkeypatch.setattr(fit,'gauss_seeds',lambda *args:[])
    monkeypatch.setattr(fit,'solve_lm',fake_solver)
    assert fit.fit_chain(Geometry(),[0,1,2],start_policy='full') is None
    expanded=cold.expand(bank)
    np.testing.assert_array_equal(expanded['states'],np.array(captured))
    assert len(captured)==24
    for i in range(24):
        sl=slice(6*i,6*(i+1))
        np.testing.assert_array_equal(expanded['offsets'][sl],bank['offsets']-bank['offsets'][3])
        for k in ('observers','observed','east','north'):np.testing.assert_array_equal(expanded[k][sl],bank[k])
