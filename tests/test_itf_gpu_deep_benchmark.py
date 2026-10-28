import sys
from pathlib import Path
import numpy as np
from test_solve_itf import injected_catalogue
from magictools.itf.fit import Geometry,fit_chain
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from gpu_deep_benchmark import seed_positions,cpu_case


def test_permutation_covers_queue_without_replacement():
    values=seed_positions(0,7000000)
    assert values.min()==0 and values.max()==6999999
    assert len(np.unique(values))==7000000
    np.testing.assert_array_equal(seed_positions(1024,100),values[1024:1124])


def test_instrumented_cpu_preserves_fit_and_all_starts(injected_catalogue):
    cat,sites,_=injected_catalogue;g=Geometry(cat,sites);g.prepare(range(6))
    settings=dict(sigma=.3,rms_limit=.6,max_nfev=80,start_policy='full')
    expected=fit_chain(g,[0,1,2],**settings,broad_starts=True)
    actual,starts,error=cpu_case(g,[0,1,2],settings)
    assert error is None and actual==expected
    assert starts==actual['seeds_attempted'] and starts>=44


def test_process_cpu_preserves_instrumented_fit(injected_catalogue):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import gpu_deep_cpu_process_check as check
    cat, sites, _ = injected_catalogue
    check._G = Geometry(cat, sites)
    check._G.prepare(range(6))
    expected = check.run((0, 1, 2))
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('fork')) as pool:
        actual = list(pool.map(check.run, [(0, 1, 2), (0, 1, 2)]))
    assert actual == [expected, expected]


def test_output_comparison_excludes_overtime_and_preserves_disagreements(tmp_path):
    import gzip, json
    from gpu_deep_summary import compare_records
    def record(position, rms, mode_count, within=True):
        return dict(source_position=position, membership=[1, 2, 3], error=None,
                    within_budget=within,
                    row=None if rms is None else dict(rms_arcsec=rms, modes=[{}]*mode_count))
    left = [record(0, .2, 1), record(1, .2, 1), record(2, .1, 1, False)]
    right = [record(0, .3, 2), record(1, None, 0), record(2, .1, 1)]
    paths = [tmp_path/'left.gz', tmp_path/'right.gz']
    for path, rows in zip(paths, [left, right]):
        with gzip.open(path, 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
    result = compare_records(*paths)
    assert result['common_within_budget'] == 2
    assert result['status_changes'] == [dict(source_position=1, left='retained', right='not_retained')]
    assert result['mode_count_changes'] == [0]
    assert len(result['rms_changes_over_001_arcsec']) == 1
