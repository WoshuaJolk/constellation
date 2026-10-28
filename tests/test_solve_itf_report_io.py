import copy
import importlib.util
import json
from pathlib import Path

import pytest

from magictools.itf.longarc import read_recovery_input, seed_bank
from magictools.itf.report_io import metadata, candidates


def reports():
    mode = dict(epoch_mjd_tdb=59000., state_au_au_per_day=[1., 2., 3., .01, .02, .03],
                range_au=5., local_uncertainty=dict(local_linearization_weak=False, range_sigma_au=.1))
    row = dict(tracklet_indices=[1, 2, 3], rank=1, span_days=4., maximal=True, modes=[mode])
    base = dict(input={'source': 'test'}, candidates=[row], trailing={'number': 42})
    other = copy.deepcopy(base)
    other['candidates'][0]['modes'].append(dict(mode, state_au_au_per_day=[2., 2., 3., .01, .02, .03]))
    other['candidates'].append(dict(row, rank=2, tracklet_indices=[4, 5, 6]))
    return base, other


def test_stream_metadata_and_modes_match_in_memory_with_trailing_fields(tmp_path):
    base, _ = reports()
    row = base['candidates'][0]
    weak = copy.deepcopy(row['modes'][0])
    weak['local_uncertainty']['local_linearization_weak'] = True
    row['modes'].append(weak)
    row['modes'].append(dict(weak, range_au=.01))
    base['candidates'].append(dict(row, rank=2, tracklet_indices=[4, 5, 6],
        modes=[dict(weak, range_au=.001)]))
    base['candidates'].append(dict(row, rank=3, maximal=False))
    path = tmp_path/'report.json'; path.write_text(json.dumps(base))
    assert metadata(path) == {k: v for k, v in base.items() if k != 'candidates'}
    assert list(candidates(path)) == base['candidates']
    for allow_weak in (False, True):
        expected = seed_bank(base, .5, allow_weak)
        header, actual, groups = read_recovery_input(path, .5, allow_weak, True)
        assert [s[:4] for s in actual] == [s[:4] for s in expected]
        assert [s[5] for s in actual] == [s[5] for s in expected]
        assert [s[4] for s in actual] == [{k: v for k, v in s[4].items() if k != 'modes'} for s in expected]
        # An ineligible row must still be available for whole-arc merging.
        assert groups == [[1, 2, 3], [4, 5, 6]]


def test_disk_merge_matches_reference_and_failed_merge_preserves_output(tmp_path):
    spec = importlib.util.spec_from_file_location('bank', Path(__file__).parents[1]/'research/solve_itf/make_recovery_bank.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    base, other = reports()
    paths = [tmp_path/'base.json', tmp_path/'other.json']
    for path, report in zip(paths, [base, other]):
        path.write_text(json.dumps(report))
    output = tmp_path/'merged.json'
    result = module.stream_combine(paths, output)
    expected = module.combine(base, [(other, module.sha256(paths[1]))])
    actual = json.loads(output.read_text())
    assert actual['candidates'] == expected['candidates']
    assert result['candidates'] == 2 and result['modes'] == 3
    assert result['additional_modes_on_shared_chains'] == 1
    saved = output.read_bytes()
    other['input'] = {'source': 'different'}; paths[1].write_text(json.dumps(other))
    with pytest.raises(ValueError, match='snapshots'):
        module.stream_combine(paths, output)
    assert output.read_bytes() == saved
