import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from gpu_e2e_summary import index_rows


def test_same_display_names_do_not_merge_distinct_catalogue_rows():
    rows=[dict(tracklet_indices=[1,2,3],tracklets=['A','B','C']),
          dict(tracklet_indices=[4,5,6],tracklets=['A','B','C'])]
    assert len(index_rows(rows))==2


def test_duplicate_exact_memberships_are_not_silently_collapsed():
    with pytest.raises(ValueError,match='Duplicate'):
        index_rows([dict(tracklet_indices=[1,2,3]),dict(tracklet_indices=[3,2,1])])
