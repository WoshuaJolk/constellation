import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/solve_itf'))
from distant_brightness_priority import brightness_priority


def test_faint_far_target_outranks_bright_near_target_within_band():
    near=brightness_priority(23,'r',40,40)
    far=brightness_priority(25,'r',150,150)
    assert far['absolute_magnitude_proxy']<near['absolute_magnitude_proxy']
    assert far['diameter_km_at_albedo_004_to_025'] is None
    assert far['lane']==near['lane']=='photometry:r'


def test_missing_inputs_preserved_and_v_size_is_conditional():
    for values in [(None,'r',40,40),(22,'',40,40),(22,'V',-1,40),(np.nan,'V',40,40)]:
        assert brightness_priority(*values)['lane']=='unsized'
    m=5*np.log10(1329/(1000*np.sqrt(.1)))+10*np.log10(150)
    low,high=brightness_priority(m,'V',150,150)['diameter_km_at_albedo_004_to_025']
    assert low<1000<high
    assert brightness_priority(m,'r',150,150)['lane']!=brightness_priority(m,'V',150,150)['lane']
