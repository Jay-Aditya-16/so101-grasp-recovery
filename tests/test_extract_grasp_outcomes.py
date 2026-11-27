import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from extract_grasp_outcomes import extract


def test_lift_proxy_and_censoring():
    t = np.array([0., 1., 2., 3.])
    xyz = np.array([[0,0,0.03], [0,0,0.03], [0,0,0.09], [0,0,0.09]])
    meta = {'recovery_events':[{'state':'closing','time_s':1.}, {'state':'gripper_check','reason':'after_lift','time_s':2.}]}
    assert extract(meta,t,xyz,[False,True,False])['label'] == 1
    assert extract(meta,t,xyz,[False,False,False])['label'] == 0
    meta['recovery_events'].pop()
    assert extract(meta,t,xyz,[False,True,False])['label'] is None
    assert extract({'recovery_events':[]},t,xyz,[False,True,False])['status'] == 'no_recovery_grasp'


def test_rejects_misalignment():
    with pytest.raises(ValueError, match='misaligned'):
        extract({}, [0,1], np.zeros((2,3)), [])
