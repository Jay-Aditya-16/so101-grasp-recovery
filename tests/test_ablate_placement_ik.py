import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ablate_placement_ik import summarize_thresholds


def test_sweep_preserves_down_ik_threshold_and_checks_path():
    base = {'in_joint_limits': True, 'position_error_m': 0.001,
            'path': {'first_penetration': None}}
    trials = [dict(base, candidate='a', direction='position_only', axis_dot=.66),
              dict(base, candidate='b', direction='down', axis_dot=.66),
              dict(base, candidate='c', direction='position_only', axis_dot=.8,
                   path={'first_penetration': {'sample': 1}})]
    result = summarize_thresholds(trials)
    assert result['0.7']['feasible_solver_outputs'] == 0
    assert result['0.65']['feasible_solver_outputs'] == 1
    assert result['0.65']['candidates'] == ['a']
    assert result['0.65']['max_tilt_deg'] == pytest.approx(49.4583981265)
