import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_grasp_study import replay


def test_replay_uses_only_observed_binary_proxies(tmp_path):
    rows = [{'sha256':'a'*64, 'scenario':'cube', 'status':'evaluated_lift_proxy', 'candidate':'low', 'label':0},
            {'sha256':'b'*64, 'scenario':'cube', 'status':'censored_before_lift_check', 'candidate':'high', 'label':None}]
    result = replay(rows, tmp_path / 'memory.json')
    assert result['updates'] == 1
    assert result['posteriors']['low']['mean'] == pytest.approx(1/3)
    assert result['posteriors']['high']['mean'] == pytest.approx(1/2)
    assert len(result['censored']) == 1
    with pytest.raises(ValueError, match='already exist'):
        replay(rows, tmp_path / 'memory.json')


def test_rejects_duplicated_trials(tmp_path):
    row = {'sha256':'a'*64, 'scenario':'cube', 'status':'no_recovery_grasp', 'candidate':None}
    with pytest.raises(ValueError, match='duplicate'):
        replay([row,row], tmp_path / 'unused.json')


def test_lift_and_task_outcomes_do_not_get_conflated(tmp_path):
    row = {'sha256':'a'*64, 'scenario':'cube', 'status':'evaluated_lift_proxy',
           'candidate':'high', 'label':1, 'task_success':False}
    lift = replay([row], tmp_path / 'lift.json', 'lift')
    task = replay([row], tmp_path / 'task.json', 'task')
    assert lift['posteriors']['high']['mean'] == pytest.approx(2/3)
    assert task['posteriors']['high']['mean'] == pytest.approx(1/3)
