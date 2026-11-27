import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from placement_bandit import ACTIONS, PlacementBandit, features


def test_update_persists_and_influences_scores(tmp_path):
    path = tmp_path / "memory.json"
    bandit = PlacementBandit(path)
    xy = [0.20, 0.0]
    assert bandit.ranked(xy)[0] == list(ACTIONS)
    bandit.update({"id": "one", "action": "y_minus", "observed_xy_m": xy, "reward": 1})
    restored = PlacementBandit(path)
    assert restored.ranked(xy)[0][0] == "y_minus"
    assert restored.scores(xy)["y_minus"]["mean"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="already"):
        restored.update({"id": "one", "action": "center", "observed_xy_m": xy, "reward": 0})
    assert len(PlacementBandit(path).records) == 1


def test_failed_attempt_reduces_uncertainty_and_changes_ranking(tmp_path):
    bandit = PlacementBandit(tmp_path / "memory.json")
    xy = [0.20, 0.0]
    before = bandit.scores(xy)["center"]["uncertainty"]
    bandit.update({"id": "miss", "action": "center", "observed_xy_m": xy, "reward": 0})
    assert bandit.scores(xy)["center"]["uncertainty"] < before
    assert bandit.ranked(xy)[0][0] == "x_plus"
    assert bandit.scores(xy)["center"]["mean"] == pytest.approx(-0.5)


@pytest.mark.parametrize("xy", [[float('nan'), 0], [0.5, 0], [0.2], [0.2, float('inf')]])
def test_bad_observations_rejected(xy):
    with pytest.raises(ValueError):
        features(xy)


def test_interrupted_and_incomplete_trials_do_not_get_labels():
    from placement_bandit import verified_outcome
    selection = {"action": "center", "observed_xy_m": [0.2, 0.0]}
    release = [{"state": "releasing", "time_s": 10.0}]
    args = ("trial", 1020, 1, "active")
    assert verified_outcome(selection, release, None, 11.0, *args) is None
    assert verified_outcome(selection, [], {"settled_placement": False}, 12.0, *args) is None
    assert verified_outcome(selection, release, {"settled_placement": False}, 10.5, *args) is None
    assert verified_outcome(selection, release, {"settled_placement": False}, 10.8, *args)["reward"] == 0
    assert verified_outcome(selection, release, {"settled_placement": True}, 10.7, *args)["reward"] == 1


def test_trial_summary_distinguishes_censored_and_frozen():
    from placement_bandit import summarize_trial
    sel = {"action": "center"}
    result = {"settled_placement": False}
    assert summarize_trial(sel, None, None, False)["status"] == "interrupted"
    assert summarize_trial(None, result, None, False)["status"] == "no_placement_selection"
    assert summarize_trial(sel, result, None, False)["reward"] is None
    assert summarize_trial(sel, result, {"reward": 0}, True)["status"] == "verified_frozen"
    assert summarize_trial(sel, result, {"reward": 0}, False)["status"] == "verified_updated"
