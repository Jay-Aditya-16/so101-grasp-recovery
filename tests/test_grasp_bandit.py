import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from grasp_bandit import GraspBandit


@pytest.fixture
def candidates():
    return ("approach_a", "approach_b")


def test_beta_updates_and_pose_isolation(tmp_path, candidates):
    bandit = GraspBandit(tmp_path / "memory.json", candidates, seed=7)
    assert bandit.posterior("cube_face", "approach_a")["mean"] == pytest.approx(.5)
    assert bandit.update("t1", "cube_face", "approach_a", 0, verified=True, evidence="sim_lift_proxy")
    state = bandit.posterior("cube_face", "approach_a")
    assert (state["alpha"], state["beta"], state["attempts"]) == (1, 2, 1)
    assert state["mean"] == pytest.approx(1 / 3)
    assert bandit.posterior("other_pose", "approach_a")["mean"] == .5
    assert bandit.update("t2", "cube_face", "approach_a", 1, verified=True, evidence="sim_lift_proxy")
    assert bandit.posterior("cube_face", "approach_a")["mean"] == .5
    reloaded = GraspBandit(bandit.path, candidates)
    assert reloaded.posterior("cube_face", "approach_a") == bandit.posterior("cube_face", "approach_a")
    with pytest.raises(ValueError, match="already"):
        reloaded.update("t1", "cube_face", "approach_a", 1, verified=True, evidence="label")


def test_rank_reproducible_and_restricts_to_feasible(tmp_path, candidates):
    left = GraspBandit(tmp_path / "left", candidates, seed=41)
    right = GraspBandit(tmp_path / "right", candidates, seed=41)
    assert left.rank("cube_face", candidates) == right.rank("cube_face", candidates)
    ranking, scores = left.rank("cube_face", ["approach_b"])
    assert ranking == ["approach_b"] and set(scores) == {"approach_b"}
    assert left.rank("cube_face", []) == ([], {})
    with pytest.raises(ValueError, match="feasible"):
        left.rank("cube_face", ["approach_a", "bad"])
    with pytest.raises(ValueError, match="unknown pose"):
        left.rank("unknown", candidates)


def test_censored_unverified_and_frozen_cannot_mutate_memory(tmp_path, candidates):
    path = tmp_path / "memory.json"
    bandit = GraspBandit(path, candidates)
    assert not bandit.update("censored", "cube", "approach_a", None, verified=True)
    assert not bandit.update("unverified", "cube", "approach_a", 0, verified=False)
    assert not path.exists()
    with pytest.raises(ValueError, match="evidence"):
        bandit.update("no-evidence", "cube", "approach_a", 1, verified=True)
    bandit.update("one", "cube", "approach_a", 1, verified=True, evidence="sim")
    before = path.read_bytes()
    frozen = GraspBandit(path, candidates, frozen=True)
    assert not frozen.update("two", "cube", "approach_a", 0, verified=True, evidence="sim")
    assert path.read_bytes() == before
    assert len(frozen.records) == 1
    with pytest.raises(ValueError, match="existing"):
        GraspBandit(tmp_path / "missing", candidates, frozen=True)


def test_bad_and_corrupted_memory_rejected(tmp_path, candidates):
    path = tmp_path / "memory.json"
    bandit = GraspBandit(path, candidates)
    with pytest.raises(ValueError, match="reward"):
        bandit.update("bad", "cube", "approach_a", True, verified=True, evidence="sim")
    bandit.update("a", "cube", "approach_a", 1, verified=True, evidence="sim")
    with pytest.raises(ValueError, match="incompatible"):
        GraspBandit(path, candidates[::-1])
    obj = json.loads(path.read_text())
    obj["records"].append(obj["records"][0])
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="duplicate"):
        GraspBandit(path, candidates)
