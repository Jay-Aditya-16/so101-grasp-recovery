"""Outcome-conditioned grasp ranking for *simulation experiments only*.

BORGES-inspired Beta-Bernoulli bandit, not BORGES: this code neither observes
object pose nor generates/checks grasps, executes a robot, or implements the
paper's pose-transition model. The caller must supply a visually identified pose
class and candidate IDs already screened for feasibility. Simulator truth may
supply offline evaluation labels only, never online policy observations.

Memory is a single-writer JSON file. A caller must not share one writable memory
across parallel processes. For parallel trials, use separate memories or merge
validated, deduplicated outcomes in one process.
"""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA = "so101-grasp-bandit-beta-v1"


def _name(value, what):
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValueError(f"{what} must be a nonempty trimmed string")
    if what == "pose class" and value.lower() in {"unknown", "unobserved"}:
        raise ValueError("unknown pose must abstain")
    return value


class GraspBandit:
    """Disjoint Beta(1,1) priors over (observed pose class, candidate ID).

    `rank` samples one posterior per supplied feasible candidate using the
    instance's seeded RNG. Samples are exploration scores, not safety scores.
    `update` requires a caller-verified binary outcome and nonempty evidence;
    it cannot itself validate camera observations or simulator truth.
    """

    def __init__(self, path, candidates, seed=0, frozen=False):
        self.path = Path(path)
        self.candidates = tuple(candidates)
        if not self.candidates or len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidates must be a nonempty list of unique IDs")
        for candidate in self.candidates:
            _name(candidate, "candidate")
        if type(frozen) is not bool:
            raise ValueError("frozen must be boolean")
        self.frozen = frozen
        self.rng = np.random.default_rng(seed)
        self.records = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("schema") != SCHEMA or data.get("candidates") != list(self.candidates) or not isinstance(data.get("records"), list):
                raise ValueError("incompatible grasp memory")
            ids = set()
            for record in data["records"]:
                self._validate_record(record)
                if record["trial_id"] in ids:
                    raise ValueError("duplicate grasp trial ID")
                ids.add(record["trial_id"])
            self.records = data["records"]
        elif frozen:
            raise ValueError("frozen mode requires an existing memory file")

    def _validate_record(self, record):
        if not isinstance(record, dict):
            raise ValueError("invalid grasp record")
        _name(record.get("trial_id"), "trial ID")
        _name(record.get("pose_class"), "pose class")
        if record.get("candidate") not in self.candidates:
            raise ValueError("unknown grasp candidate")
        if type(record.get("reward")) is not int or record["reward"] not in (0, 1):
            raise ValueError("reward must be integer 0 or 1")
        _name(record.get("evidence"), "evidence")

    def posterior(self, pose_class, candidate):
        pose_class = _name(pose_class, "pose class")
        if candidate not in self.candidates:
            raise ValueError("unknown grasp candidate")
        successes = sum(r["reward"] for r in self.records if r["pose_class"] == pose_class and r["candidate"] == candidate)
        attempts = sum(1 for r in self.records if r["pose_class"] == pose_class and r["candidate"] == candidate)
        alpha, beta = 1 + successes, 1 + attempts - successes
        return {"alpha": alpha, "beta": beta, "attempts": attempts, "mean": alpha / (alpha + beta)}

    def rank(self, pose_class, feasible_candidates):
        """Sample and rank *only* caller-screened feasible IDs; empty abstains."""
        _name(pose_class, "pose class")
        candidates = tuple(feasible_candidates)
        if len(set(candidates)) != len(candidates) or any(c not in self.candidates for c in candidates):
            raise ValueError("feasible candidates must be unique known IDs")
        scores = {}
        for candidate in candidates:
            state = self.posterior(pose_class, candidate)
            scores[candidate] = {**state, "sample": float(self.rng.beta(state["alpha"], state["beta"]))}
        return sorted(candidates, key=lambda candidate: -scores[candidate]["sample"]), scores

    def update(self, trial_id, pose_class, candidate, reward, *, verified=False, evidence=None):
        """Return True only if a verified outcome was atomically recorded.

        Censored/unverified outcomes and frozen evaluation leave memory untouched.
        Verified labels are supplied by the caller; this class does not infer them.
        """
        _name(trial_id, "trial ID")
        _name(pose_class, "pose class")
        if candidate not in self.candidates:
            raise ValueError("unknown grasp candidate")
        if type(verified) is not bool:
            raise ValueError("verified must be boolean")
        if reward is not None and (type(reward) is not int or reward not in (0, 1)):
            raise ValueError("reward must be integer 0 or 1, or None if censored")
        if not verified or reward is None or self.frozen:
            return False
        record = {"trial_id": trial_id, "pose_class": pose_class, "candidate": candidate,
                  "reward": reward, "evidence": _name(evidence, "evidence")}
        self._validate_record(record)
        if any(r["trial_id"] == trial_id for r in self.records):
            raise ValueError("trial already recorded")
        records = [*self.records, record]
        payload = {"schema": SCHEMA, "candidates": list(self.candidates), "records": records}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as out:
                name = out.name
                json.dump(payload, out, indent=2, allow_nan=False)
                out.write("\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, self.path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)
        self.records = records
        return True
