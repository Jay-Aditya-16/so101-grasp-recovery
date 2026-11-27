"""Simulation-only, disjoint linear UCB for screened box placement targets.

Observations are camera-estimated cube XY. Reward labels are supplied only after
an evaluated placement attempt; no MuJoCo state is exposed to selection.
"""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ACTIONS = ("center", "x_plus", "y_minus", "x_plus_y_minus", "x_minus", "y_plus")
OFFSETS = ((0, 0), (.015, 0), (0, -.015), (.015, -.015), (-.015, 0), (0, .015))
ALPHA = 0.25


def features(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)) or not (0.08 <= xy[0] <= 0.30 and -0.18 <= xy[1] <= 0.18):
        raise ValueError("observed position outside the supported cube workspace")
    return np.array([1.0, (xy[0] - .20) / .10, xy[1] / .18])


class PlacementBandit:
    def __init__(self, path):
        self.path = Path(path)
        self.records = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("schema") != "so101-placement-bandit-v1" or data.get("actions") != list(ACTIONS):
                raise ValueError("incompatible bandit memory")
            ids = set()
            for record in data["records"]:
                self._validate(record)
                if record["id"] in ids:
                    raise ValueError("duplicate bandit trial id")
                ids.add(record["id"])
            self.records = data["records"]

    @staticmethod
    def _validate(record):
        if not isinstance(record.get("id"), str) or not record["id"] or record.get("action") not in ACTIONS or type(record.get("reward")) is not int or record["reward"] not in (0, 1):
            raise ValueError("invalid bandit record")
        features(record["observed_xy_m"])

    def scores(self, xy):
        x = features(xy)
        output = {}
        for action in ACTIONS:
            a, b = np.eye(3), np.zeros(3)
            for record in self.records:
                if record["action"] == action:
                    f = features(record["observed_xy_m"])
                    a += np.outer(f, f)
                    b += (2 * record["reward"] - 1) * f
            mean = float(x @ np.linalg.solve(a, b))
            uncertainty = float(np.sqrt(x @ np.linalg.solve(a, x)))
            output[action] = {"mean": mean, "uncertainty": uncertainty, "score": mean + ALPHA * uncertainty}
        return output

    def ranked(self, xy):
        scores = self.scores(xy)
        return sorted(ACTIONS, key=lambda action: -scores[action]["score"]), scores

    def update(self, record):
        self._validate(record)
        if any(r["id"] == record["id"] for r in self.records):
            raise ValueError("trial already recorded")
        records = [*self.records, record]
        payload = {"schema": "so101-placement-bandit-v1", "actions": list(ACTIONS), "records": records}
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


def verified_outcome(selection, events, result, time_s, trial_id, seed, episode, mode):
    """Create a label only for a complete, observed simulator placement attempt."""
    if selection is None or result is None:
        return None
    release = next((event for event in events if event["state"] == "releasing"), None)
    if release is None or not (result["settled_placement"] or time_s - release["time_s"] >= 0.75):
        return None
    return {"id": trial_id, "action": selection["action"], "observed_xy_m": selection["observed_xy_m"],
            "reward": int(result["settled_placement"]), "seed": seed, "episode": episode,
            "mode": mode, "evidence": "simulated_settled_placement_after_release"}


def summarize_trial(selection, result, record, frozen):
    """Summarize all trials, including censored outcomes; never invent a label."""
    if result is None:
        status = "interrupted"
    elif selection is None:
        status = "no_placement_selection"
    elif record is None:
        status = "unverified_placement"
    elif frozen:
        status = "verified_frozen"
    else:
        status = "verified_updated"
    return {"status": status, "action": selection["action"] if selection else None,
            "reward": record["reward"] if record is not None else None,
            "learner": "centered-linear-ucb-v1"}
