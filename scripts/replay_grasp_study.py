"""Offline, single-writer replay of simulator grasp proxies into a bandit.

This experimental analysis does not make online decisions. The one pose class is
an explicit fixed-cube assumption, not a camera-estimated stable pose. Labels
come from privileged simulation state and contact and cannot be used on hardware.
"""
import argparse
import json
from pathlib import Path

from grasp_bandit import GraspBandit

CANDIDATES = ("low", "center", "high")
POSE_CLASS = "fixed_red_cube_assumed_single_class"


def replay(rows, memory_path, outcome="lift"):
    if outcome not in ("lift", "task"):
        raise ValueError("outcome must be lift or task")
    memory_path = Path(memory_path)
    if memory_path.exists():
        raise ValueError("memory output must not already exist")
    censored = []
    verified = []
    seen = set()
    for row in rows:
        sha = row.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or sha in seen:
            raise ValueError("missing or duplicate capture digest")
        seen.add(sha)
        if row.get("scenario") != "cube":
            raise ValueError("this experiment only supports the known cube")
        candidate = row.get("candidate")
        if row.get("status") != "evaluated_lift_proxy" or candidate is None:
            censored.append({"sha256": sha, "status": row.get("status")})
            continue
        if outcome == "task" and type(row.get("task_success")) is not bool:
            censored.append({"sha256": sha, "status": "unscored_task"})
            continue
        reward = row.get("label") if outcome == "lift" else int(row["task_success"])
        if type(reward) is not int or reward not in (0, 1) or candidate not in CANDIDATES:
            raise ValueError("invalid candidate or grasp proxy")
        verified.append((sha, candidate, reward))
    bandit = GraspBandit(memory_path, CANDIDATES, seed=0)
    for sha, candidate, reward in verified:
        bandit.update(sha, POSE_CLASS, candidate, reward, verified=True,
                      evidence="offline_simulator_lift_proxy" if outcome == "lift" else "offline_complete_task_outcome")
    return {"pose_class_assumption": POSE_CLASS, "outcome": outcome, "updates": len(bandit.records),
            "censored": censored, "posteriors": {candidate: bandit.posterior(POSE_CLASS, candidate) for candidate in CANDIDATES},
            "limitations": "Offline privileged simulator proxy, tiny development sample, not online grasp learning or a BORGES reproduction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proxies", type=Path)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcome", choices=("lift", "task"), default="lift")
    args = parser.parse_args()
    result = replay(json.loads(args.proxies.read_text()), args.memory, args.outcome)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
