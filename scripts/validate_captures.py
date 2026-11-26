"""Check saved rollout alignment and labels without using simulator ground truth for control."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.traces:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            n = len(data["action"])
            obs = ("time", "qpos", "qvel", "cube_xyz", "top_rgb", "wrist_rgb")
            transitions = ("action", "bilateral_contact", "box_support", "settled", "reward", "terminated", "truncated")
            problems = []
            for key in obs:
                if len(data[key]) != n + 1:
                    problems.append(f"{key}: expected {n + 1} observations")
            for key in transitions:
                if len(data[key]) != n:
                    problems.append(f"{key}: expected {n} transitions")
            if n == 0:
                problems.append("empty transition sequence")
            else:
                result = meta.get("result")
                success = bool(result and result.get("task_success", result.get("settled_placement", False)))
                if bool(data["terminated"][-1]) != success or bool(data["truncated"][-1]) == success:
                    problems.append("terminal flags do not match outcome")
                if not np.allclose(np.diff(data["time"]), 1 / meta["fps"], atol=1e-5):
                    problems.append("timestamps are not aligned to the recorded control rate")
                if not np.isfinite(data["action"]).all() or not np.isfinite(data["qpos"]).all():
                    problems.append("non-finite action or joint state")
            if meta.get("scenario") == "drawer" and len(data["drawer_position"]) != n + 1:
                problems.append("drawer observations do not align with camera observations")
            rows.append({"trace": str(path), "scenario": meta.get("scenario", "cube"), "transitions": n,
                         "valid": not problems, "problems": problems})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    if not all(row["valid"] for row in rows):
        raise SystemExit("One or more rollout captures failed validation")


if __name__ == "__main__":
    main()
