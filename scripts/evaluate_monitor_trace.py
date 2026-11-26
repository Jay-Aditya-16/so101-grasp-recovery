"""Evaluate monitor alerts on saved ACT traces using simulator outcomes as labels."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from failure_monitor import GraspMonitor


def evaluate_trace(path, tutorial):
    sys.path.insert(0, str(tutorial.resolve()))
    from common.robot import SO101Sim

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
        robot = SO101Sim(cube_pos=metadata["start_xy"], box_pos=(0.05, 0.22), render=False)
        monitor = GraspMonitor()
        event = None
        close_frame = None
        for index, action in enumerate(data["action"]):
            robot.data.qpos[:6] = data["qpos"][index, :6]
            mujoco.mj_kinematics(robot.model, robot.data)
            if close_frame is None and float(action[5]) < monitor.close_threshold:
                close_frame = index
            event = monitor.update(
                data["top_rgb"][index], float(action[5]),
                float(robot.data.site_xpos[robot.site_id][2]),
            )
            if event is not None:
                break

        result = metadata["result"]
        if result is None:
            outcome = "unlabeled"
        elif result["settled_placement"]:
            outcome = "success"
        elif result["lifted_5cm"]:
            outcome = "lifted_not_placed"
        else:
            outcome = "failure_no_lift"
        status = event.status if event else "no_event"
        # Hash the actual policy/scene trace rather than its filename to reveal accidental duplicates.
        digest = hashlib.sha256()
        for key in ("action", "qpos", "cube_xyz"):
            digest.update(np.ascontiguousarray(data[key]).tobytes())
        row = {
            "trace": str(path),
            "trace_sha256": digest.hexdigest(),
            "outcome": outcome,
            "monitor": status,
            "alert_time_s": event.frame / metadata["fps"] if event else None,
            "closure_time_s": close_frame / metadata["fps"] if close_frame is not None else None,
            "alert_after_closure_s": (
                (event.frame - close_frame) / metadata["fps"]
                if event and close_frame is not None else None
            ),
            "lifted_5cm": result["lifted_5cm"] if result else None,
            "settled_placement": result["settled_placement"] if result else None,
            "reason": event.reason if event else None,
        }
        robot.close()
        return row


def summarize(rows):
    unique = {}
    for row in rows:
        unique.setdefault(row["trace_sha256"], row)
    rows = list(unique.values())
    labeled = [r for r in rows if r["outcome"] in ("success", "failure_no_lift", "lifted_not_placed")]
    failures = [r for r in labeled if r["outcome"] != "success"]
    successes = [r for r in labeled if r["outcome"] == "success"]
    alerts = [r for r in labeled if r["monitor"] == "suspected_miss"]
    # Unknown and no-event states are abstentions: the monitor does not verify success.
    detected_failures = sum(r["monitor"] == "suspected_miss" for r in failures)
    false_alarms = sum(r["monitor"] == "suspected_miss" for r in successes)
    return {
        "unique_labeled_trials": len(labeled),
        "failed_trials": len(failures),
        "successful_trials": len(successes),
        "suspected_miss_alerts": len(alerts),
        "failure_detection_fraction": detected_failures / len(failures) if failures else None,
        "false_alarms_on_success": false_alarms,
        "abstentions": sum(r["monitor"] != "suspected_miss" for r in labeled),
        "interpretation": "Exploratory only; small, scene-specific sample. Abstentions are not counted as correct success predictions.",
        "traces": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, help="Write aggregate metrics and trace rows as JSON")
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    report = summarize([evaluate_trace(path, args.tutorial) for path in args.traces])
    output = json.dumps(report, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
