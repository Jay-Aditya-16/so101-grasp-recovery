"""Evaluate geometry-to-IK proposals on saved observations without executing motion."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from failure_monitor import GraspMonitor
from object_geometry import estimate_object
from phase2_planner import inspect_path, propose_approach


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tutorial.resolve()))
    from common.robot import SO101Sim

    rows = []
    for path in args.traces:
        with np.load(path, allow_pickle=False) as trace:
            metadata = json.loads(trace["metadata"].item())
            robot = SO101Sim(cube_pos=metadata["start_xy"], box_pos=(0.05, 0.22), render=False)
            cam = robot.model.camera("top").id
            monitor = GraspMonitor()
            alert = None
            for i, action in enumerate(trace["action"]):
                robot.data.qpos[:] = trace["qpos"][i]
                mujoco.mj_forward(robot.model, robot.data)
                event = monitor.update(trace["top_rgb"][i], float(action[5]), float(robot.data.site_xpos[robot.site_id][2]))
                if event:
                    alert = i
                    break
            for label, index in (("initial", 0), ("alert", alert)):
                row = {"trace": str(path), "observation": label, "frame": index}
                if index is None:
                    row["status"] = "no_event"
                else:
                    robot.data.qpos[:] = trace["qpos"][index]
                    mujoco.mj_forward(robot.model, robot.data)
                    estimate = estimate_object(trace["top_rgb"][index], robot.data.cam_xpos[cam],
                                               robot.data.cam_xmat[cam], robot.model.cam_fovy[cam])
                    if estimate is None:
                        row.update(status="abstain", reason="object_not_reliably_visible")
                        # Inspect a known home pose only as a diagnostic. Never execute it.
                        home = robot.model.key_qpos[0, robot.arm_qadr].copy()
                        row["home_view_diagnostic"] = inspect_path(robot, robot.q_arm, home)
                    else:
                        row["object_estimate"] = estimate
                        row["proposal"] = propose_approach(robot, estimate["world_xy_m"])
                        row["status"] = row["proposal"]["status"]
                rows.append(row)
            robot.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"scope": "offline kinematic candidate inspection; no motion executed", "trials": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
