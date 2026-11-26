"""Evaluate RGB-based object localization on recorded simulator observations."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from object_geometry import estimate_object
from failure_monitor import GraspMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", choices=("initial", "alert"), default="initial")
    parser.add_argument("--camera", choices=("top", "wrist"), default="top")
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tutorial.resolve()))
    from common.robot import SO101Sim

    rows = []
    for path in args.traces:
        with np.load(path, allow_pickle=False) as trace:
            meta = json.loads(trace["metadata"].item())
            robot = SO101Sim(cube_pos=meta["start_xy"], box_pos=(0.05, 0.22), render=False)
            cam = robot.model.camera("top" if args.camera == "top" else "wrist_cam").id
            mujoco.mj_forward(robot.model, robot.data)
            frame = 0
            if args.frame == "alert":
                monitor = GraspMonitor()
                frame = None
                for index, action in enumerate(trace["action"]):
                    robot.data.qpos[:6] = trace["qpos"][index, :6]
                    mujoco.mj_kinematics(robot.model, robot.data)
                    if monitor.update(trace["top_rgb"][index], float(action[5]),
                                      float(robot.data.site_xpos[robot.site_id][2])):
                        frame = index
                        break
                if frame is None:
                    rows.append({"trace": str(path), "frame": None, "reason": "no monitor event"})
                    robot.close()
                    continue
            robot.data.qpos[:] = trace["qpos"][frame]
            mujoco.mj_forward(robot.model, robot.data)
            rgb = trace["top_rgb" if args.camera == "top" else "wrist_rgb"][frame]
            result = estimate_object(rgb, robot.data.cam_xpos[cam], robot.data.cam_xmat[cam], robot.model.cam_fovy[cam])
            robot.data.qpos[:6] = trace["qpos"][frame, :6]
            mujoco.mj_kinematics(robot.model, robot.data)
            gripper = robot.data.site_xpos[robot.site_id].copy()
            # Privileged cube position is used only for scoring, after the estimate.
            actual_xy = trace["cube_xyz"][frame, :2]
            row = {"trace": str(path), "frame": frame, "estimate": result, "gripper_world_xyz_m": gripper.tolist()}
            if result is not None:
                estimate = np.asarray(result["world_xy_m"])
                row["object_minus_gripper_xy_m"] = (estimate - gripper[:2]).tolist()
                row["ground_truth_xy_m"] = actual_xy.tolist()
                row["localization_error_mm"] = float(np.linalg.norm(estimate - actual_xy) * 1000)
            rows.append(row)
            robot.close()
    report = {"scope": f"{args.frame} {args.camera} observations; red cube and assumed 30 mm height", "trials": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
