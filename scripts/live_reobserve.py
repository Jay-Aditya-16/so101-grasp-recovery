"""Explore a simulated return-to-view from a captured suspected miss.

Not a physical robot controller; sampled paths and a single rollout are not safety proof.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from failure_monitor import GraspMonitor
from object_geometry import estimate_object
from phase2_planner import inspect_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tutorial.resolve()))
    from common.robot import SO101Sim
    with np.load(args.trace, allow_pickle=False) as data:
        meta = json.loads(data["metadata"].item())
        robot = SO101Sim(cube_pos=meta["start_xy"], box_pos=(0.05, 0.22), render=False)
        monitor = GraspMonitor()
        selected = None
        for i, action in enumerate(data["action"]):
            robot.data.qpos[:] = data["qpos"][i]
            robot.data.qvel[:] = data["qvel"][i]
            mujoco.mj_forward(robot.model, robot.data)
            event = monitor.update(data["top_rgb"][i], float(action[5]), float(robot.data.site_xpos[robot.site_id][2]))
            if event is not None:
                if event.status != "suspected_miss":
                    raise RuntimeError("Monitor did not report suspected_miss; return-to-view is not attempted")
                selected = i
                robot.data.ctrl[:6] = action
                break
        if selected is None:
            raise RuntimeError("No monitor event")
    initial_cube = robot.cube_pos().copy()  # privileged scoring only
    home = robot.model.key_qpos[0, robot.arm_qadr].copy()
    start = robot.q_arm.copy()
    path = inspect_path(robot, start, home)
    if path["first_penetration"] is not None:
        raise RuntimeError("Sampled home trajectory intersects an object")
    print(f"Opening viewer: captured miss at frame {selected}, returning to view.", flush=True)
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = robot.model.camera("fixed").id
        viewer.sync()
        time.sleep(2)
        completed = True
        for frame in range(80):
            if not viewer.is_running():
                completed = False
                break
            started = time.monotonic()
            alpha = (frame + 1) / 80
            robot.data.ctrl[robot.arm_act] = start * (1 - alpha) + home * alpha
            robot.step(round(0.05 / robot.model.opt.timestep))
            viewer.sync()
            time.sleep(max(0, 0.05 - (time.monotonic() - started)))
    if not completed:
        raise RuntimeError("Viewer closed; reobservation did not complete")
    with mujoco.Renderer(robot.model, height=240, width=320) as renderer:
        renderer.update_scene(robot.data, camera="top")
        rgb = renderer.render().copy()
    cam = robot.model.camera("top").id
    estimate = estimate_object(rgb, robot.data.cam_xpos[cam], robot.data.cam_xmat[cam], robot.model.cam_fovy[cam])
    final_cube = robot.cube_pos().copy()  # privileged scoring only
    result = {"trace": str(args.trace), "alert_frame": selected, "path": path,
              "estimate": estimate, "actual_object_displacement_m": float(np.linalg.norm(final_cube - initial_cube)),
              "localization_error_mm": float(np.linalg.norm(np.asarray(estimate["world_xy_m"]) - final_cube[:2]) * 1000) if estimate else None,
              "scope": "one simulator-only reobservation; no grasp or placement attempted"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    robot.close()


if __name__ == "__main__":
    main()
