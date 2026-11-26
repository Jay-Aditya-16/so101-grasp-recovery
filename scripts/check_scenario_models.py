"""Compile OOD scenes and check the drawer's sliding joint in isolation."""
import argparse
import json
import sys
from pathlib import Path

import mujoco

from scenario_suite import CASES, scenario_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tutorial.resolve()))
    from common.robot import SO101Sim
    rows = []
    for scenario in CASES:
        with scenario_scene(scenario) as kwargs:
            robot = SO101Sim(cube_pos=(0.8, 0.8) if scenario == "drawer" else (0.2, -0.06),
                             box_pos=(0.05, 0.22), render=False, **kwargs)
        geom = robot.model.geom("cube_geom").id
        row = {"scenario": scenario, "target_geom_type": mujoco.mjtGeom(robot.model.geom_type[geom]).name,
               "target_size_m": robot.model.geom_size[geom].tolist(), "target_initial_z_m": float(robot.cube_pos()[2]),
               "green_distractor_present": mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube_green") >= 0}
        if scenario == "drawer":
            joint = robot.model.joint("drawer_slide").id
            dof = int(robot.model.jnt_dofadr[joint])
            before = float(robot.data.joint("drawer_slide").qpos[0])
            robot.data.qfrc_applied[dof] = 0.5
            robot.step(100)
            after = float(robot.data.joint("drawer_slide").qpos[0])
            row.update(drawer_joint_range_m=robot.model.jnt_range[joint].tolist(),
                       isolated_force_N=0.5, isolated_duration_s=100 * robot.model.opt.timestep,
                       drawer_before_m=before, drawer_after_m=after)
            if not after > before:
                raise RuntimeError("Drawer did not move in the open direction under isolated force")
        rows.append(row)
        robot.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
