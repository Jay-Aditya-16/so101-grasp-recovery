"""Cross-check MuJoCo site translational Jacobian against central differences."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tutorial.resolve()))
    from common.robot import SO101Sim, ARM_JOINTS
    rows = []
    for path in args.traces:
        with np.load(path, allow_pickle=False) as trace:
            metadata = json.loads(trace["metadata"].item())
            robot = SO101Sim(cube_pos=metadata["start_xy"], box_pos=(0.05, 0.22), render=False)
            for frame in (0, len(trace["action"]) // 2):
                robot.data.qpos[:] = trace["qpos"][frame]
                mujoco.mj_forward(robot.model, robot.data)
                jac = np.zeros((3, robot.model.nv))
                mujoco.mj_jacSite(robot.model, robot.data, jac, None, robot.site_id)
                ids = [robot.model.joint(n).id for n in ARM_JOINTS]
                columns = robot.model.jnt_dofadr[ids]
                numerical = np.zeros((3, 5))
                epsilon = 1e-6
                for j, adr in enumerate(robot.arm_qadr):
                    original = robot.data.qpos[adr]
                    robot.data.qpos[adr] = original + epsilon
                    mujoco.mj_forward(robot.model, robot.data)
                    plus = robot.data.site_xpos[robot.site_id].copy()
                    robot.data.qpos[adr] = original - epsilon
                    mujoco.mj_forward(robot.model, robot.data)
                    minus = robot.data.site_xpos[robot.site_id].copy()
                    robot.data.qpos[adr] = original
                    numerical[:, j] = (plus - minus) / (2 * epsilon)
                error = float(np.max(np.abs(jac[:, columns] - numerical)))
                rows.append({"trace": str(path), "frame": frame, "max_jacobian_difference_m_per_rad": error})
            robot.close()
    output = {"method": "central finite difference, epsilon 1e-6 rad", "results": rows,
              "max_difference_m_per_rad": max(row["max_jacobian_difference_m_per_rad"] for row in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
