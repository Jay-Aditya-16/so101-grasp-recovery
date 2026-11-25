"""Evaluate monitor alerts on saved ACT traces using simulator outcomes as labels."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from failure_monitor import GraspMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tutorial.resolve()))
    from common.robot import SO101Sim

    rows = []
    for path in args.traces:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(data["metadata"].item())
            robot = SO101Sim(cube_pos=metadata["start_xy"], box_pos=(0.05, 0.22), render=False)
            monitor = GraspMonitor()
            event = None
            for index, action in enumerate(data["action"]):
                robot.data.qpos[:6] = data["qpos"][index, :6]
                mujoco.mj_kinematics(robot.model, robot.data)
                event = monitor.update(
                    data["top_rgb"][index], float(action[5]),
                    float(robot.data.site_xpos[robot.site_id][2]),
                )
                if event is not None:
                    break
            result = metadata["result"]
            rows.append({
                "trace": str(path),
                "lifted_5cm": result["lifted_5cm"] if result else None,
                "settled_placement": result["settled_placement"] if result else None,
                "monitor": event.status if event else "no_event",
                "event_time_s": event.frame / metadata["fps"] if event else None,
                "reason": event.reason if event else None,
            })
            robot.close()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
