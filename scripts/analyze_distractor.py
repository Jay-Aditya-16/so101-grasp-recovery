"""Measure simulated distractor motion from saved free-joint states."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scenario_suite import scenario_scene


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
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            if meta.get("scenario") != "two_objects":
                raise ValueError(f"{path} is not a two-object trace")
            with scenario_scene("two_objects") as kwargs:
                robot = SO101Sim(cube_pos=meta["start_xy"], box_pos=(0.05, 0.22), render=False, **kwargs)
            joint = robot.model.joint("cube_green_free").id
            address = int(robot.model.jnt_qposadr[joint])
            xyz = data["qpos"][:, address:address + 3]
            distance = np.linalg.norm(xyz - xyz[0], axis=1)
            rows.append({"trace": str(path), "initial_xyz_m": xyz[0].tolist(),
                         "max_displacement_m": float(distance.max()), "final_displacement_m": float(distance[-1]),
                         "note": "Simulator state label; motion includes normal settling and is not a contact detector."})
            robot.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
