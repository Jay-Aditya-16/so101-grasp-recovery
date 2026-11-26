"""Export captured transitions for independent MATLAB trace analysis."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.traces:
        with np.load(path, allow_pickle=False) as data:
            n = len(data["action"])
            metadata = json.loads(data["metadata"].item())
            flags = {name: np.zeros(n, dtype=int) for name in ("after_closure", "after_lift", "after_transport")}
            for event in metadata.get("recovery_events", []):
                if event.get("state") == "gripper_check" and event.get("reason") in flags and n:
                    index = int(np.argmin(np.abs(data["time"][:n] - event["time_s"])))
                    flags[event["reason"]][index] = 1
            with (args.output_dir / (path.stem + ".csv")).open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["time_s", "gripper_position", "gripper_target", "cube_z_m", *flags])
                writer.writerows(zip(data["time"][:n], data["qpos"][:n, 5], data["action"][:, 5],
                                     data["cube_xyz"][:n, 2], *flags.values()))


if __name__ == "__main__":
    main()
