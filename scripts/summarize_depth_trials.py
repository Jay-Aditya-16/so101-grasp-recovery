"""Score logged two-view localization against simulator-only reference positions."""
import argparse
import hashlib
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
            metadata = json.loads(data["metadata"].item())
            checks = []
            for check in metadata.get("depth_checks", []):
                if "wrist_xy_m" not in check:
                    continue
                index = int(np.argmin(np.abs(data["time"] - check["time_s"])))
                actual = data["cube_xyz"][index, :2]
                checks.append({"frame": index, "timestamp_s": float(data["time"][index]),
                               "ground_truth_xy_m": actual.tolist(),
                               "wrist_error_mm": float(np.linalg.norm(np.asarray(check["wrist_xy_m"]) - actual) * 1000),
                               "overhead_error_mm": float(np.linalg.norm(np.asarray(check["overhead_xy_m"]) - actual) * 1000),
                               "depth": check})
            rows.append({"trace": str(path), "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "transitions": int(len(data["action"])), "observations": int(len(data["time"])),
                         "result": metadata["result"], "checks": checks})
    report = {"scope": "simulator-only reference; observations scored at wrist depth decision time", "trials": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
