"""Extract simulator-only, per-recovery grasp-stage proxies from rollout captures.

Privileged cube positions and contacts are used for offline labels only. A label
is censored unless the controller reached its after-lift observation stage.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def extract(meta, time, xyz, contacts):
    time = np.asarray(time, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    contacts = np.asarray(contacts, dtype=bool)
    if time.ndim != 1 or len(time) < 2 or xyz.shape != (len(time), 3) or len(contacts) != len(time) - 1:
        raise ValueError("misaligned transition and observation arrays")
    if not np.isfinite(time).all() or not np.isfinite(xyz).all() or np.any(np.diff(time) <= 0):
        raise ValueError("invalid timestamps or object positions")
    events = meta.get("recovery_events", [])
    closing = next((e for e in events if e.get("state") == "closing"), None)
    if closing is None:
        return {"status": "no_recovery_grasp", "label": None}
    checks = [e for e in events if e.get("state") == "gripper_check" and e.get("reason") == "after_lift" and e["time_s"] >= closing["time_s"]]
    if meta.get("viewer_closed") or not checks:
        return {"status": "censored_before_lift_check", "label": None}
    start = int(np.searchsorted(time, closing["time_s"], side="left"))
    end = int(np.searchsorted(time, checks[0]["time_s"], side="right"))
    if start >= len(time) or end <= start or checks[0]["time_s"] > time[-1] + 1e-5:
        raise ValueError("events fall outside recorded observations")
    lift = float(np.max(xyz[start:end, 2]) - xyz[start, 2])
    bilateral = bool(np.any(contacts[start:min(end, len(contacts))]))
    return {"status": "evaluated_lift_proxy", "label": int(lift >= .05 and bilateral),
            "peak_lift_since_closing_m": lift, "bilateral_contact_seen": bilateral,
            "closing_time_s": closing["time_s"], "lift_check_time_s": checks[0]["time_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("captures", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.captures:
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            row = extract(meta, data["time"], data["cube_xyz"], data["bilateral_contact"])
        row.update(source=str(path), sha256=digest, seed=meta.get("seed"), scenario=meta.get("scenario", "cube"),
                   task_success=meta.get("result", {}).get("task_success") if meta.get("result") else None,
                   label_source="privileged MuJoCo state and contact; offline only")
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
