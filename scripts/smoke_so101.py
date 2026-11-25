"""Check that the upstream SO-101 scene compiles and responds to a joint command."""

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator", type=Path, help="Path to the so101-mujoco checkout")
    args = parser.parse_args()

    source = args.simulator.resolve()
    if not (source / "so101_assets.py").is_file():
        parser.error(f"so101_assets.py not found in {source}")
    sys.path.insert(0, str(source))
    from so101_assets import add_workspace, menagerie_spec

    spec = menagerie_spec()
    add_workspace(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    if model.nu != 6 or model.camera("top_cam").id < 0:
        raise RuntimeError("Unexpected actuator count or missing overhead camera")

    base = model.actuator("Rotation").id
    position = model.jnt_qposadr[model.joint("Rotation").id]
    mujoco.mj_step(model, data, nstep=500)
    initial = float(data.qpos[position])
    data.ctrl[base] = 0.25
    mujoco.mj_step(model, data, nstep=500)
    moved = float(data.qpos[position]) - initial
    if not np.isfinite(data.qpos).all() or moved < 0.1:
        raise RuntimeError(f"Unexpected response to base joint command: {moved:.3f} rad")
    print(f"MuJoCo {mujoco.__version__}: {model.nu} actuators, {model.ncam} camera, "
          f"{data.time:.1f} s simulated, base moved {moved:.3f} rad")


if __name__ == "__main__":
    main()
