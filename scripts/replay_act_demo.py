"""Replay saved 20 Hz joint targets in the live tutorial MuJoCo scene."""

import argparse
import json
from pathlib import Path
import sys

import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq


def load_demo(dataset_root, episode):
    """Load a reference trajectory and reconstruct its initial cube position."""
    from ch06.record_demos import FPS, IMG_H, IMG_W
    from common.robot import SO101Sim
    from common.vision import SimCamera, find_color_center
    from lerobot.datasets import LeRobotDataset

    root = dataset_root.resolve()
    dataset = LeRobotDataset("caspar/so101_pickplace_sim_150", root=root, video_backend="pyav")
    if not 0 <= episode < dataset.num_episodes:
        raise ValueError("Episode is outside the dataset")
    if dataset.fps != FPS:
        raise RuntimeError("The recorded and simulator control rates differ")
    rows = pq.read_table(
        root / "data",
        columns=["index", "action", "observation.state", "timestamp"],
        filters=[("episode_index", "=", episode)],
    ).sort_by("index")
    actions = np.array(rows["action"].to_pylist(), dtype=np.float64)
    states = np.array(rows["observation.state"].to_pylist(), dtype=np.float64)
    timestamps = rows["timestamp"].to_numpy()
    if actions.shape != (len(timestamps), 6) or states.shape != actions.shape:
        raise RuntimeError("The demonstration must store six targets and six measured joints per frame")
    if not np.isfinite(actions).all() or not np.allclose(
        timestamps, np.arange(len(actions)) / FPS, atol=1e-5
    ):
        raise RuntimeError("The recorded actions are invalid or the timeline has missing samples")

    start = dataset[rows["index"][0].as_py()]["observation.images.top"]
    rgb = (start.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    red = find_color_center(rgb, "red")
    blue = find_color_center(rgb, "blue")
    if red is None or blue is None:
        raise RuntimeError("Could not find the red cube and blue box in the first frame")

    box_xy = (0.05, 0.22)
    template = SO101Sim(box_pos=box_xy, render=False)
    camera = SimCamera(template.model, "top", IMG_W, IMG_H)
    try:
        reference, _ = camera.capture(template.data)
        reference_box = find_color_center(reference, "blue")
        if reference_box is None:
            raise RuntimeError("Could not locate the box in the simulator's top camera")
        box_alignment = np.linalg.norm(np.subtract(blue["center"], reference_box["center"]))
        if box_alignment > 3:
            raise RuntimeError(f"Top camera box alignment differs by {box_alignment:.1f} pixels")
        cube = template.model.geom("cube_geom")
        top_height = template.cube_pos()[2] + cube.size[2]
        depth = template.data.cam_xpos[camera.cam_id][2] - top_height
        cube_xy = camera.pixel_to_world(template.data, *red["center"], depth)[:2]
    finally:
        camera.close()
        template.close()

    return cube_xy, actions, states, box_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.tutorial.resolve()))
    from ch06.record_demos import FPS
    from common.robot import SO101Sim
    from live_act_eval import rollout

    cube_xy, actions, states, box_alignment = load_demo(args.dataset, args.episode)
    robot = SO101Sim(cube_pos=cube_xy, box_pos=box_xy, render=False)
    if not np.allclose(robot.data.qpos[:6], states[0], atol=1e-4):
        raise RuntimeError("The demonstration and simulator start from different joint poses")
    print(f"Episode {args.episode}: {len(actions)} targets, box alignment {box_alignment:.2f} px. Opening viewer.", flush=True)
    frame = 0
    joint_errors = []

    def get_action():
        nonlocal frame
        index = min(frame, len(actions) - 1)
        if frame < len(actions):
            joint_errors.append(robot.data.qpos[:6] - states[frame])
        frame += 1
        return actions[index]

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = robot.model.camera("fixed").id
        result = rollout(robot, viewer, get_action, FPS, len(actions) / FPS + 0.75)
    if result is None:
        print("Viewer closed. Replay stopped without a score.", flush=True)
        return
    result.update(
        dataset_episode=args.episode,
        start_xy=cube_xy.tolist(),
        frames_replayed=min(frame, len(actions)),
        mean_abs_joint_error_rad=round(float(np.mean(np.abs(joint_errors))), 5),
        final_xyz=robot.cube_pos().tolist(),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
