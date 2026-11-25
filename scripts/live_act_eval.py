"""Run a Colab ACT checkpoint in the tutorial scene with a live MuJoCo viewer."""

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


def rollout(robot, viewer, get_action, fps, seconds, on_step=None):
    cube_geom = robot.model.geom("cube_geom").id
    # The first box geom is the floor in the pinned tutorial scene.
    box_floor = int(robot.model.body_geomadr[robot.box_id])
    jaw_bodies = {robot.model.body(name).id for name in ("gripper", "moving_jaw_so101_v1")}
    physics_steps = round(1 / (fps * robot.model.opt.timestep))
    start_height = float(robot.cube_pos()[2])
    peak_height = start_height
    settled_frames = 0
    placed = False
    bilateral_contact = False

    viewer.sync()
    time.sleep(2)
    for _ in range(round(seconds * fps)):
        if not viewer.is_running():
            return None
        started = time.monotonic()
        robot.data.ctrl[:6] = get_action()
        robot.step(physics_steps)
        viewer.sync()

        cube_position = robot.cube_pos()
        peak_height = max(peak_height, float(cube_position[2]))
        distance = np.abs(cube_position[:2] - robot.box_pos()[:2])
        cube_contacts = set()
        for contact in robot.data.contact:
            if contact.geom1 == cube_geom:
                cube_contacts.add(int(contact.geom2))
            elif contact.geom2 == cube_geom:
                cube_contacts.add(int(contact.geom1))
        touching_both = jaw_bodies <= {int(robot.model.geom_bodyid[geom]) for geom in cube_contacts}
        bilateral_contact |= touching_both
        supported = box_floor in cube_contacts
        stable = supported and np.all(distance <= 0.03) and np.linalg.norm(
            robot.data.joint("cube_free").qvel[:3]
        ) < 0.03
        settled_frames = settled_frames + 1 if stable else 0
        placed = settled_frames >= round(0.5 * fps) and peak_height - start_height >= 0.05
        if on_step is not None:
            on_step(touching_both, supported, stable, placed)
        if placed:
            break
        time.sleep(max(0, 1 / fps - (time.monotonic() - started)))

    return {
        "bilateral_gripper_contact": bilateral_contact,
        "lifted_5cm": peak_height - start_height >= 0.05,
        "settled_placement": placed,
        "max_lift_m": round(peak_height - start_height, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scene-check", action="store_true", help="Check cameras without moving the arm")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--demo-dataset", type=Path, help="Reconstruct the starting pose from this local dataset")
    parser.add_argument("--demo-episode", type=int, default=0)
    parser.add_argument("--capture", type=Path, help="Save one rollout as a compressed NumPy archive")
    parser.add_argument("--monitor", action="store_true", help="Report suspected missed grasps during execution")
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds <= 0:
        parser.error("--episodes and --seconds must be positive")
    if args.capture and (args.episodes != 1 or args.scene_check or args.capture.exists()):
        parser.error("--capture requires one rollout and a new output path")
    if not args.scene_check and (
        args.checkpoint is None or not (args.checkpoint / "model.safetensors").is_file()
    ):
        parser.error("pass --checkpoint pointing to the extracted pretrained_model folder")

    tutorial = args.tutorial.resolve()
    if not (tutorial / "ch06" / "record_demos.py").is_file():
        parser.error("--tutorial must point to the Physical AI tutorial checkout")
    sys.path.insert(0, str(tutorial))
    from ch06.record_demos import FPS, IMG_H, IMG_W, TASK, random_cube
    from common.robot import SO101Sim

    camera_keys = {"top": "observation.images.top", "wrist_cam": "observation.images.wrist"}
    rng = np.random.default_rng(args.seed)
    reference = None
    if args.demo_dataset:
        from replay_act_demo import load_demo

        reference = load_demo(args.demo_dataset, args.demo_episode)
    if not args.scene_check:
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        config = ACTConfig.from_pretrained(args.checkpoint, cli_overrides=["--device=cpu"])
        config.pretrained_backbone_weights = None
        policy = ACTPolicy.from_pretrained(args.checkpoint, config=config, strict=True).eval()
        preprocess, postprocess = make_pre_post_processors(
            policy.config, str(args.checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    for episode in range(args.episodes):
        if args.monitor:
            from failure_monitor import GraspMonitor

            monitor = GraspMonitor()
            reported = False
        cube = reference[0].tolist() if reference is not None else random_cube(rng)
        robot = SO101Sim(cube_pos=cube, box_pos=(0.05, 0.22), render=False)
        if reference is not None and not np.allclose(robot.data.qpos[:6], reference[2][0], atol=1e-4):
            raise RuntimeError("The recorded and simulated initial joint positions differ")
        capture = {key: [] for key in ("time", "qpos", "qvel", "cube_xyz", "top_rgb", "wrist_rgb",
                                       "action", "bilateral_contact", "box_support", "settled", "reward")}
        with mujoco.Renderer(robot.model, height=IMG_H, width=IMG_W) as renderer:
            if args.scene_check:
                for camera in camera_keys:
                    renderer.update_scene(robot.data, camera=camera)
                    print(f"{camera}: {renderer.render().shape}")
                return

            def observe():
                obs = {
                    "observation.state": torch.tensor(robot.data.qpos[:6], dtype=torch.float32)[None],
                    "task": [TASK],
                }
                for camera, key in camera_keys.items():
                    renderer.update_scene(robot.data, camera=camera)
                    frame = renderer.render().copy()
                    if args.capture:
                        capture["top_rgb" if camera == "top" else "wrist_rgb"].append(frame)
                    rgb = torch.from_numpy(frame)
                    obs[key] = rgb.permute(2, 0, 1)[None].float() / 255
                if args.capture:
                    capture["time"].append(robot.data.time)
                    capture["qpos"].append(robot.data.qpos.copy())
                    capture["qvel"].append(robot.data.qvel.copy())
                    capture["cube_xyz"].append(robot.cube_pos())
                return obs

            def get_action():
                nonlocal reported
                obs = observe()
                with torch.inference_mode():
                    action = postprocess(policy.select_action(preprocess(obs)))[0].cpu().numpy()
                if action.shape != (6,) or not np.isfinite(action).all():
                    raise RuntimeError("ACT returned an invalid six-joint target")
                if args.monitor:
                    image = (obs["observation.images.top"][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    event = monitor.update(image, float(action[5]), float(robot.data.site_xpos[robot.site_id][2]))
                    if event and not reported:
                        print(f"Monitor at {robot.data.time:.2f}s: {event.status} — {event.reason}", flush=True)
                        reported = True
                return action

            def capture_step(touching_both, supported, stable, placed):
                capture["action"].append(robot.data.ctrl[:6].copy())
                capture["bilateral_contact"].append(touching_both)
                capture["box_support"].append(supported)
                capture["settled"].append(stable)
                capture["reward"].append(float(placed))

            policy.reset()
            with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = robot.model.camera("fixed").id
                result = rollout(robot, viewer, get_action, FPS, args.seconds,
                                 on_step=capture_step if args.capture else None)
                if args.capture and capture["action"]:
                    observe()
        if args.capture and capture["action"]:
            n = len(capture["action"])
            terminated = np.zeros(n, dtype=bool)
            terminated[-1] = bool(result and result["settled_placement"])
            truncated = np.zeros(n, dtype=bool)
            truncated[-1] = not terminated[-1]
            metadata = {"checkpoint": str(args.checkpoint), "fps": FPS, "start_xy": cube,
                        "reference_episode": args.demo_episode if reference is not None else None,
                        "viewer_closed": result is None, "result": result}
            if reference is not None:
                capture["reference_action"] = reference[1]
                capture["reference_joint_state"] = reference[2]
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            with args.capture.open("xb") as output:
                np.savez_compressed(output, **capture, metadata=json.dumps(metadata),
                                    terminated=terminated, truncated=truncated)
            print(f"Saved {n} transitions to {args.capture}", flush=True)
        if result is None:
            print("Viewer closed. Episode stopped without a score.")
            break
        print(json.dumps({"episode": episode + 1, "seed": args.seed, "start_xy": cube, **result}), flush=True)


if __name__ == "__main__":
    main()
