"""Run a Colab ACT checkpoint in the tutorial scene with a live MuJoCo viewer."""

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

import mujoco
import mujoco.viewer
import numpy as np


def placement_frame_supported(supported, jaw_contact, distance_xy, linear_speed):
    """A settled frame requires box support, released jaws, centrality and low speed."""
    return bool(supported and not jaw_contact and np.all(np.abs(distance_xy) <= 0.03) and linear_speed < 0.03)


def drawer_success_step(displacement_m, open_frames, fps):
    """Require the drawer to remain open at least 60 mm for half a second."""
    count = open_frames + 1 if displacement_m >= 0.06 else 0
    return count, count >= round(0.5 * fps)


def object_inside_box_xy(geom_type, geom_size, geom_rotation, object_xy, box_xy):
    """Score full projected footprint inside the tutorial box's 47 mm inner half-width."""
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.full(2, float(geom_size[0]))
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        extent = (np.abs(np.asarray(geom_rotation).reshape(3, 3)) @ np.asarray(geom_size))[:2]
    else:
        return False
    return bool(np.all(np.abs(np.asarray(object_xy) - np.asarray(box_xy)) + extent <= 0.047))


def rollout(robot, viewer, get_action, fps, seconds, on_step=None, task="pick_place", scenario="cube"):
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
    max_drawer = 0.0
    drawer_open_frames = 0
    drawer_opened = False
    distractor_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube_green")
    distractor_start = robot.data.xpos[distractor_id].copy() if distractor_id >= 0 else None
    distractor_max_motion = 0.0

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
        if distractor_start is not None:
            distractor_max_motion = max(distractor_max_motion, float(np.linalg.norm(robot.data.xpos[distractor_id] - distractor_start)))
        peak_height = max(peak_height, float(cube_position[2]))
        distance = np.abs(cube_position[:2] - robot.box_pos()[:2])
        cube_contacts = set()
        for contact in robot.data.contact:
            if contact.geom1 == cube_geom:
                cube_contacts.add(int(contact.geom2))
            elif contact.geom2 == cube_geom:
                cube_contacts.add(int(contact.geom1))
        contact_bodies = {int(robot.model.geom_bodyid[geom]) for geom in cube_contacts}
        touching_both = jaw_bodies <= contact_bodies
        bilateral_contact |= touching_both
        supported = box_floor in cube_contacts
        stable = placement_frame_supported(supported, bool(jaw_bodies & contact_bodies), distance,
                                           np.linalg.norm(robot.data.joint("cube_free").qvel[:3]))
        stable = stable and object_inside_box_xy(robot.model.geom_type[cube_geom], robot.model.geom_size[cube_geom],
                                                 robot.data.geom_xmat[cube_geom], robot.data.geom_xpos[cube_geom, :2],
                                                 robot.box_pos()[:2])
        settled_frames = settled_frames + 1 if stable else 0
        placed = settled_frames >= round(0.5 * fps) and peak_height - start_height >= 0.05
        if task == "drawer":
            drawer_q = float(robot.data.joint("drawer_slide").qpos[0])
            max_drawer = max(max_drawer, drawer_q)
            drawer_open_frames, drawer_opened = drawer_success_step(drawer_q, drawer_open_frames, fps)
        task_success = drawer_opened if task == "drawer" else placed
        if on_step is not None:
            on_step(touching_both, supported, stable, task_success)
        if task_success:
            break
        time.sleep(max(0, 1 / fps - (time.monotonic() - started)))

    result = {
        "bilateral_gripper_contact": bilateral_contact,
        "lifted_5cm": peak_height - start_height >= 0.05,
        "settled_placement": placed,
        "max_lift_m": round(peak_height - start_height, 4),
        "task": task,
        "task_success": drawer_opened if task == "drawer" else placed,
    }
    if task == "drawer":
        result.update(drawer_opened_6cm=drawer_opened, max_drawer_open_m=round(max_drawer, 5))
    if scenario == "two_objects":
        result["distractor_max_displacement_m"] = round(distractor_max_motion, 6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scene-check", action="store_true", help="Check cameras without moving the arm")
    from scenario_suite import CASES
    parser.add_argument("--scenario", choices=CASES, default="cube", help="Simulation test case")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--demo-dataset", type=Path, help="Reconstruct the starting pose from this local dataset")
    parser.add_argument("--demo-episode", type=int, default=0)
    parser.add_argument("--capture", type=Path, help="Save one rollout as a compressed NumPy archive")
    parser.add_argument("--capture-dir", type=Path, default=Path("datasets/local-rollouts"), help="Directory for automatically saved trial archives")
    parser.add_argument("--no-capture", action="store_true", help="Do not automatically save a rollout archive")
    parser.add_argument("--monitor", action="store_true", help="Report suspected missed grasps during execution")
    parser.add_argument("--recovery", action="store_true", help="Enable experimental simulator-only recovery on suspected misses")
    parser.add_argument("--depth-gate", action="store_true", help="Use Depth Anything V2 relative-depth check during recovery")
    parser.add_argument("--depth-view", choices=("top", "wrist"), default="top", help="View used by the optional depth gate")
    parser.add_argument("--recover-unknown-empty", action="store_true", help="Experimental empty-gripper fallback for unknown alerts")
    parser.add_argument("--recover-unknown-stationary", action="store_true", help="Experimental fresh stationary-cube fallback for unknown alerts")
    parser.add_argument("--retry-only", action="store_true", help="After return-to-view, reset ACT instead of using object-aware recovery")
    parser.add_argument("--results", type=Path, help="Save episode results as JSON")
    args = parser.parse_args()
    if args.depth_gate and not args.recovery:
        parser.error("--depth-gate requires --recovery")
    if args.recover_unknown_empty and not args.recovery:
        parser.error("--recover-unknown-empty requires --recovery")
    if args.recover_unknown_stationary and not args.recovery:
        parser.error("--recover-unknown-stationary requires --recovery")
    if args.retry_only and (not args.recovery or args.depth_gate):
        parser.error("--retry-only requires --recovery and cannot use --depth-gate")
    if args.recovery:
        args.monitor = True
    if args.demo_dataset and args.scenario != "cube":
        parser.error("recorded demonstrations are only compatible with the original cube scene")
    if args.episodes < 1 or args.seconds <= 0:
        parser.error("--episodes and --seconds must be positive")
    if args.capture and (args.episodes != 1 or args.scene_check or args.capture.exists()):
        parser.error("--capture requires one rollout and a new output path")
    if args.capture and args.no_capture:
        parser.error("--capture and --no-capture cannot be combined")
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
    from scenario_suite import scenario_scene

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

    depth_gate = None
    if args.depth_gate:
        from depth_gate import DepthAnythingGate
        print("Loading Depth Anything V2 Small for recovery.", flush=True)
        depth_gate = DepthAnythingGate()
    episode_results = []
    for episode in range(args.episodes):
        capture_path = args.capture
        if not args.scene_check and capture_path is None and not args.no_capture:
            capture_path = args.capture_dir / f"rollout-seed{args.seed}-episode{episode + 1}-{uuid.uuid4().hex}.npz"
        if args.monitor:
            from failure_monitor import GraspMonitor, empty_closed_gripper, stationary_cube_after_unknown

            monitor = GraspMonitor()
            reported = False
            empty_evidence_frames = 0
            stationary_evidence_frames = 0
        cube = reference[0].tolist() if reference is not None else random_cube(rng)
        if args.scenario == "drawer":
            cube = [0.8, 0.8]  # hidden placeholder for the unchanged policy interface
        with scenario_scene(args.scenario) as scene_kwargs:
            robot = SO101Sim(cube_pos=cube, box_pos=(0.05, 0.22), render=False, **scene_kwargs)
        if args.recovery:
            from phase2_controller import RecoveryController
            recovery = RecoveryController(robot, FPS, depth_gate=depth_gate, depth_view=args.depth_view,
                                          retry_only=args.retry_only)
        if reference is not None and not np.allclose(robot.data.qpos[:6], reference[2][0], atol=1e-4):
            raise RuntimeError("The recorded and simulated initial joint positions differ")
        capture = {key: [] for key in ("time", "qpos", "qvel", "cube_xyz", "top_rgb", "wrist_rgb",
                                       "action", "bilateral_contact", "box_support", "settled", "reward")}
        if args.scenario == "drawer":
            capture["drawer_position"] = []
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
                    if capture_path:
                        capture["top_rgb" if camera == "top" else "wrist_rgb"].append(frame)
                    rgb = torch.from_numpy(frame)
                    obs[key] = rgb.permute(2, 0, 1)[None].float() / 255
                if capture_path:
                    capture["time"].append(robot.data.time)
                    capture["qpos"].append(robot.data.qpos.copy())
                    capture["qvel"].append(robot.data.qvel.copy())
                    capture["cube_xyz"].append(robot.cube_pos())
                    if args.scenario == "drawer":
                        capture["drawer_position"].append(float(robot.data.joint("drawer_slide").qpos[0]))
                return obs

            def get_action():
                nonlocal reported, empty_evidence_frames, stationary_evidence_frames
                obs = observe()
                image = (obs["observation.images.top"][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                if args.recovery and recovery.state == "policy_retry":
                    policy.reset()
                    recovery.log("retrying", "ACT resumed; further interventions disabled")
                if args.recovery and recovery.state not in ("idle", "retrying"):
                    wrist = (obs["observation.images.wrist"][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    return recovery.step(image, wrist)
                with torch.inference_mode():
                    action = postprocess(policy.select_action(preprocess(obs)))[0].cpu().numpy()
                if action.shape != (6,) or not np.isfinite(action).all():
                    raise RuntimeError("ACT returned an invalid six-joint target")
                if args.monitor:
                    event = monitor.update(image, float(action[5]), float(robot.data.site_xpos[robot.site_id][2]))
                    if event and not reported:
                        print(f"Monitor at {robot.data.time:.2f}s: {event.status} — {event.reason}", flush=True)
                        reported = True
                        if args.recovery and event.status == "suspected_miss":
                            recovery.start()
                            return recovery.step(image)
                    if args.recovery and recovery.state == "idle" and args.recover_unknown_empty and event and event.status == "unknown":
                        empty = empty_closed_gripper(float(action[5]), robot.gripper_opening())
                        empty_evidence_frames = empty_evidence_frames + 1 if empty else 0
                        if empty_evidence_frames >= 3:
                            print("Monitor: unknown + 3 consecutive empty-closure frames; attempting recovery", flush=True)
                            recovery.events.append({"state": "trigger", "time_s": float(robot.data.time),
                                                    "reason": "unknown_empty_gripper"})
                            recovery.start()
                            return recovery.step(image)
                    if args.recovery and recovery.state == "idle" and args.recover_unknown_stationary and event and event.status == "unknown":
                        stationary = stationary_cube_after_unknown(image, monitor.close_center,
                                                                   monitor.frame - 1 - event.frame, FPS)
                        stationary_evidence_frames = stationary_evidence_frames + 1 if stationary else 0
                        if stationary_evidence_frames >= 3:
                            print("Monitor: unknown + 3 consecutive stationary-cube frames; attempting recovery", flush=True)
                            recovery.events.append({"state": "trigger", "time_s": float(robot.data.time),
                                                    "reason": "unknown_stationary_cube"})
                            recovery.start()
                            return recovery.step(image)
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
                                 on_step=capture_step if capture_path else None,
                                 task="drawer" if args.scenario == "drawer" else "pick_place", scenario=args.scenario)
                if capture_path and capture["action"]:
                    observe()
        if capture_path and capture["action"]:
            n = len(capture["action"])
            terminated = np.zeros(n, dtype=bool)
            terminated[-1] = bool(result and result["task_success"])
            truncated = np.zeros(n, dtype=bool)
            truncated[-1] = not terminated[-1]
            metadata = {"checkpoint": str(args.checkpoint), "fps": FPS, "start_xy": cube,
                        "seed": args.seed, "episode": episode + 1,
                        "scenario": args.scenario,
                        "mode": "recovery" if args.recovery else "act",
                        "reference_episode": args.demo_episode if reference is not None else None,
                        "viewer_closed": result is None, "result": result}
            if args.recovery:
                metadata.update(recovery_events=recovery.events, depth_checks=recovery.depth_checks,
                                depth_view=args.depth_view if args.depth_gate else None,
                                recover_unknown_empty=args.recover_unknown_empty,
                                recover_unknown_stationary=args.recover_unknown_stationary, retry_only=args.retry_only)
            if reference is not None:
                capture["reference_action"] = reference[1]
                capture["reference_joint_state"] = reference[2]
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            with capture_path.open("xb") as output:
                np.savez_compressed(output, **capture, metadata=json.dumps(metadata),
                                    terminated=terminated, truncated=truncated)
            print(f"Saved {n} transitions to {capture_path}", flush=True)
        if result is None:
            print("Viewer closed. Episode stopped without a score.")
            break
        summary = {"episode": episode + 1, "seed": args.seed, "scenario": args.scenario, "start_xy": cube, **result}
        summary["capture_path"] = str(capture_path) if capture_path else None
        if args.recovery:
            summary["recovery_events"] = recovery.events
            summary["depth_checks"] = recovery.depth_checks
        episode_results.append(summary)
        print(json.dumps(summary), flush=True)
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(episode_results, indent=2) + "\n")


if __name__ == "__main__":
    main()
