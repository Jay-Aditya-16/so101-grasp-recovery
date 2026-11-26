"""Inspect relative monocular depth on recorded SO-101 overhead images.

The model predicts relative depth, not metres. This script does not control the arm.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from failure_monitor import GraspMonitor

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--frame", choices=("initial", "alert"), default="initial")
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tutorial.resolve()))
    from common.robot import SO101Sim
    processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL, revision=REVISION).eval()
    torch.set_num_threads(2)
    rows = []
    for path in args.traces:
        with np.load(path, allow_pickle=False) as trace:
            metadata = json.loads(trace["metadata"].item())
            frame = 0
            if args.frame == "alert":
                robot = SO101Sim(cube_pos=metadata["start_xy"], box_pos=(0.05, 0.22), render=False)
                monitor = GraspMonitor()
                frame = None
                for index, action in enumerate(trace["action"]):
                    robot.data.qpos[:6] = trace["qpos"][index, :6]
                    mujoco.mj_kinematics(robot.model, robot.data)
                    if monitor.update(trace["top_rgb"][index], float(action[5]), float(robot.data.site_xpos[robot.site_id][2])):
                        frame = index
                        break
                robot.close()
                if frame is None:
                    rows.append({"trace": str(path), "status": "no_monitor_event"})
                    continue
            rgb = trace["top_rgb"][frame]
            qpos = trace["qpos"][frame].copy()
        inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
        start = time.perf_counter()
        with torch.inference_mode():
            predicted = model(**inputs).predicted_depth
            depth = F.interpolate(predicted[:, None], size=rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0].numpy()
        elapsed = time.perf_counter() - start
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = (cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (179, 255, 255))) > 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if count < 2:
            rows.append({"trace": str(path), "status": "no_red_object", "inference_s": elapsed})
            continue
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        obj = labels == idx
        ring = cv2.dilate(obj.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & ~obj
        object_value = float(np.median(depth[obj]))
        ring_value = float(np.median(depth[ring]))
        robot = SO101Sim(cube_pos=metadata["start_xy"], box_pos=(0.05, 0.22), render=False)
        robot.data.qpos[:] = qpos
        mujoco.mj_forward(robot.model, robot.data)
        with mujoco.Renderer(robot.model, height=rgb.shape[0], width=rgb.shape[1]) as renderer:
            renderer.enable_depth_rendering()
            renderer.update_scene(robot.data, camera="top")
            true_depth = renderer.render().copy()
        robot.close()
        true_object = float(np.median(true_depth[obj]))
        true_ring = float(np.median(true_depth[ring]))
        rows.append({"trace": str(path), "frame": frame, "status": "ok", "inference_s": elapsed,
                     "object_pixels": int(obj.sum()), "ring_pixels": int(ring.sum()),
                     "object_relative_value": object_value, "nearby_relative_value": ring_value,
                     "object_minus_nearby": object_value - ring_value,
                     "simulator_object_depth_m": true_object, "simulator_nearby_depth_m": true_ring,
                     "simulator_object_minus_nearby_m": true_object - true_ring})
    report = {"model": MODEL, "revision": REVISION, "device": "cpu", "frame": args.frame,
              "interpretation": "relative model scores, not metres; higher predicted score indicates closer for this checkpoint; surrounding pixels may include other objects",
              "trials": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
