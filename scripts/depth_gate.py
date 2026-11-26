"""Relative monocular depth check for a visible red cube; simulation only."""
import time

import cv2
import numpy as np

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


def relative_depth_decision(rgb, depth):
    """Accept when the visible cube scores closer than a surrounding image ring.

    Scores are relative, not metric. This is an experimental binary check, not a
    calibrated confidence or an independent object detector.
    """
    if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all():
        return {"accepted": False, "reason": "invalid_depth"}
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) |
            cv2.inRange(hsv, (170, 120, 70), (179, 255, 255))) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 2:
        return {"accepted": False, "reason": "no_red_object"}
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    obj = labels == idx
    ring = cv2.dilate(obj.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & ~obj
    if obj.sum() < 60 or not ring.any():
        return {"accepted": False, "reason": "insufficient_pixels"}
    contrast = float(np.median(depth[obj]) - np.median(depth[ring]))
    return {"accepted": contrast > 0.0, "reason": "cube_closer" if contrast > 0.0 else "depth_ambiguous",
            "relative_contrast": contrast, "object_pixels": int(obj.sum()), "ring_pixels": int(ring.sum())}


class DepthAnythingGate:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL, revision=REVISION).eval()

    def __call__(self, rgb):
        from PIL import Image
        import torch
        import torch.nn.functional as F
        start = time.perf_counter()
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        with torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
            depth = F.interpolate(prediction[:, None], size=rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
        result = relative_depth_decision(rgb, depth)
        result["wall_time_s"] = time.perf_counter() - start
        result["model"] = MODEL
        result["revision"] = REVISION
        return result
