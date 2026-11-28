"""Known-cube hold cues for offline evaluation; no simulator ground truth input."""
import cv2
import numpy as np


def wrist_red_visible(rgb):
    """Red pixels in the lower center of a wrist image, not object identity."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        return None
    h, w = rgb.shape[:2]
    if h < 16 or w < 16:
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (179, 255, 255))
    region = mask[int(.35*h):h, int(.25*w):int(.75*w)]
    return bool(np.count_nonzero(region) >= 60 * h * w / (240 * 320))


def verify_hold(wrist_frames, gripper_positions, mode="fusion"):
    """Return held/empty/unknown based on a three-frame persistent cue.

    Thresholds are inherited from the known-cube recovery heuristic. Visual
    visibility is not evidence of physical contact. Simulation research only.
    """
    if mode not in ("gripper", "vision", "fusion"):
        raise ValueError("invalid verifier mode")
    positions = np.asarray(gripper_positions, dtype=float)
    if mode != "vision":
        if positions.shape != (3,) or not np.isfinite(positions).all():
            return "unknown"
        obstructed = bool(np.all((positions >= .08) & (positions <= .8)))
        closed_empty = bool(np.all(positions < .03))
        if mode == "gripper":
            return "held" if obstructed else "empty" if closed_empty else "unknown"
    frames = list(wrist_frames)
    if len(frames) != 3:
        return "unknown"
    visible = [wrist_red_visible(frame) for frame in frames]
    if any(v is None for v in visible):
        return "unknown"
    red = all(visible)
    no_red = not any(visible)
    if mode == "vision":
        return "held" if red else "unknown"
    if obstructed and red:
        return "held"
    if closed_empty and no_red:
        return "empty"
    return "unknown"
