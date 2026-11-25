"""Online monitor for a visible red cube and a position-controlled gripper.

Inputs are camera frames, measured joints, and forward-kinematic tip height.
Ground-truth object position and contact are never used by the monitor.
"""

from dataclasses import dataclass

import cv2
import numpy as np


def red_center(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    upper = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([179, 255, 255]))
    count, _, stats, centers = cv2.connectedComponentsWithStats(lower | upper)
    if count < 2:
        return None
    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[label, cv2.CC_STAT_AREA] < 20:
        return None
    return centers[label]


@dataclass
class MonitorEvent:
    frame: int
    status: str
    reason: str


class GraspMonitor:
    """Flag a possible miss after closure and upward movement.

    Thresholds target the tutorial scene's 20 Hz and radian joint convention.
    A hidden object yields an unknown state, not a success claim.
    """

    def __init__(self, close_threshold=0.10, lift_m=0.02, stationary_px=15, max_age_frames=40):
        self.close_threshold = close_threshold
        self.lift_m = lift_m
        self.stationary_px = stationary_px
        self.max_age_frames = max_age_frames
        self.reset()

    def reset(self):
        self.frame = 0
        self.last_visible = None
        self.last_visible_frame = None
        self.close_center = None
        self.close_height = None
        self.closed = False
        self.event = None

    def update(self, top_rgb, gripper_target, tip_height):
        center = red_center(top_rgb)
        if not self.closed and gripper_target < self.close_threshold:
            self.closed = True
            if self.last_visible_frame is not None and self.frame - self.last_visible_frame <= self.max_age_frames:
                self.close_center = self.last_visible
            self.close_height = tip_height
        if center is not None:
            self.last_visible = center
            self.last_visible_frame = self.frame
        if self.closed and self.event is None and tip_height - self.close_height >= self.lift_m:
            if center is None or self.close_center is None:
                status, reason = "unknown", "cube obscured when gripper rose"
            elif np.linalg.norm(center - self.close_center) <= self.stationary_px:
                status, reason = "suspected_miss", "gripper rose while cube remained near its prior image position"
            else:
                status, reason = "unknown", "cube moved in image; verify grasp before claiming success"
            self.event = MonitorEvent(self.frame, status, reason)
        self.frame += 1
        return self.event
