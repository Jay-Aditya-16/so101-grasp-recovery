"""Estimate a red object's tabletop position from RGB and camera calibration.

This known-scene prototype does not command the robot or use simulator object state.
"""
import cv2
import numpy as np


def estimate_object(rgb, camera_position, camera_rotation, fovy_deg, plane_z=0.03, min_visible_px=60):
    """Return a tabletop estimate, or None if the red region is ambiguous.

    The plane is the assumed visible top height of the 30 mm tutorial cube. The
    estimate becomes biased if the object height, calibration or color differs.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (179, 255, 255))
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    components = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]
    if len(components) != 1:
        return None
    i = components[0]
    x, y, width, height, area = stats[i]
    # Development threshold for this fixed-scale scene, not a calibrated confidence.
    if area < min_visible_px:
        return None
    h, w = rgb.shape[:2]
    if x == 0 or y == 0 or x + width >= w or y + height >= h:
        return None
    u, v = centers[i]
    focal = (h / 2) / np.tan(np.deg2rad(fovy_deg) / 2)
    direction = np.asarray(camera_rotation).reshape(3, 3) @ np.array([(u - w / 2) / focal, -(v - h / 2) / focal, -1.0])
    origin = np.asarray(camera_position)
    if not np.isfinite(direction).all() or abs(direction[2]) < 1e-8:
        return None
    distance = (plane_z - origin[2]) / direction[2]
    if distance <= 0:
        return None
    point = origin + distance * direction
    return {"pixel_xy": [float(u), float(v)], "world_xy_m": point[:2].tolist(),
            "bbox_xywh_px": [int(x), int(y), int(width), int(height)], "area_px": int(area),
            "assumed_top_height_m": plane_z}
