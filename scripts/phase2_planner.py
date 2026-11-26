"""Generate and inspect a simulation-only grasp candidate; never actuates a robot."""
import numpy as np
import mujoco


def check_pose(robot, q, target, max_position_error_m=0.005, min_direction_dot=0.98):
    """Independently verify FK position, downward tool axis and joint bounds."""
    q = np.asarray(q, dtype=float)
    if q.shape != (5,) or not np.isfinite(q).all():
        return {"valid": False, "reason": "invalid_joint_vector"}
    ids = [robot.model.joint(n).id for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
    limits = robot.model.jnt_range[ids]
    in_limits = bool(np.all(q >= limits[:, 0] - 1e-9) and np.all(q <= limits[:, 1] + 1e-9))
    data = mujoco.MjData(robot.model)
    data.qpos[:] = robot.data.qpos
    data.qpos[robot.arm_qadr] = q
    mujoco.mj_forward(robot.model, data)
    xyz = data.site_xpos[robot.site_id].copy()
    axis = data.site_xmat[robot.site_id].reshape(3, 3)[:, 0]
    position_error = float(np.linalg.norm(xyz - target))
    direction_dot = float(np.dot(axis, [0, 0, -1]))
    return {"valid": in_limits and position_error <= max_position_error_m and direction_dot >= min_direction_dot,
            "in_joint_limits": in_limits, "position_error_m": position_error,
            "downward_axis_dot": direction_dot, "site_xyz_m": xyz.tolist()}


def inspect_path(robot, start, end, samples=101, allow_cube_contact=False):
    """Sample joint interpolation for robot contacts; this is not continuous collision checking."""
    m = robot.model
    data = mujoco.MjData(m)
    data.qpos[:] = robot.data.qpos
    excluded = {0, m.body("cube").id, m.body("box").id, m.body("cam_target").id}
    collisions = []
    for i, alpha in enumerate(np.linspace(0.0, 1.0, samples)):
        data.qpos[robot.arm_qadr] = (1 - alpha) * np.asarray(start) + alpha * np.asarray(end)
        mujoco.mj_forward(m, data)
        for c in data.contact:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if allow_cube_contact and robot.cube_id in (b1, b2):
                continue
            if (b1 not in excluded or b2 not in excluded) and c.dist < -0.001:
                collisions.append({"sample": i, "geom1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                   "geom2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
                                   "penetration_m": float(-c.dist)})
                break
        if collisions:
            break
    return {"samples": samples, "first_penetration": collisions[0] if collisions else None,
            "caveat": "Discrete kinematic samples only; no dynamic, swept-volume or real-world guarantee."}


def propose_approach(robot, estimated_xy):
    """Compute one top-down approach above a visible cube; no motion is sent."""
    xy = np.asarray(estimated_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        return {"status": "abstain", "reason": "invalid_object_estimate"}
    target = np.array([xy[0], xy[1], 0.08])
    q, solver_error = robot.ik(target)
    pose = check_pose(robot, q, target)
    path = inspect_path(robot, robot.q_arm, q)
    status = "candidate_only" if pose["valid"] and path["first_penetration"] is None else "abstain"
    return {"status": status, "target_xyz_m": target.tolist(), "joint_target_rad": q.tolist(),
            "solver_combined_error": float(solver_error), "independent_fk_check": pose, "path_check": path}
