"""Offline IK diagnostics from one recorded simulation state; does not actuate.

This reconstructs the tutorial model and may use privileged simulator state to
reproduce contacts. Tested geometric paths do not prove transport safety.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


from phase2_planner import cone_directions


def attached_pose(site_xyz, site_rotation, relative_xyz, relative_rotation):
    """Compose a fixed site-to-object transform in world coordinates."""
    rotation = np.asarray(site_rotation, dtype=float).reshape(3, 3)
    return (np.asarray(site_xyz, dtype=float) + rotation @ relative_xyz,
            rotation @ relative_rotation)


def inspect_rigid_cube_path(robot, start, end, samples=101):
    """Check one ideal rigid attachment in joint-space interpolation, offline only.

    This cannot model slip, force, cube-robot contact or continuous swept volume.
    """
    import mujoco
    m = robot.model
    d = mujoco.MjData(m)
    d.qpos[:] = robot.data.qpos
    d.qvel[:] = robot.data.qvel
    mujoco.mj_forward(m, d)
    site0 = d.site_xpos[robot.site_id].copy()
    rotation0 = d.site_xmat[robot.site_id].reshape(3, 3).copy()
    cube0 = d.xpos[robot.cube_id].copy()
    cube_rot0 = d.xmat[robot.cube_id].reshape(3, 3).copy()
    relative_xyz = rotation0.T @ (cube0 - site0)
    relative_rotation = rotation0.T @ cube_rot0
    cube_adr = int(m.jnt_qposadr[m.joint('cube_free').id])
    root = int(m.jnt_bodyid[m.joint('shoulder_pan').id])
    robot_bodies = set()
    for body in range(1, m.nbody):
        parent = body
        while parent and parent != root:
            parent = int(m.body_parentid[parent])
        if parent == root:
            robot_bodies.add(body)
    for i, alpha in enumerate(np.linspace(0., 1., samples)):
        d.qpos[robot.arm_qadr] = (1-alpha)*start + alpha*end
        mujoco.mj_forward(m, d)
        site = d.site_xpos[robot.site_id].copy()
        rotation = d.site_xmat[robot.site_id].reshape(3, 3).copy()
        cube_position, cube_rotation = attached_pose(site, rotation, relative_xyz, relative_rotation)
        d.qpos[cube_adr:cube_adr+3] = cube_position
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(cube_rotation.ravel()))
        d.qpos[cube_adr+3:cube_adr+7] = quat
        mujoco.mj_forward(m, d)
        for contact in d.contact:
            b1, b2 = int(m.geom_bodyid[contact.geom1]), int(m.geom_bodyid[contact.geom2])
            if robot.cube_id not in (b1, b2):
                continue
            other = b2 if b1 == robot.cube_id else b1
            if other in robot_bodies or other == robot.cube_id:
                continue
            if contact.dist < -0.001:
                return {'samples': samples, 'first_penetration': {'sample': i,
                    'geom1': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                    'geom2': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
                    'penetration_m': float(-contact.dist)}}
    return {'samples': samples, 'first_penetration': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--tutorial', type=Path, required=True)
    parser.add_argument('--time', type=float, default=15.25)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tutorial.resolve()))
    import mujoco
    from common.ik import solve_ik
    from common.robot import SO101Sim, ARM_JOINTS, SITE, DOWN
    from phase2_planner import check_pose, inspect_path
    from scenario_suite import scenario_scene
    with np.load(args.trace, allow_pickle=False) as archive:
        meta = json.loads(archive['metadata'].item())
        if meta.get('scenario') != 'cube':
            raise ValueError('Only the pinned cube simulation can be reconstructed')
        i = int(np.argmin(np.abs(archive['time'] - args.time)))
        qpos, qvel = archive['qpos'][i].copy(), archive['qvel'][i].copy()
        measured_time = float(archive['time'][i])
    with scenario_scene('cube') as scene:
        robot = SO101Sim(cube_pos=meta['start_xy'], box_pos=(.05, .22), render=False, **scene)
    if qpos.shape != robot.data.qpos.shape or qvel.shape != robot.data.qvel.shape:
        raise ValueError('Trace and simulation state shapes differ')
    robot.data.qpos[:] = qpos
    robot.data.qvel[:] = qvel
    mujoco.mj_forward(robot.model, robot.data)
    target = np.array([.05, .22, .12])
    home = robot.model.key_qpos[0, robot.arm_qadr].copy()
    seeds = [('current', robot.q_arm), ('home', home)]
    directions = [('down', DOWN), ('position_only', None)] + [(f'cone40_{i}', d) for i, d in enumerate(cone_directions())]
    records = []
    for direction_name, direction in directions:
        for seed_name, seed in seeds:
            data = mujoco.MjData(robot.model)
            data.qpos[:] = robot.data.qpos
            data.qpos[robot.arm_qadr] = seed
            q, solver_error = solve_ik(robot.model, data, SITE, target, direction, ARM_JOINTS)
            check = check_pose(robot, q, target, min_direction_dot=.70)
            path = inspect_path(robot, robot.q_arm, q, allow_cube_contact=True) if check['valid'] else None
            records.append({'direction': direction_name, 'seed': seed_name, 'solver_error': float(solver_error),
                            'joint_target_rad': q.tolist(), 'pose': check, 'sampled_path': path,
                            'rigid_cube_path': inspect_rigid_cube_path(robot, robot.q_arm, q) if check['valid'] else None})
    report = {'trace': str(args.trace), 'requested_time_s': args.time, 'matched_time_s': measured_time,
              'target_xyz_m': target.tolist(), 'candidate_count': len(records), 'records': records,
              'limitations': 'Offline kinematic replay only; source state includes simulator ground truth. Rigid-cube path assumes no slip and ignores cube-robot contact; no continuous swept-volume, dynamic or real-world safety guarantees.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pose_valid': sum(r['pose']['valid'] for r in records),
        'path_valid': sum(bool(r['sampled_path'] is not None and r['sampled_path']['first_penetration'] is None) for r in records),
        'output': str(args.output)}))


if __name__ == '__main__':
    main()
