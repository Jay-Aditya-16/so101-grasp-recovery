"""Offline IK/tilt diagnostics at a recorded placement-planning state; no motion.

Inputs include simulator state, so results are oracle diagnostics only. This is
not a closed-loop ablation, and does not establish safe transfer or placement.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from phase2_planner import check_pose, inspect_path
from placement_bandit import ACTIONS, OFFSETS


def summarize_thresholds(trials):
    """Vary the position-only tilt limit while retaining strict down-IK limit."""
    thresholds = {}
    for dot in (.98, .70, .65, .60):
        matches = [r for r in trials if r['in_joint_limits'] and r['position_error_m'] <= .005
                   and r['axis_dot'] >= (.98 if r['direction'] == 'down' else dot)
                   and r['path'] is not None and r['path']['first_penetration'] is None]
        thresholds[str(dot)] = {'feasible_solver_outputs': len(matches),
                                'candidates': sorted({r['candidate'] for r in matches}),
                                'max_tilt_deg': float(np.degrees(np.arccos(dot)))}
    return thresholds


def run(trace_path, tutorial):
    sys.path.insert(0, str(tutorial.resolve()))
    from common.ik import solve_ik
    from common.robot import ARM_JOINTS, SITE, DOWN, SO101Sim

    with np.load(trace_path, allow_pickle=False) as trace:
        meta = json.loads(trace["metadata"].item())
        if meta.get("scenario", "cube") != "cube":
            raise ValueError("Only original cube scene is supported")
        event = next((x for x in meta.get("recovery_events", []) if x["state"] == "aborted" and
                      (x.get("reason") or "").startswith("no IK solution")), None)
        if event is None:
            raise ValueError("A recorded IK abort is required")
        event_time = event["time_s"]
        index = int(np.argmin(np.abs(trace["time"] - event_time)))
        if abs(float(trace["time"][index]) - event_time) > 1e-5:
            raise ValueError("No matching recorded state")
        qpos = trace["qpos"][index].copy()
        robot = SO101Sim(cube_pos=meta["start_xy"], box_pos=(0.05, 0.22), render=False)
        try:
            if qpos.shape != robot.data.qpos.shape:
                raise ValueError("Recorded state dimension differs from scene")
            robot.data.qpos[:] = qpos
            mujoco.mj_forward(robot.model, robot.data)
            starts = [robot.q_arm.copy(), robot.model.key_qpos[0, robot.arm_qadr].copy()]
            trials = []
            for name, (dx, dy) in zip(ACTIONS, OFFSETS):
                target = np.array([0.05+dx, 0.22+dy, 0.12])
                for direction_name, direction in (("down", DOWN), ("position_only", None)):
                    for seed_name, seed in zip(("current", "home"), starts):
                        scratch = mujoco.MjData(robot.model)
                        scratch.qpos[:] = robot.data.qpos
                        scratch.qpos[robot.arm_qadr] = seed
                        q, _ = solve_ik(robot.model, scratch, SITE, target, direction, ARM_JOINTS)
                        p = check_pose(robot, q, target, min_direction_dot=-1.)
                        # Matches existing planner: 101 samples, expected cube contact excluded.
                        path = inspect_path(robot, robot.q_arm, q, allow_cube_contact=True) if p['in_joint_limits'] and p['position_error_m'] <= .005 else None
                        trials.append({'candidate': name, 'direction': direction_name, 'seed': seed_name,
                                       'position_error_m': p['position_error_m'], 'axis_dot': p['downward_axis_dot'],
                                       'in_joint_limits': p['in_joint_limits'], 'path': path})
            thresholds = summarize_thresholds(trials)
            logged = ast.literal_eval(event['reason'].split(': ', 1)[1])
            if len(logged) != len(trials):
                raise ValueError('recorded and reconstructed solution counts differ')
            reproduction = {'max_position_error_difference_m': max(abs(a['position_error_m']-b['position_error_m']) for a,b in zip(logged,trials)),
                            'max_axis_dot_difference': max(abs(a['downward_axis_dot']-b['axis_dot']) for a,b in zip(logged,trials))}
            return {'trace': str(trace_path), 'seed': meta.get('seed'), 'time_s': event_time,
                    'frame': index, 'thresholds': thresholds, 'reproduction': reproduction, 'solutions': trials,
                    'scope': 'Offline FK/IK and sampled path counterfactual only; no action executed, no carried-object sweep, dynamics or physical safety validation.'}
        finally:
            robot.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tutorial', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.trace, args.tutorial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'seed': result['seed'], 'time_s': result['time_s'], 'thresholds': result['thresholds']}, indent=2))


if __name__ == '__main__':
    main()
