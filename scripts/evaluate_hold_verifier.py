"""Compare camera/joint hold cues with privileged offline simulation labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from hold_verifier import verify_hold


def truth_label(cube_z, bilateral, initial_z, stage="after_lift"):
    """Conservative stage label; ambiguous states are excluded from accuracy."""
    z = np.asarray(cube_z, dtype=float)
    contact = np.asarray(bilateral, dtype=bool)
    if z.shape != (5,) or contact.shape != (5,) or not np.isfinite(z).all() or not np.isfinite(initial_z):
        return "unknown"
    if np.count_nonzero(contact) >= 3 and (stage == "after_closure" or np.all(z - initial_z >= .05)):
        return "held"
    if not np.any(contact) and np.all(z - initial_z <= .02):
        return "empty"
    return "unknown"


def loss_detection(time, z, contact, wrist, positions, initial_z, start, end):
    """Offline signed delay after a conservative simulator empty reference."""
    reference = None
    detections = {"gripper": None, "fusion": None}
    for i in range(max(5, start), min(end, len(time)-1) + 1):
        if reference is None and truth_label(z[i-4:i+1], contact[i-5:i], initial_z) == "empty":
            reference = float(time[i])
        for mode in detections:
            if detections[mode] is None and verify_hold(wrist[i-2:i+1], positions[i-2:i+1], mode) == "empty":
                detections[mode] = float(time[i])
    return {"reference_empty_time_s": reference, **{
        mode + "_empty_time_s": value for mode, value in detections.items()}, **{
        mode + "_signed_delay_s": (value - reference if value is not None and reference is not None else None)
        for mode, value in detections.items()}}


def evaluate_trace(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(d['metadata'].item())
        time, qpos = d['time'], d['qpos']
        z, contact, wrist = d['cube_xyz'][:, 2], d['bilateral_contact'], d['wrist_rgb']
        if len(time) != len(contact)+1 or len(qpos) != len(time) or len(wrist) != len(time):
            raise ValueError('unaligned capture')
        rows = []
        for event in meta.get('recovery_events', []):
            if event.get('state') != 'gripper_check' or event.get('reason') not in ('after_closure', 'after_lift', 'after_transport'):
                continue
            i = int(np.argmin(abs(time - event['time_s'])))
            if i < 5 or abs(time[i] - event['time_s']) > 1e-5:
                raise ValueError('stage timestamp is not aligned')
            truth = truth_label(z[i-4:i+1], contact[i-5:i], float(z[0]), event['reason'])
            decisions = {mode: verify_hold(wrist[i-2:i+1], qpos[i-2:i+1, 5], mode)
                         for mode in ('gripper', 'vision', 'fusion')}
            rows.append({'capture': str(path), 'seed': meta.get('seed'), 'candidate': meta.get('grasp_candidate'),
                         'stage': event['reason'], 'time_s': float(time[i]), 'truth': truth, **decisions})
        lift = next((r for r in rows if r['stage'] == 'after_lift'), None)
        transport = next((r for r in rows if r['stage'] == 'after_transport'), None)
        if lift and transport:
            start = int(np.argmin(abs(time - lift['time_s'])))
            end = int(np.argmin(abs(time - transport['time_s'])))
            loss = loss_detection(time, z, contact, wrist, qpos[:, 5], float(z[0]), start, end)
        else:
            loss = None
        with Path(path).open('rb') as capture_file:
            capture_sha256 = hashlib.file_digest(capture_file, 'sha256').hexdigest()
        info = {'sha256': capture_sha256, 'seed': meta.get('seed'), 'scenario': meta.get('scenario', 'cube'),
                'viewer_closed': meta.get('viewer_closed'), 'task_success': (meta.get('result') or {}).get('task_success'),
                'sim_grasp_friction': meta.get('sim_grasp_friction'), 'sim_push_force': meta.get('sim_push_force')}
        return rows, loss, info


def summarize(rows):
    result = {}
    for mode in ('gripper', 'vision', 'fusion'):
        matrix = {truth: {decision: 0 for decision in ('held', 'empty', 'unknown')}
                  for truth in ('held', 'empty', 'unknown')}
        for row in rows:
            matrix[row['truth']][row[mode]] += 1
        result[mode] = matrix
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('captures', nargs='+', type=Path)
    args = parser.parse_args()
    batches = [(path, *evaluate_trace(path)) for path in args.captures]
    rows = [row for _, batch, _, _ in batches for row in batch]
    coverage = [{'capture': str(path), **info, 'stages_evaluated': len(batch),
                 'stages': [row['stage'] for row in batch], 'loss_detection': loss}
                for path, batch, loss, info in batches]
    report = {'captures': coverage, 'rows': rows, 'confusion': summarize(rows),
              'confusion_by_stage': {stage: summarize([r for r in rows if r['stage'] == stage])
                                     for stage in ('after_closure', 'after_lift', 'after_transport')},
              'limitations': 'Contact and object height are privileged simulator-only ground truth proxies. Correlated stages are not independent trials.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['confusion'], indent=2))


if __name__ == '__main__':
    main()
