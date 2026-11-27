"""Compare stored camera-based XY with simulator truth at the recovery observation.

This script consumes recorded ground truth strictly as an offline diagnostic.
"""
import argparse
import ast
import json
from pathlib import Path
import numpy as np


def summarize(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data['metadata'].item())
        event = next((event for event in meta.get('recovery_events', []) if event['state'] == 'open' and
                      (event.get('reason') or '').startswith('observed cube at ')), None)
        if event is None:
            raise ValueError(f'{path}: recovery observation event missing')
        xy = np.array(ast.literal_eval(event['reason'].removeprefix('observed cube at ')), dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError('invalid observed position')
        frame = int(np.argmin(abs(data['time'] - event['time_s'])))
        if abs(float(data['time'][frame]) - event['time_s']) > 1e-5:
            raise ValueError('missing aligned observation')
        truth = data['cube_xyz'][frame, :2]
        return {'trace': str(path), 'seed': meta['seed'], 'frame': frame, 'time_s': event['time_s'],
                'observed_xy_m': xy.tolist(), 'simulator_truth_xy_m': truth.tolist(),
                'xy_error_mm': float(1000 * np.linalg.norm(xy - truth)),
                'settled_placement': meta['result']['settled_placement']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('traces', type=Path, nargs='+')
    args = p.parse_args()
    output = {'scope': 'Recorded observation vs simulator truth, offline only; not a calibrated uncertainty estimate',
              'rows': [summarize(t) for t in args.traces]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
