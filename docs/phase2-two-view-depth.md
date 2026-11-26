# Two-view recovery and Depth Anything V2

The experimental `--recovery --depth-gate --depth-view wrist` mode uses the overhead RGB image and calibrated camera pose to estimate a red cube's position after returning to view. At the approach pose, it projects the wrist-camera red centroid onto an assumed horizontal 30 mm cube-top plane using the wrist camera's pose calculated from the current joints. It evaluates Depth Anything V2 Small on that wrist RGB image; the model checks whether the visible red region has a higher relative depth score than its nearby ring. The controller accepts the wrist estimate only if the red region is visible, the model's ordering is positive, and its projected position is within 10 mm of the overhead estimate. It then uses the wrist estimate for descent.

Depth Anything V2's output is **relative**, not a measurement in metres. This experiment does not use it as metric depth. The fixed 10 mm agreement threshold and relative ordering are uncalibrated heuristics; the wrist image can include cube side faces, the gripper, and occlusion. Agreement does not establish correctness. The robot is controlled from camera images, current joint state, known cube-height and fixed-box assumptions; simulator object coordinates are used only for subsequent evaluation.

## Results

Two live wrist-depth recovery trials on seeds 1010 and 1001 both lifted and placed the cube. The existing controller without depth also placed on both; seed 1001 was used during controller development, so neither result establishes a success-rate improvement. At the wrist decision frame the simulator XY reference showed:

| Seed | Overhead estimate error | Wrist estimate error | Cameras' disagreement | Outcome with wrist depth |
|---|---:|---:|---:|---|
| 1010 | 1.72 mm | 7.54 mm | 5.87 mm | Placed |
| 1001 | 2.06 mm | 7.65 mm | 6.14 mm | Placed |

The wrist estimate was less accurate on both tested trials, although each passed the current agreement gate. The relative-depth model accepted both and its wall-clock calls took 1.74 and 1.48 seconds respectively; a separate top-depth trial took approximately 13 seconds, illustrating variability. These are not real-time latency benchmarks.

The top-camera depth gate was separately tested live on seeds 1010 and 1001; both placed, matching the existing no-depth outcomes. Seed 1011 without depth failed with a monitor `unknown`, so recovery did not start. This depth gate is downstream of the monitor and cannot repair that trigger failure. No paired depth run was performed on 1011 and its raw frames were not captured. The results do not show that depth improves either success rate or localization accuracy. See `reports/phase2-two-view-depth-evaluation.json`, the top-depth reports and the no-depth 1011 result.

## Recorded trials

Future completed or viewer-closed `live_act_eval.py` runs automatically write a uniquely named NPZ archive under `datasets/local-rollouts/` unless `--no-capture` is passed; `--capture` chooses an explicit path. Each archive contains paired overhead and wrist RGB, timestamps, joint positions and velocities, actions, simulator object positions, contacts, reward and outcome flags. Recovery runs also store events and depth decisions. Simulator object/contact values are privileged labels, not policy inputs. Unexpected process crashes can interrupt recording before the archive is finalized; not every earlier demo was captured.

The private repository includes selected compact rollout archives: ACT failures `act-020000-demo-000.npz`, `phase1-act-seed1001.npz`, `phase1-act-seed1002.npz`, an ACT success `phase1-act-seed1003.npz`, and the two wrist-depth success archives. Raw recordings are simulator trajectories, not a ready-made LeRobot training dataset. Do not use failed actions as expert demonstrations. Additional automatic captures are initially local and ignored by Git; explicitly add reviewed selected datasets to a PR or move larger collections to appropriate dataset storage. LeRobot demonstration dataset and model checkpoint remain separate from these captures.

## Reproduction

The tested environment was the local SO-101 venv plus LeRobot 0.6.1 and Transformers 5.5.4. Use a compatible environment with the model cached or downloadable; do not expose credentials. For example on this Mac:

```sh
PYTHONPATH=/private/tmp/so101-depth-compatible:/private/tmp/so101-lerobot-public061 \
HF_HOME=/private/tmp/so101-dataset-validation HF_HUB_CACHE=/private/tmp/so101-dataset-validation/hub \
HF_HUB_OFFLINE=1 /Users/jayaditya/so101-lerobot-juxi/.venv/bin/mjpython scripts/live_act_eval.py \
  --tutorial ../so101-physicalai-tutorial --checkpoint checkpoints/single-cube-020000.VdbmMp/pretrained_model \
  --seed 1010 --seconds 21 --recovery --depth-gate --depth-view wrist --results outputs/wrist-eval.json
```

Compare with the same command without `--depth-gate --depth-view wrist`; use the same seed, duration, scene, and checkpoint. The `--depth-view top` mode checks the overhead image instead. `scripts/summarize_depth_trials.py` compares wrist and overhead XY estimates to privileged simulator labels in saved captures.

Next, collect paired held-out poses and object types, measure false acceptance and abstention, evaluate metric calibration and the cause of wrist-centroid bias, improve the failure trigger, and verify lift and contact with observations. Do not use this simulation controller on the physical arm.
