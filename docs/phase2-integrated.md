# Integrated Phase 2 simulator prototype

`scripts/live_act_eval.py --recovery` combines the fixed ACT checkpoint, Phase 1 monitor, an experimental return to the home observation pose, red-cube localization, top-down IK approach, gripper closure, lift and fixed-box placement. The controller uses camera images, joint state, a known cube-top plane at 30 mm, and the tutorial's fixed box coordinates `(0.05, 0.22)` m. It does not use MuJoCo cube positions or contacts to choose actions. The evaluator uses simulator state and contacts separately to score results. ACT is frozen, the recovery policy is hand-engineered, and no online learning or RL is implemented.

Only `suspected_miss` starts recovery; `unknown` continues ACT. The arm returns to the known viewing pose, obtains a fresh overhead observation, and generates joint targets with the tutorial IK solver. Each endpoint is independently checked with forward kinematics and joint limits. Up to 101 joint-interpolation samples are checked for penetrations greater than 1 mm. Expected cube contacts are ignored for grasp and transfer; the held object's swept volume is not checked. A position-only IK fallback is used if exact downward orientation fails, with tool tilt limited to about 26 degrees during lift and 46 degrees near the box. These choices were tuned in the simulator, not validated physical limits. On failed IK or path checks, the controller holds its measured joint pose.

## Live experiments

All experiments used the same simulator scene and 20k ACT checkpoint, model SHA-256 `974eda64112cb6505fa45d06bc9f74c9f848cc0db4416b8eab86a7021140a142`. Seeds 1001 and 1002 were development cases used to tune IK tilt limits. Their earlier ACT baselines failed to lift; after tuning, the recovery trials lifted and placed under the prior score (which did not explicitly require jaw release). Seed 1003 was a known ACT success and ran with recovery enabled but no intervention; its result also used the earlier score. They are **not** independent tests of the final implementation.

After selecting the 0.70 transfer-direction threshold, seed 1010 was evaluated with paired initial conditions and 21 simulated seconds per run. The baseline made no bilateral contact, no 5 cm lift, and no placement. The recovery run made bilateral contact, lifted 96.2 mm and placed. Its placement score required box-floor contact, distance within 30 mm of the box center in both axes, speed below 0.03 m/s and no jaw contact for 0.5 seconds after at least a 5 cm lift. The stricter release criterion was added after the first 1010 run and checked by rerunning recovery. This is one new paired case, not a success-rate estimate. See the per-trial JSON reports in `reports/phase2-*1010.json` and the development reports.

The monitor is a known-scene red-cube heuristic. During the recovery motion the controller does not reobserve the grasp, verify a held object before transfer, model a carried object's swept volume, or guarantee the candidate path is collision-free between discrete samples. It does not plan object-relative orientations beyond a top-down cube grasp, handle different shapes or colors, or implement a retry-only control condition. The camera depth model remains an offline experiment. Two-jaw contact alone is not proof of a stable grasp. The arm and controller are **simulation only**: neither this score nor the sampled checks establish physical safety or robustness.

## Reproduce

On the configured Mac environment, run the viewer with matching seeds and episode duration:

```sh
PYTHONPATH=/private/tmp/so101-lerobot-public061 HF_HOME=/private/tmp/so101-dataset-validation \
  /Users/jayaditya/so101-lerobot-juxi/.venv/bin/mjpython scripts/live_act_eval.py \
  --tutorial ../so101-physicalai-tutorial \
  --checkpoint checkpoints/single-cube-020000.VdbmMp/pretrained_model \
  --seed 1010 --seconds 21 --recovery --results outputs/recovery.json
```

For the baseline, replace `--recovery` with `--monitor` and use a different results path. The viewer runs live; closing it ends the episode without a success score. The required checkpoints and raw traces are local and excluded from Git.

## Remaining validation

Before treating Phase 2 as validated: gather predeclared held-out paired baseline/retry/recovery trials including successful baseline cases, cube positions and orientations, moved objects, occlusions and false alarms; report uncertainty and timing; add observation-based grasp verification and replan after slippage; check the carried cube and robot through the entire swept path with a conservative margin; and evaluate independent camera and kinematic calibration. No inference from these simulation tests should be treated as authorization to run this controller on an SO-101 arm.
