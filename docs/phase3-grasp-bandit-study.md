# Grasp-candidate collection and BORGES-inspired memory

## Scope

This is a development experiment on the single, known red cube in the pinned MuJoCo scene. The original ACT checkpoint is frozen. After the existing failure detector invokes scripted recovery, three predeclared candidate **target gripper-site heights** are tested: `low=0.015 m`, `center=0.020 m` and `high=0.025 m`, at the camera-estimated cube XY. These are small variations in the contact geometry of a known cube, not a general object-grasp generator or three identified stable poses. The existing IK, orientation, joint limit and sampled robot-path checks are applied. The tolerance is 5 mm, equal to the gap between candidate heights, so requested heights alone do not prove distinct physical grasps. Later runs log the predicted FK site pose, and camera/action archives permit further review.

## Outcome and memory

The collector stores camera frames, joint states, actions, contact flags and simulator cube state per trial. The offline extractor labels a completed lift-check stage as positive if, between closure and that check, the cube rises by at least 5 cm and two-jaw contact is observed at least once in that interval. The events need not occur simultaneously. This is a **simulator proxy**, not a stable-grasp guarantee and not a sensor available to a physical robot. No selected grasp, interrupted runs, and attempts that do not reach the lift check are censored. Placement success and the controller’s post-transport gripper obstruction heuristic are recorded separately. An obstruction reading is not a verified hold.

`scripts/grasp_bandit.py` implements a BORGES-inspired, per-context/candidate Beta-Bernoulli memory with Thompson sampling: for prior `Beta(1,1)`, after `s` positive and `f` negative verified labels the posterior is `Beta(1+s,1+f)`. The posterior mean is `(1+s)/(2+s+f)`; it depends on an assumed stationary Bernoulli model and is not a safety or success guarantee. Only caller-provided feasible candidates may be ranked; the memory rejects unknown pose labels, duplicate trial IDs and unverified labels. Parallel workers must never write the same memory: the offline replay uses one coordinator and per-capture SHA-256 identifiers. A deliberately interrupted run is not entered as a failure.

The replay can maintain separate memories for the lift proxy and completed full-task outcome. Full-task failures can be caused by grasp slip, perception or placement planning, so they are not automatically failed grasps. Only trials that reached the lift check are used in either replay, which introduces selection bias. The present replay assigns a single **assumed fixed-cube class**; it does not estimate stable object poses. It updates from offline simulator-labelled trajectories and does not select or execute robot actions. It does not reproduce the pose-transition model or guarantees in [BORGES](https://proceedings.mlr.press/v155/danielczuk21a.html). The same two seeds and different heights are correlated; posterior counts are bookkeeping, not calibrated evidence of general performance.

## Filtering decision

An EKF has not been applied. Current recovery uses one red-centroid XY observation after returning to a fixed view; no evidence shows that a filter would improve it. A filter cannot remove unknown object-height or camera-calibration bias or create measurements through occlusion. If repeated tracking is introduced, start with a linear XY constant-velocity Kalman filter and compare against raw measurements and simple smoothing on separate held-out episodes. Test lag, dropout, positional error and uncertainty calibration. An EKF would be appropriate only for a justified nonlinear observation model, such as direct pixel projection with known calibration and observable geometry; do not treat monocular relative depth as metric ground truth. See [MathWorks' linear Kalman filter guide](https://www.mathworks.com/help/fusion/ug/linear-kalman-filters.html).

## Execution and reproducibility

The predeclared plan is `reports/phase3-grasp-study/plan.json`; per-trial result JSON, source capture hashes and validation are versioned there. Full camera/action archives are stored locally under `datasets/local-rollouts/borges-*.npz` and are not pushed to Git. Two full ACT/MuJoCo processes were initially launched in parallel with live windows; the 8 GB Mac experienced severe memory pressure, so the 1031-high startup was intentionally terminated without a capture and rerun sequentially. This interruption is recorded and is not a failed grasp. Remaining runs were collected serially.

A controller run is simulation-only, for example:

```sh
PYTHONPATH=/private/tmp/so101-lerobot-public061 HF_HOME=/private/tmp/so101-dataset-validation HF_HUB_OFFLINE=1 \
 /Users/jayaditya/so101-lerobot-juxi/.venv/bin/mjpython scripts/live_act_eval.py \
 --tutorial ../so101-physicalai-tutorial --checkpoint checkpoints/single-cube-020000.VdbmMp/pretrained_model \
 --seed 1030 --seconds 25 --recovery --recover-unknown-empty --recover-unknown-stationary \
 --experimental-cone-ik --grasp-candidate low --results outputs/example.json
```

The viewer is live. Geometry and collision checks are discrete and do not cover the held object's swept volume, contact forces, motor torques, or real hardware safety. Replayed data do not demonstrate a learned closed-loop improvement.

## Results

All six completed recordings passed alignment validation. Each candidate is shown once per seed; these are development trials, not an independent test set.

| Seed | Candidate | Lift proxy | Complete pick-and-place | Post-transport obstruction heuristic |
| --- | --- | --- | --- | --- |
| 1030 | low | 1 | success | plausible |
| 1030 | center | 1 | success | plausible |
| 1030 | high | 1 | failure | not plausible |
| 1031 | low | 1 | success | plausible |
| 1031 | center | 1 | success | plausible |
| 1031 | high | 1 | success | plausible |

On 1030-high the object reached the lift proxy, but after transport the gripper appeared empty and the controller aborted. The two labels correctly capture different outcomes; the failure could arise from grasp retention, motion, or other factors, so it is not automatically a failed grasp. The first 1031-high process was intentionally interrupted before recording; the table contains its later successful rerun.

The lift replay produces `Beta(3,1)` for each candidate. The separate task replay yields `Beta(3,1)` for low and center and `Beta(2,2)` for high. With only two correlated trials per candidate, these are examples of the update mechanics, not reliable success-probability estimates or a comparison of learned versus fixed control. Neither replay was used to choose these trials. MATLAB independently reproduced both sets of posterior means exactly from the stored labels; this checks the arithmetic, not the validity of the proxy or robot physics.

For the four runs with detailed FK fields, the predicted center/high z coordinates were approximately 20.50/25.90 mm on seed 1030; the predicted low/center z coordinates were 15.51/20.47 mm on seed 1031. These are solver-predicted endpoints, not measured contact geometry; the initial low/high runs did not log the detailed FK field. Full physics robustness and grasp orientation remain unmeasured.

Next, obtain more independent object positions, genuine failed lift and lost-object examples, and a sensor-based hold verifier. Validate observed candidate feasibility before letting the grasp bandit select in simulation. Then compare fixed, random feasible and learned choices under matched budgets on genuinely held-out seeds, keeping planning failures and censored attempts in the report. Until then there is no demonstrated closed-loop BORGES improvement.
