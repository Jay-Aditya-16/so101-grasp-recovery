# Phase III: outcome-based placement selection, first increment

This first Phase III increment leaves the 20k-step ACT checkpoint and Phase II recovery controller unchanged except for the ordering of **box placement** candidates. It does not yet learn grasp selection, fix an arbitrary policy, integrate a world model, or demonstrate lifelong learning. The existing Phase II controller is limited to the original red-cube MuJoCo scene; Phase III is explicitly restricted to this scene and is not for physical hardware.

## Decision and learning

The observation is the camera-estimated cube XY (overhead by default, or the wrist estimate if its optional gate is enabled), normalized as `[1, (x - 0.20)/0.10, y/0.18]`. Estimates outside x=[0.08, 0.30], y=[-0.18, 0.18] metres or non-finite values are rejected. The actions are the existing six configured positions within the fixed box. For each action, regularized least squares maintains `A = I + Σ xxᵀ` and `b = Σ reward·x`; it ranks with `xᵀ A⁻¹ b + 0.25 sqrt(xᵀ A⁻¹ x)`. This contextual bandit is a small one-step form of reinforcement learning; its scores are neither calibrated probabilities nor an assurance that an action will work. Fresh memory begins with equal scores and retains the original order on ties.

The selected action is the first ranked candidate that passes the **existing** endpoint IK and sampled path checks. The bandit cannot skip these checks. These checks ignore expected cube contact and do not establish continuous collision avoidance or account for the held object's swept volume. An operator can stop the viewer; no real arm is supported.

Reward is a simulator evaluation label, not a controller observation: 1 requires the existing strict settled placement criterion; 0 requires a completed release and sufficient observation time without a successful placement. Interrupted and incomplete attempts are not learned from. The record stores the chosen candidate, camera-derived XY, outcome, and trial ID in a local JSON file; writes are atomic, and repeated IDs are rejected. Memory is single-process and must not be concurrently written. Shadow mode retains the original Phase II ordering but records the action and result; active mode ranks the candidates. Memory must not be shared between evaluation conditions or the held-out test set.

Placement outcome is affected by grasp quality and other earlier behavior, so this reward has imperfect credit assignment. One success or failure does not show generalization or prevent recurrence. The controller has not learned a red-cube grasp correction from this bandit.

## Run live

```sh
PYTHONPATH=/private/tmp/so101-lerobot-public061 HF_HOME=/private/tmp/so101-dataset-validation HF_HUB_OFFLINE=1 \
 /Users/jayaditya/so101-lerobot-juxi/.venv/bin/mjpython scripts/live_act_eval.py \
 --tutorial ../so101-physicalai-tutorial --checkpoint checkpoints/single-cube-020000.VdbmMp/pretrained_model \
 --seed 1020 --seconds 25 --recovery --recover-unknown-empty --recover-unknown-stationary \
 --bandit-mode active --bandit-memory outputs/phase3/dev-memory.json \
 --results outputs/phase3/dev-1020.json
```

The viewer is live, and rollout archives are saved automatically. Start with a new memory file for independent experiments; `--bandit-mode shadow` records without changing ordering. Closing the viewer early does not add a label. For a held-out evaluation, add `--bandit-frozen` and use a fixed existing memory file to prevent updates. Do not use development seeds as independent evidence.

## Next evaluation gate

Collect separate development and predeclared held-out cube positions and compare frozen Phase II, shadow/no-memory and learned active memory under matched initial conditions and identical horizons. Record every trial, interventions, actual chosen candidate, completed releases, successes, and abstentions. Report counts and uncertainty, and compare time and collisions as well as task success. Include first-attempt and post-update outcomes; do not train on held-out labels. Then investigate grasp-specific, object-conditioned choices and online verification, with a clear strategy for human feedback and safe exploration. Only pursue depth, world-model or VLM additions if controlled comparisons show that they add value.

Related work: [ACT / ALOHA](https://tonyzhaozh.github.io/aloha/) motivates the frozen imitation baseline; [BORGES](https://proceedings.mlr.press/v155/danielczuk21a.html) investigates outcome-informed exploratory grasping. This implementation is a simpler placement bandit, not a reproduction of BORGES.

## Initial development observations

Two live development trials (seeds 1020 and 1022) both reached a settled placement and wrote successful center-target records to `reports/phase3-dev/dev-memory.json`. The second trial used the first outcome when computing its scores; it still chose the same center target as the Phase II ordering. These are previously used development poses, with no measured advantage over Phase II and no observed failure update. Recorded NPZ videos and transitions are local under `datasets/local-rollouts/phase3-dev-1020.npz` and `phase3-dev-1022.npz`; compact event summaries and memory are in `reports/phase3-dev/`. MATLAB R2025b independently recomputed the six scores at the first observed position using both records; its scores agree with Python to within 3.4e-16. This verifies the implementation's arithmetic for this input, not the physical model or learning effectiveness.

A third live run repeated development seed 1020 with `--bandit-frozen`: placement succeeded, the event log reported `memory frozen`, and the memory file was byte-identical before and after. It is a wiring check on a repeated seed, not a held-out evaluation.
