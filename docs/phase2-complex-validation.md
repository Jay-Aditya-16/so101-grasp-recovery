# Phase 2: varied objects and drawer validation

## Scope

We kept the 20,000-step ACT policy and the existing recovery logic frozen. The policy was trained for a red cube pick-and-place task, not drawer opening. Tests use the SO-101 MuJoCo tutorial scene, 20 Hz control, seed 1020, and a 25-second simulated horizon. Each row below is **one paired starting condition**, not a population success rate. The recovery variant used `--recovery --recover-unknown-empty --recover-unknown-stationary`; Depth Anything was not enabled. All task rollouts were displayed live and their camera frames, state, actions, outcomes, and recovery events were recorded.

| Case | ACT alone | ACT + existing recovery | Observation |
|---|---|---|---|
| 20 mm red cube | No lift, no placement | No lift, no placement | Reobservation found a 56-pixel red component, below the detector's 60-pixel gate. |
| 30 mm green cube | 9.3 mm peak lift, no placement | 9.3 mm peak lift, no placement | The red-only detector found no target after reobservation. |
| 36 mm red sphere | No lift, no placement | 90.3 mm peak lift; no placement | The sphere slipped during transport; a gripper check detected an empty closure and stopped the attempt. |
| 50 × 24 × 30 mm red cuboid plus 30 mm green cube | 4.3 mm peak lift, no placement | 4.1 mm peak lift, no placement | The red cuboid was targeted, but the recovery grasp did not hold. The second object was a distractor, not an independently selected target. |
| Articulated drawer with red handle | 0 mm peak opening | 9.01 mm peak opening | Both fail the 60 mm sustained-opening threshold. Recovery treated the red handle as a pick target, not a drawer handle. |

The original red 30 mm cube control with recovery still placed, with a 96.3 mm peak lift. In the two-object traces, simulator ground truth shows the green cube's maximum displacement was about 0.15 mm in each run. This includes normal settling and does not prove the object was never touched. The recovery's gripper-angle check considered the drawer handle's obstruction plausible (0.091 rad), showing why a blocked gripper is not proof of grasping the intended object.

The drawer is a simple slide-joint test fixture with a tray, front and handle, not a calibrated household drawer. To verify its joint could move, an isolated 0.5 N generalized force for 0.2 s moved the joint 42.7 mm in the opening direction. This is a mechanism check, not a robot demonstration. Drawer success requires >= 60 mm displacement for 0.5 s. The unchanged ACT task inputs and recovery controller still implement pick-and-place; neither was given a learned drawer-opening policy or task-specific pulling logic. The ACT checkpoint is not language-conditioned by the task string. Failure therefore does not imply that all drawer-opening policies fail.

The sphere, green, two-object and drawer trials were captured before adding full-footprint placement scoring. All reported failures have no settled placement under the earlier criterion, so the stricter criterion cannot turn them into successes. The small-cube pair and original-cube control used the updated scorer, which requires the target's projected full box/sphere footprint to fit inside the container, no jaw contact, box-floor support, low speed and a 5 cm lift. The changed collision-path checker identifies actual robot bodies rather than treating an extra simulated object as part of the robot. These checks are still discrete, simulator-only and do not guarantee continuous clearance or safe physical forces.

## MATLAB and data

MATLAB R2025b independently analyzed exported cube-height, gripper-angle and drawer-position traces. See `reports/complex-v1/matlab-summary.csv`, `matlab-comparison.png`, and `matlab-trajectories.png`. Peak lift is a stage metric, not proof of placement. In particular, the lifted sphere was dropped. MATLAB uses privileged simulator state for evaluation, never for the policy decision.

The 11 recorded archives are `datasets/local-rollouts/complex-*.npz`; `reports/complex-v1/capture-validation.json` confirms observation/action alignment and outcome flags. `reports/complex-v1/scene-checks.json` records compiled shape dimensions and the drawer joint check; `distractor-motion.json` records simulator-derived distractor motion. Trials and summaries include failures. Earlier unsuccessful runs have not been relabeled as successes.

## Reproduce

Use the configured local SO-101 environment and the pinned tutorial and checkpoint documented in `docs/phase2-integrated.md`. Add, for example, `--scenario sphere --seed 1020 --seconds 25 --monitor` for ACT alone. For the matched recovery run replace `--monitor` with `--recovery --recover-unknown-empty --recover-unknown-stationary`. Available cases are `cube`, `small_cube`, `cuboid`, `sphere`, `green_cube`, `two_objects` and `drawer`; the standalone `cuboid` case was compiled but not run independently. Both views remain visible through the MuJoCo viewer and the recorded camera frames. Use `scripts/check_scenario_models.py`, `scripts/validate_captures.py`, and `scripts/analyze_distractor.py` to reproduce the model, trace, and distractor checks. `scripts/export_rollout_csv.py`, `matlab/analyze_phase2_batch.m`, and `matlab/plot_complex_cases.m` reproduce the MATLAB outputs.

## Interpretation and next work

These results **do not validate Phase 2 across object types or tasks**. They identify concrete issues to address before claiming generalization: colour-specific detection, fixed-height and fixed-grasp assumptions, slip handling, target selection among multiple objects, and a task-specific drawer strategy. More poses, object orientations, shapes, repeated trials and retry-only comparisons are needed. A drawer skill requires a separate task model or training; a correction layer over an existing pick policy cannot be assumed to invent that skill. This suite remains simulation-only.

Relevant context: the [ACT paper](https://arxiv.org/abs/2304.13705) describes the trained imitation policy and its action chunks; [LeRobot](https://github.com/huggingface/lerobot) provides the policy interface; and the [imitation-learning generalization study](https://arxiv.org/abs/2307.03659) motivates separating different sources of distribution shift.
