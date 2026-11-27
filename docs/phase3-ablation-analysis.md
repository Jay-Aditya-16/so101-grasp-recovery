# Ablation and bottleneck analysis: SO-101 simulated pick-and-place

## Research question and scope

Which part of the recovery stack causes the observed failures, and what evidence supports the next implementation? This analysis uses existing recorded MuJoCo trials of the fixed red cube and the frozen 20,000-step ACT checkpoint. It mixes historical development cases with one predeclared pilot; these are small, non-independent demonstrations, not a population success-rate estimate. No new arm motion was executed for the analysis below. Object positions and contacts used for measurement are privileged simulator state; the controller does not receive them.

The evidence includes *system-level comparisons* and a controlled **offline IK threshold sensitivity analysis**. It does not separately ablate each recovery component in closed loop, since retry, observation, geometry and grasp control change together in the available historical runs. Do not attribute the recovered success to a specific individual component solely from this table.

## Recorded system comparisons

| Case | ACT | Return-and-retry ACT | Object-aware recovery | Notes |
|---|---|---|---|---|
| seed 1020 | failed; lift 0 mm | failed; lift 0 mm | placed; lift 96.3 mm | development data |
| seed 1022 | failed; lift 6.8 mm | failed; lift 5.9 mm | placed; lift 97.6 mm | tested after earlier tuning; not a formal held-out set |
| seed 1030 | not tested in this comparison | not tested | lifted 97.6 mm, failed to place | paired Phase II and frozen Phase III both failed identically |

The 1020/1022 rows show that the combined scripted recovery was more useful than retry alone in those two cases; they do not isolate sensing, grasping, or planning. The seed-1030 held-out-to-memory pilot did not demonstrate any benefit from Phase III. The ACT-only and retry conditions were not run for 1030, so their outcomes are unknown. For seed 1022, the wrist-depth version and the no-depth recovery both placed; this single matched comparison does not establish any performance improvement from Depth Anything V2. Its relative depth output is not a calibrated metric distance sensor.

## Failure decomposition for seed 1030

At the fresh overhead observation, camera-derived cube XY differed from simulator truth by **2.07 mm**. The gripper subsequently made bilateral contact and the object rose **97.6 mm**, but transfer planning returned no candidate satisfying its existing position, orientation and sampled path checks. This shows transfer planning was the *proximal observed failure* in this trial; it does not establish that perception and grasping are reliable across other objects or conditions. Offline observation-error diagnostics on two development traces give 2.95 mm at 1020 and 0.38 mm at 1022. These are three selected simulated observations, not calibrated perception accuracy.

## Offline tilt-threshold sensitivity

At the recorded 1030 abort frame (15.25 s), replaying the six box targets, two IK direction settings and two IK initial seeds exactly reproduced all 24 logged position errors and axis dot products (maximum differences 0). The simulator state at the recorded frame was used only to initialize this offline analysis. For the position-only IK branch, retain the 5 mm endpoint bound, joint bounds and existing 101-sample path check, and vary only the minimum dot between tool axis and down. The strictly downward IK branch keeps its original 0.98 limit.

| Position-only minimum dot | Maximum axis tilt | Passing solver outputs | Distinct targets |
|---:|---:|---:|---:|
| 0.70, existing | 45.57° | 0/24 | 0/6 |
| 0.65 | 49.46° | 2/24 | 2/6 |
| 0.60 | 53.13° | 6/24 | 6/6 |

The dot product is `cos(tilt)` for unit tool and down axes. The best passing candidate at a 0.65 threshold has dot approximately 0.6645 (48.36°) and position error about 0.58 mm. These results localize the rejection to the measured endpoint tilt for this saved state. **They do not show that changing the limit produces a stable grasp or successful placement.** A sample-wise robot-only collision check is not continuous collision detection; expected cube contacts are excluded and the held cube is not swept along the path. Finger friction, object slip, arm forces and gripper torque are not checked. Large tilt could drop the cube or cause a collision, so lowering the threshold is a hypothesis for a separately controlled simulation test, not a hardware or deployment recommendation.

## Implementation and evaluation priorities

1. Make placement feasibility explicit: log each candidate's target, initial IK seed, position error, tool-axis dot, joint-limit result, and first sampled collision; distinguish `no IK candidate` from a selected placement failure. Consider a small, physically motivated set of alternative transit or approach poses and report their checks. Preserve the original threshold as a baseline. Add carried-object geometry and quantify collision-check resolution before expanding admissible orientation.
2. Add a grasp/holding-state verifier with independently labelled missed, slipped and successful grasps; keep simulator contact and object state only for offline labels. Give unknown or occluded states an explicit abstention instead of converting them to failure.
3. Run predeclared, matched and order-randomized test cases with baseline ACT, retry, Phase II, and frozen Phase III; preserve every start, interruption, failure and success. Evaluate IK checks, success, hold stability, clearance, aborts and time. Add deliberate failures and alternate initial placements only to a development set. Freeze all choices before held-out trials, and state any changed robot scene, checkpoint, camera or horizon.
4. Investigate BORGES-style candidate-conditioned *grasp* selection only after producing observable, stage-specific labels. Before any JEPA-style planner, collect synchronized observations and actions, train an action-conditioned model, and beat persistence/no-action and simple dynamics baselines on held-out episodes. Keep both proposals outside the control loop until they demonstrate incremental value.

## Reproduce

- `scripts/ablate_placement_ik.py --tutorial ../so101-physicalai-tutorial --trace datasets/local-rollouts/phase3-v2-phase2-1030.npz --output reports/phase3-v2/ablation-ik-1030.json`
- `scripts/summarize_recovery_observation.py --output reports/phase3-v2/ablation-observation-errors.json datasets/local-rollouts/phase3-dev-1020.npz datasets/local-rollouts/phase3-dev-1022.npz datasets/local-rollouts/phase3-v2-phase2-1030.npz`

Use the configured local MuJoCo Python environment; the raw camera/action NPZs are local, while the compact results are versioned. IK and image outputs may change with different tutorial, simulator or library versions. Existing strict success evaluates stable support, released jaws, projected footprint and linear speed after lift; it does not measure angular velocity or certify force/stability in the real world.
