# SO-101 Vision-Guided Failure Detection and Recovery

Project report — evidence snapshot: 26 November 2026

Private repository: https://github.com/Jay-Aditya-16/so101-grasp-recovery

Evidence snapshot commit: 2a7f788a4d7fa47a144716a7d0b0b4a050ab5ede. The report and its renderer are added in a later documentation commit.

## 1. Research objective

The goal is to investigate whether a robot can recognize a grasp mistake and try a more appropriate action during the same task, without retraining its entire base policy. The longer-term inspiration is fast learning from experience. The operational question is whether a layer added to a fixed Action Chunking with Transformers (ACT) policy can improve outcomes on controlled robot manipulation trials.

The current component is best described as a vision-based failure detection and recovery controller. It is hand-engineered and uses perception, gripper feedback, camera geometry, and inverse kinematics. It is not a demonstration of consciousness, human-equivalent understanding, online reinforcement learning, or guaranteed one-shot learning. A person may also repeat or forget a mistake; those abilities are hypotheses to investigate, not assumptions.

## 2. Hardware, software and data

- Development computer: MacBook Air M1 with 8 GB RAM. Local work used CPU inference and MuJoCo simulation with a live viewer.
- Intended robot: SO-101 arm. The results in this report are simulation results; physical-arm recovery has not been evaluated.
- Cloud training: Google Colab L4 GPU. The 20,000-step ACT model was trained on the simulated dataset; the A100 was not used for that run.
- Simulator: MuJoCo, using the SO-101 scene in the Physical AI tutorial at revision ea970233882d19e43fab3acf69cba1b26d10d495. The scene includes an overhead camera, wrist camera, cube and container.
- Dataset: caspar/so101_pickplace_sim_150, pinned revision b4fb342927bb0ee7dcbe4b90472a0c257ce4f5de. That revision contains 117 episodes, 20,124 frames, 20 Hz recordings, overhead and wrist RGB, and six joint-state/action values.
- Training setup: 105 training episodes, 12 held-out episodes, ACT trained to predict 50-action chunks and execute 20 actions before replanning. The local evaluation uses the exported 20,000-step checkpoint.

## 3. Simulator and ACT baseline

We first verified that the MuJoCo scene could run and respond to commands. We replayed recorded demonstration episode 0: it lifted the cube approximately 154 mm and placed it in the box. This was a replay of expert actions, not a successful learned policy.

On the same reconstructed starting pose, the 20,000-step ACT checkpoint did not establish two-jaw contact, lifted the cube at most about 9.2 mm and did not place it. A kinematic comparison indicated roughly 20 mm horizontal gripper offset at the first close command versus about 1.5 mm for the expert reference. This is a single-trajectory observation, not a proven root cause for all failures.

Training loss alone does not establish that an ACT model can complete a task. Full rollouts, calibration, matching camera views, data coverage and held-out tests matter. The dataset and checkpoint are separate from the later recovery traces; unsuccessful policy actions should not be treated as expert demonstrations.

## 4. Phase I — detecting a possible failure

The first monitor uses overhead RGB, commanded gripper position and a gripper-tip height derived from the robot's joint state. It looks for the red cube remaining near an earlier image position while the gripper closes and rises. The monitor emits suspected miss or unknown; it does not control the arm or use simulator object coordinates in its decision.

In four unique exploratory ACT traces, the monitor flagged all three observed no-lift failures and abstained on the single successful placement because the cube was occluded. The alert appeared 0.6–1.8 simulated seconds after the first close command in the three failures. This is not a reliable detection or false-alarm rate: the sample is small, the scene and colour are fixed, and abstention is not a verified success prediction.

Later experiments added two optional cues for an unknown alert: consecutive readings consistent with an empty closed gripper, or a fresh visible red region remaining near its earlier location. These are scene-specific heuristics and may produce false positives. A gripper obstruction can be caused by something other than a successfully grasped object.

## 5. Phase II — geometry and recovery controller

The controller intervenes after a trigger, returns to a viewing pose, locates a visible red target in the overhead image and projects it into the tabletop frame using camera calibration and an assumed 30 mm cube-top height. It computes a top-down approach, closes the gripper, checks joint opening for a plausible obstruction, and attempts lift and placement in the known box.

The inverse-kinematics output is checked against forward kinematics, joint limits and tool direction. Selected joint-interpolation samples are inspected for simulated penetrations. This is not continuous collision checking; it does not validate the held object's swept volume, contact forces, the actual hardware, or physical safety. Gripper-angle evidence is not proof that the correct object is held.

Numerical checks compared the simulator's gripper-position Jacobian against finite differences at eight sampled states. The maximum difference was about 8.87 × 10^-11 metres per radian. This checks the numerical derivative in those simulator states; it does not independently prove that the simulated robot matches a physical arm.

## 6. Matched red-cube experiments

The team compared ACT alone, a return-to-view followed by an ACT retry, and object-aware recovery. Two starting poses used a 25-second simulated horizon. For seed 1020, ACT and retry did not lift, while recovery lifted about 96.3 mm and placed. For seed 1022, ACT peaked at 6.8 mm and retry at 5.9 mm without placement, while recovery lifted about 97.6 mm and placed.

The original successful ACT seed 1003 also completed with no recovery intervention. Seed 1021 exposed an uncertain monitor event and an unreachable box-centre pose; the visual fallback and alternative point inside the box were developed using that case, so it is a development case rather than a held-out result. Seed 1011 needed a longer horizon to settle after release; its 21-second trial is a failure under that time limit.

These are a small number of trials on the same simulated red cube and fixed container. They demonstrate a working prototype in particular conditions, not a robust success rate or general object understanding. The recovery system uses hand-designed motion primitives rather than an independently learned correction policy.

[figure:reports/phase2-matlab-matched.png]

## 7. MATLAB analysis

MATLAB R2025b independently read exported simulator traces and plotted cube-height change and gripper-joint position. It measured approximately 96.28 mm and 97.61 mm peak lift for recovery on the two matched poses, versus at most 6.85 mm for ACT and retry in those trials. The measured gripper position during recovered grasps was roughly 0.15 rad.

These plots corroborate the recorded MuJoCo trajectories; they are not an independent real-world physics validation. Simulator object and contact values are privileged evaluation labels, not inputs to the recovery controller. The analysis scripts and CSV results are in the repository.

## 8. Two cameras and monocular depth

The overhead camera gives global task geometry; the wrist view provides a closer perspective. An experimental mode connects Depth Anything V2 Small to the wrist view as a relative-depth ordering check before descent. It also compares the wrist-projected position with the overhead estimate and abstains if the views disagree beyond a fixed threshold.

Depth Anything V2's standard checkpoint predicts relative depth, not calibrated metric distance. In the two initial wrist-depth tests the robot placed the cube, but the existing recovery without depth also placed. Wrist position errors were approximately 7.5–7.6 mm, compared with overhead errors of approximately 1.7–2.1 mm. On a later test, errors were about 7.32 mm and 0.38 mm respectively. No accuracy improvement from depth has been demonstrated.

CPU inference was around 1.5–1.7 seconds in the recorded wrist trials, with a separate top-view call taking about 13 seconds. The sample is too small for a latency benchmark. More calibration, held-out views and object types are needed before relying on depth estimates for motion.

## 9. More complex cases and drawer test

The existing pick-trained ACT policy and recovery layer were evaluated on additional MuJoCo scenes, one paired seed per condition. The policy and recovery assumptions remained unchanged. A successful placement was not inferred merely from a lift.

| Scene | ACT alone | ACT with recovery | Key finding |
| --- | --- | --- | --- |
| 20 mm red cube | No placement | No placement | Visible red region fell below the detector gate. |
| 30 mm green cube | No placement | No placement | Recovery's detector was red-specific. |
| 36 mm red sphere | No placement | Lifted 90.3 mm; dropped | Gripper feedback detected loss during transport. |
| Red cuboid and green cube | No placement | No placement | The red cuboid grasp failed; the green cube was a distractor. |
| Drawer with red handle | 0 mm opening | 9.01 mm opening | Both failed a 60 mm sustained-opening criterion. |

The original 30 mm red-cube recovery control still placed, lifting 96.3 mm. In the two-object case, simulator state showed approximately 0.15 mm maximum motion of the green distractor, including normal settling; this does not prove it was never touched. The drawer was a simple MuJoCo slide-joint fixture, not a calibrated household drawer. An isolated applied-force check confirmed its slider could move; it did not demonstrate robot drawer manipulation.

The policy's limited training distribution is a plausible contributor to failure on new objects and tasks, but it is not the only cause. The recovery controller itself assumes red colour, fixed height, a top-down grasp and a fixed placement box. The drawer handle was treated as a pick target, and the controller has no drawer-pulling skill. We cannot attribute these failures solely to the base policy's training.

[figure:reports/complex-v1/matlab-comparison.png]

## 10. Recordings and reproducibility

Live simulations record overhead and wrist frames, joint states, actions, simulator outcome labels, and recovery events. Selected successes, failures and varied-object trials are stored in the private repository under datasets/local-rollouts. The complex-case evaluation validated alignment and outcome flags for 11 recorded archives. Failed actions remain failure data, not expert action labels.

The repository includes the ACT training notebook, simulator runners, monitoring and recovery code, MATLAB analysis, test cases, scene definitions, results and the project roadmap. The model checkpoint and source dataset are retained separately from the repository. A set of four short successful simulation videos and still images was exported to Downloads/SO101-Phase2-Success.

## 11. Remaining phases and research plan

Phase III: record verified recovery contexts, candidate actions and outcomes. Start with a small contextual bandit or outcome memory over feasible candidates, then compare with the current fixed recovery controller on matched held-out trials. Update from outcomes, keep uncertainty and abstention, and measure whether experience improves decisions. One-attempt improvement is not guaranteed.

Phase IV: evaluate compact vision-language models and action-conditioned world models or JEPA-style representations only if they add measurable value over the simpler controller. Compare any VLA or planner under the same observations, task conditions, limits and evaluation protocol. Check embodiment, compute and latency on the M1 and Colab L4.

Phase V: calibrate the physical SO-101 and its cameras, first run a supervised observation-only evaluation, then separately validate bounded physical actions and compare matched physical trials. No simulation result establishes physical safety or sim-to-real transfer.

Before making broader claims, expand held-out tests across objects, poses, colours, camera changes, occlusions and successful baseline cases; measure false alarms, abstentions, timing, collision and slip; add stronger observation-based grasp verification and carried-object path checks; and obtain a task-specific drawer strategy or training. The current Phase II is a working red-cube simulator prototype, not general object or drawer-task capability.

## 12. Sources and project records

- ACT / ALOHA paper: https://arxiv.org/abs/2304.13705
- LeRobot: https://github.com/huggingface/lerobot
- Source dataset: https://huggingface.co/datasets/caspar/so101_pickplace_sim_150
- MuJoCo: https://github.com/google-deepmind/mujoco
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- Visual imitation-learning generalization study: https://arxiv.org/abs/2307.03659
- Local research records: docs/phase1-research.md, docs/phase2-integrated.md, docs/phase2-two-view-depth.md, docs/phase2-matlab-evaluation.md, docs/phase2-complex-validation.md and docs/project-roadmap.md.
