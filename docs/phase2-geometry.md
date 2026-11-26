# Phase 2: object-relative geometry, first experiment

The first component estimates the red object's image centroid and projects it onto an assumed horizontal plane using the known overhead camera intrinsics and extrinsics. It computes the difference from the gripper site obtained through forward kinematics. Simulator cube coordinates are read only after inference to calculate error. This runs on saved camera and joint observations; it sends no robot commands.

The assumed top plane is 30 mm above the tabletop, matching this specific cube. The estimator abstains for multiple red regions, border clipping, or a visible region under 60 pixels. The threshold was chosen after inspecting these development traces; it is not validated confidence and can reject smaller distant objects or accept misleading partial masks. The method does not infer depth, full shape, orientation, uncertainty bounds, reachability, or collisions. It does not handle arbitrary objects or fuse the wrist view.

## Observations

On the four existing ACT traces, initial image localization errors against simulator ground truth were 1.90, 2.06, 1.94 and 2.68 mm. Before the visibility gate, estimates at the three failure-alert frames had errors of 13.37, 9.07 and 8.05 mm due to small, partially visible red regions; the successful trial was fully obscured. The visibility gate then abstained at all four alert frames. These are development examples, not independent accuracy measurements. The current geometry is consequently **not ready to guide recovery**: the key issue is recovering a reliable view after occlusion, not finding a correction from these alert frames.

The precise per-trial inputs and initial/alert results are in `reports/phase2-initial-geometry.json` and `reports/phase2-alert-geometry.json`. The trace archives remain local in `datasets/local-rollouts`, excluded from Git. See `scripts/evaluate_geometry.py --help` for reproduction; `--frame alert` selects the first Phase 1 monitor event. The earlier 8–13 mm figures are from the ungated development experiment.

## Research and design implications

- [LeRobot ACT documentation](https://huggingface.co/docs/lerobot/act) and the [ACT paper](https://arxiv.org/abs/2304.13705): keep the trained ACT policy as an independent baseline, preserving the observation and action contract and comparing matched rollouts.
- [LeRobot open-source implementation](https://github.com/huggingface/lerobot): reuse policy processors and dataset conventions rather than assuming raw joint or image scaling.
- [Active Observation for Robust Visual Servoing](https://arxiv.org/abs/2611.30658): motivates actively restoring visibility and checking relative object/gripper pose; its reported results concern a different setup and are not evidence of performance here.
- [FoRe](https://arxiv.org/abs/2611.27011): motivates recovery from post-failure states and later local reinforcement learning. Its force inputs and embodiment differ from our setup.
- [RecoveryChaining](https://arxiv.org/abs/2410.13979): motivates explicit recovery behaviors. Our estimator currently has no trained recovery policy.

## Next implementation work

Add synchronized wrist-view interpretation and a conservative observation-recovery behavior in simulation. Validate camera calibration and object estimates at post-failure frames on held-out traces. Then generate object-relative grasp candidates, verify reachability and collisions, and integrate bounded moves with the fixed ACT policy. Compare matched baseline, retry-only, and recovery trials only once those pieces exist. A stale pre-grasp object position is not a safe replacement when the object may have moved.
