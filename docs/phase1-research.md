# Phase 1 research and evaluation

The phase 1 monitor uses the overhead image, gripper command and a tip height calculated from joint positions. The first version detects red objects in the known simulated scene. The wrist camera is logged for subsequent multi-view models and human review. It abstains when the visible evidence is insufficient. It does not control the arm.

Relevant open research:
- [Guardian](https://arxiv.org/abs/2512.01946) studies VLM failure detection from multi-view images and synthesized failures. Its approach informs future comparisons against a learned monitor.
- [AHA](https://arxiv.org/abs/2410.00371) is an open-source VLM for failure detection and explanations. Benchmark its actual compute and accuracy on our camera setup before relying on it.
- [ViFailback](https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset) offers failure diagnosis and correction examples. Its robot, data format and camera views differ from this SO-101 scene.
- [SmolVLA](https://huggingface.co/blog/smolvla) is a 450M-parameter VLA that can serve as a later action-policy comparison, subject to matching its robot action interface.
- [V-JEPA 2](https://github.com/facebookresearch/vjepa2) offers action-conditioned prediction for planning. Its published robot checkpoint has its own embodiment and compute requirements. Benchmark compatibility before using it for SO-101.
- [RecoveryChaining](https://arxiv.org/abs/2410.13979) and [RESPRECT](https://arxiv.org/abs/2401.14858) motivate learned recovery and residual reinforcement learning after reliable outcome labels exist.

## Initial evidence

On the saved 20k ACT failure for demonstration 0, the monitor flagged a suspected miss at 5.05 s simulated time. The reference is the last red cube detection before closure, 35 frames (1.75 s) old. Motion or occlusion during that gap could invalidate the inference. The successful expert demonstration returned unknown because the cube was occluded at the check. These are two exploratory examples, not accuracy estimates.

Next collect labeled, disjoint successful and failed scenes with moved cubes, lighting variation, occlusion and non-red objects. Score missed grasps and false alarms independently, report abstentions and detection latency, and compare against simulator outcomes and human video labels. Use a separate pose set to select thresholds. Record model, scene revision, pose and timing for each run.

Use `--monitor` with `scripts/live_act_eval.py` to show live alerts. Use `matlab/analyze_rollout.m` on a CSV with time_s, gripper_position, gripper_target, cube_z_m. The cube position in this plot is privileged simulator data, used for analysis rather than monitor decisions.

## Minimal evaluation at seed 1001

The ACT checkpoint was run in the live viewer at seed 1001. The monitor flagged suspected miss at 4.10 s simulated time. The ground-truth evaluator reported no bilateral contact, no 5 cm lift, and no placement in the 12 s episode. Its 240 transitions are saved locally as `datasets/local-rollouts/phase1-act-seed1001.npz`.

Offline replay using `scripts/evaluate_monitor_trace.py` reproduced both alerts: the previous demo-0 failed ACT attempt at 5.05 s and seed 1001 at 4.10 s. Both timed out without a 5 cm lift. The separately observed successful expert demonstration returned unknown owing to occlusion. The two failures and one successful example are exploratory, with no new false alarm on the expert; this is not a measured false-positive rate and is insufficient to validate generalization. Time values are simulated time, not wall-clock inference latency. Pose selection and thresholds were not blinded. The previous MATLAB analysis of demo-0 reported peak cube lift 9.17 mm.
