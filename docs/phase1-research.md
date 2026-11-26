# Phase 1 research and evaluation

The phase 1 monitor uses the overhead image, gripper command and a tip height calculated from joint positions. The first version detects red objects in the known simulated scene. The wrist camera is logged for subsequent multi-view models and human review. It abstains when the visible evidence is insufficient. It does not control the arm.

Relevant open research:
- [Guardian](https://arxiv.org/abs/2512.01946) studies VLM failure detection from multi-view images and synthesized failures. Its approach informs future comparisons against a learned monitor.
- [AHA](https://arxiv.org/abs/2410.00371) is an open-source VLM for failure detection and explanations. Benchmark its actual compute and accuracy on our camera setup before relying on it.
- [ViFailback](https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset) offers failure diagnosis and correction examples. Its robot, data format and camera views differ from this SO-101 scene.
- [SmolVLA](https://huggingface.co/blog/smolvla) is a 450M-parameter VLA that can serve as a later action-policy comparison, subject to matching its robot action interface.
- [V-JEPA 2](https://github.com/facebookresearch/vjepa2) offers action-conditioned prediction for planning. Its published robot checkpoint has its own embodiment and compute requirements. Benchmark compatibility before using it for SO-101.
- [RecoveryChaining](https://arxiv.org/abs/2410.13979) and [RESPRECT](https://arxiv.org/abs/2401.14858) motivate learned recovery and residual reinforcement learning after reliable outcome labels exist.

## Evaluation and evidence (2026-11-26)

The monitor was evaluated on live-viewer rollouts of the saved 20k ACT checkpoint. Offline replay of captured traces uses only top-camera RGB, commanded gripper position, and tip height from forward kinematics for the alert decision. MuJoCo outcome fields are used only as labels. Duplicate trace content is removed by SHA-256 before aggregation.

| Unique trial | Ground-truth outcome | Monitor | Alert after first close command |
|---|---|---|---:|
| Matched demo-0 pose | No 5 cm lift | `suspected_miss` | 1.8 s |
| Seed 1001 | No 5 cm lift | `suspected_miss` | 0.6 s |
| Seed 1002 | No 5 cm lift | `suspected_miss` | 1.6 s |
| Seed 1003 | Lift and settled placement | `unknown` (cube occluded) | 1.7 s |

Across these **four unique exploratory trials**, it flagged all 3 no-lift failures (3/3) and made no suspected-miss alert on the single success (0/1). It abstained on that success. Observed trial coverage is 3/4; it did not verify any success. These counts are too small and scene-specific to estimate general accuracy, recall, or false-alarm rate. Two demo-0 captures were identical and are counted once.

The alert delay relative to first commanded closure was 0.6–1.8 simulated seconds. This is closure-to-alert timing, not latency from the physical onset of failure and not wall-clock inference time. The detector uses the last visible cube position before closing, so motion or occlusion during that gap may invalidate the inference.

Next collect labeled, disjoint successful and failed scenes with moved cubes, lighting variation, occlusion and non-red objects. Score missed grasps and false alarms independently, report abstentions and detection delay, and compare against simulator outcomes and human video labels. Use separate poses for threshold selection and final evaluation. Record model, scene revision, pose and timing for each run.

Use `--monitor` with `scripts/live_act_eval.py` to show live alerts. Reproduce the four-trial offline summary (with rollout archives available locally) using:

```sh
PYTHONPATH=/private/tmp/so101-lerobot-public061 HF_HOME=/private/tmp/so101-dataset-validation \
  /Users/jayaditya/so101-lerobot-juxi/.venv/bin/python scripts/evaluate_monitor_trace.py \
  --tutorial ../so101-physicalai-tutorial --json-out reports/phase1-monitor-evaluation.json \
  datasets/local-rollouts/act-020000-demo-000.npz \
  datasets/local-rollouts/phase1-act-seed1001.npz \
  datasets/local-rollouts/phase1-act-seed1002.npz \
  datasets/local-rollouts/phase1-act-seed1003.npz
```

The evaluator emits deduplicated trace hashes, outcome labels, alerts, abstentions, and closure-to-alert times. Run `matlab/analyze_rollout.m` on a CSV with time_s, gripper_position, gripper_target, cube_z_m. Cube position in MATLAB plots is privileged simulator data used for analysis, never for monitor decisions.

The MATLAB analysis of demo-0 measured 9.17 mm peak lift. Phase 1 completes the monitoring prototype and minimal data/analysis pass. It does **not** establish a generalized or deployment-ready failure detector; expand the held-out set before automated recovery uses these alerts.
