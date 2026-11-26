# SO-101 Online Grasp Recovery: Project Roadmap

## Objective

Test whether a small, object-aware recovery layer can improve a fixed imitation policy after a grasp failure, without retraining the base policy. The project uses an SO-101 arm, LeRobot-compatible demonstrations, MuJoCo simulation, and later physical trials. “Childlike learning” is the motivation; the engineering claim is narrower: detect a failed attempt, infer a plausible cause from observations, choose a grounded correction, verify the result, and retain evidence from the outcome.

This is not a claim of robot consciousness or human-equivalent lifelong learning. A single failed action does not always reveal the cause, and humans also forget or repeat mistakes. The system must be allowed to abstain when evidence is ambiguous.

## Research question and hypotheses

**Question:** Can an object-aware online recovery layer raise grasp/lift/place success for a fixed ACT policy on held-out object poses and appearances, with fewer attempts and no base-policy retraining?

- **H1:** A visual/proprioceptive monitor that detects a miss or failed lift will identify more recoverable failures than a fixed retry rule, at a measured false-alarm cost.
- **H2:** Diagnosis grounded in the visible object, gripper, and their relative geometry will outperform blind fixed-direction nudges and blind retries.
- **H3:** A small outcome memory or contextual bandit can improve candidate selection across repeated, sufficiently similar situations, while abstaining on unfamiliar ones.
- **H4:** Gains in simulation transfer only after separate camera, calibration, dynamics, and safety checks on the physical arm.

These are hypotheses to test, not promised outcomes. Success is defined by independently observed contact, lift, and stable placement—not training loss or a visually plausible motion.

## Current setup and resource constraints

- **Development laptop:** MacBook Air M1, 8 GB RAM. Use it for CPU MuJoCo rollouts, rendering, data inspection, unit tests, and small inference models. Avoid training large vision transformers locally.
- **Cloud training:** Google Colab NVIDIA L4 (not A100); Kaggle GPUs are optional when available. Keep datasets/checkpoints in Drive or Hugging Face and avoid repeatedly downloading them to the laptop.
- **Robot:** SO-101 arm, available for physical validation after the simulator work. Start with a bounded, supervised task and conservative joint limits.
- **Dataset:** pinned `caspar/so101_pickplace_sim_150` revision `b4fb342927bb0ee7dcbe4b90472a0c257ce4f5de`, locally staged as `datasets/caspar-single-cube` (117 episodes, 20,124 frames, 20 Hz, overhead and wrist RGB, six joint positions and targets).
- **Base policy:** ACT checkpoint trained for 20,000 steps on Colab L4; exported archive was `020000.zip`. The checkpoint is evaluated in the compatible SO-101 MuJoCo tutorial scene. Do not describe the expert demonstration replay as a learned-policy success.
- **Simulator/source:** SO-101 MuJoCo tutorial checkout pinned at `ea970233882d19e43fab3acf69cba1b26d10d495`; the separate upstream Juxi SO-101 MuJoCo smoke test is also documented in the README.
- **Reproducibility:** save code, configs, pinned revisions, metrics, and compact evaluation traces in this private GitHub repository. Keep large raw datasets/checkpoints in their source storage or Git LFS/Drive; do not commit credentials or silently commit bulky generated media.

## Phases and gates

### Phase 0 — Reproducible simulator and baseline (completed as an initial smoke test)

1. Install/pin the SO-101 MuJoCo scene and compatible LeRobot code.
2. Confirm the simulator steps, joints respond, cameras render, and the recorded demonstration can be replayed.
3. Train/load the fixed ACT baseline and define deterministic evaluation poses/seeds.
4. Score grasp contact, lift (5 cm threshold), and stable placement separately. Preserve a live viewer during every rollout.

**Evidence so far:** simulator smoke test passed; replay of dataset episode 0 lifted the cube 154 mm and placed it. A matched rollout of the 20k ACT checkpoint failed: no bilateral contact, 9.2 mm maximum lift, and no placement. The suspected ~20 mm horizontal gripper offset at closure is a single-trajectory diagnosis only.

**Gate:** stable reproduction of both expert replay and baseline result, plus a frozen train/development/test pose split before comparing recovery variants.

### Phase 1 — Failure monitoring and outcome data (prototype and minimal exploratory evaluation complete)

1. Capture RGB views, joint observation/command, timestamps, and episode outcome for baseline rollouts.
2. Run the current known-scene red-cube visual monitor with proprioception-derived gripper height. It emits `suspected_miss` or `unknown`; it does not control motion and must not read simulator ground truth.
3. Use MuJoCo object/contact state only as an evaluation label/oracle. Keep it out of the deployed monitor input.
4. Measure missed detections, false alarms, abstentions, and detection delay on disjoint positive/negative scenes; include movement, occlusion, and lighting changes.
5. Review a small number of traces in MATLAB and retain compact, reproducible traces and plots.

**Evidence so far:** four unique live-viewer ACT trials were analyzed (three no-lift failures and one successful lift/place). All three failures received `suspected_miss`; the successful trial abstained as `unknown` due to occlusion. It raised no false alarm on that one success, but verified no successes. Closure-to-alert delay was 0.6–1.8 simulated seconds. Duplicate traces are removed by content hash. These are exploratory counts, not a reliable false-alarm or generalization estimate. The current detector is scene/color-specific and heuristic. MATLAB analysis of the first failure measured 9.17 mm peak lift.

**Phase 1 exit:** the monitor is implemented, non-controlling, replayable, and evaluated on a minimal labeled set with failures, a success, and explicit abstention accounting. **Deployment gate remains open:** collect a larger held-out set with predeclared thresholds and enough successes/failures to estimate misses, false alarms, coverage, and latency. Do not enable automatic recovery based on this sample. If monitor performance is weak, improve sensing/labels before recovery.

### Phase 2 — Object-aware diagnosis and grounded correction (in progress)

Goal: replace unguided directional retries with corrections supported by the observed scene.

Initial object/gripper geometry is implemented and assessed on four existing traces. It abstains at all four monitor-alert frames because of occlusion or insufficient visible area; recovery control is not implemented. See [Phase 2 geometry](phase2-geometry.md).

An initial monocular depth study is documented in [Phase 2 depth](phase2-monocular-depth.md): the model separated the visible cube from its surroundings initially but did not reliably order partial cube and gripper pixels at alerts. Its relative output requires independent metric calibration.

1. Estimate the target object’s image region, center, orientation, approximate footprint/width, and confidence from the overhead view; use wrist view as a second, local view. Start with robust color/shape geometry in the fixed simulated scene and human review. Later compare a compact detector/segmenter or VLM; do not assume an LLM has precise metric geometry.
2. Estimate gripper pose from joint state/forward kinematics and camera calibration. Compute object-to-gripper offset in a tabletop coordinate frame, with uncertainty. Fuse viewpoints only when calibrated and time-synchronized.
3. Classify failure stage from observable evidence: missed approach, no closure/contact, slip/no lift, collision/occlusion, or unknown. Keep simulator contact/object state as labels for evaluation, not as controller input.
4. Generate a small set of **object-relative grasp candidates**: approach direction and height based on object footprint/orientation, gripper width, workspace reachability, and collision checks. Candidate choices are derived from geometry and task constraints, not arbitrary left/right/up/down perturbations.
5. Rank feasible candidates using uncertainty-aware scoring learned from demonstrations/outcomes; if confidence is low or no candidate is safe, stop and request human intervention.
6. Execute only short, bounded approach/close/lift probes in simulation; reobserve after each probe. Validate contact and lift before proceeding to placement.
7. Compare the fixed baseline with (a) retry-only, (b) geometry-aware diagnosis without memory, and (c) geometry-aware correction. Use identical held-out initial poses, paired seeds, and confidence intervals.

**Gate:** recovery improves held-out task success over baseline and retry-only without unacceptable collisions, unsafe commands, or false recovery triggers. If not, report the failure modes and revise perception/candidate geometry.

### Phase 3 — Online outcome memory and reinforcement learning (after Phase 2 is safe)

1. Log each attempted correction with a context (object-relative geometry, estimated object family, grasp stage, uncertainty), action/candidate, and verified reward/outcome.
2. Start with a contextual bandit over the safe candidate set, not unconstrained continuous-action RL. Reward stages can be: valid contact, stable lift, successful placement; penalize collision, dropped object, excessive motion, and timeout.
3. Update only from verified outcomes. Keep a conservative prior, confidence bounds, a per-episode retry limit, and an abstain/human fallback.
4. Compare no-memory recovery against persistent memory on held-out repetitions and object/pose shifts. Test whether memory helps similar contexts and correctly avoids transfer to dissimilar objects.
5. Consider offline RL/residual policy learning only after enough diverse, correctly labeled failure-and-recovery transitions exist. Do not train directly on failed ACT actions as if they were expert targets.

**Gate:** statistically credible improvement over Phase 2 without unsafe exploration or regression on the original task. Any online update is logged, reversible, and evaluated before being retained.

### Phase 4 — World models, compact VLM/SLM, and policy comparisons (optional research track)

1. Establish the geometric recovery baseline first.
2. Evaluate an image-language model for semantic object/failure descriptions and candidate explanation, with explicit uncertainty and no direct low-level motor control.
3. Evaluate a small learned visual dynamics/world model only if the dataset supports predicting action-conditioned outcomes. Compare predicted outcomes with MuJoCo and held-out traces.
4. Compare against a VLA or a VLM/LLM planner under the same objects, camera views, action limits, and evaluation seeds. Separate semantic understanding from metric localization/control.
5. V-JEPA 2/JEPA-style representations are a research candidate, not an assumed solution: verify checkpoint, hardware, embodiment, licensing, and inference costs before integration. Likewise, do not claim an SLM/world model is “active intelligence” without an ablation that shows measurable benefit.

**Gate:** retain a learned model only if it measurably improves held-out recovery over the simpler geometry/bandit layer at acceptable latency and compute.

### Phase 5 — Physical SO-101 validation and deployment (after simulation gates)

1. Recalibrate and verify joint ordering, limits, camera views, workspace, gripper offsets, and emergency stop.
2. Run the baseline and recovery in supervised, low-speed trials with a clear area and soft/light objects. Begin with zero autonomous recovery; shadow-log proposals first.
3. Enable one bounded correction at a time, with collision/joint-limit checks, timeouts, maximum attempts, and immediate operator stop.
4. Compare matched baseline/recovery trials; report failures and safety interruptions alongside success.
5. Only deploy settings that pass the physical validation. Simulation success is not evidence of sim-to-real transfer.

## Evaluation protocol

- Freeze development poses separately from final held-out test poses. Do not tune thresholds on the reported test set.
- Use paired initial conditions and the same random seeds across baseline and variants.
- Report trial count, success and stage rates, confidence intervals, abstention, recovery attempts, time-to-success, collisions/drops, and failure categories.
- Report detector alert quality separately from task success. Report simulated time separately from wall-clock inference latency.
- Keep these baselines distinct: raw ACT; ACT plus fixed retry; ACT plus object-aware correction; and, only if feasible, VLM/VLA planning. Use identical observations and safety constraints.
- Show the live MuJoCo viewer during interactive runs. Save traces and videos for later review, but do not substitute a saved recording for a requested live view.
- MATLAB is for offline trace analysis/plots and independent calculations; MuJoCo/LeRobot remain the simulation and policy execution stack.

## Research and implementation references

The following references informed the roadmap and are starting points, not proof that a component will work on this setup:

- [ACT / ALOHA](https://arxiv.org/abs/2304.13705): action chunks and temporal ensembling mitigate compounding error in imitation learning; does not provide general online error correction by itself.
- [LeRobot](https://github.com/huggingface/lerobot): dataset, policy training, and SO-101 ecosystem; pin versions and match feature/action contracts.
- [DAgger](https://proceedings.mlr.press/v15/ross11a.html): addresses covariate shift through expert corrections; useful conceptual baseline, but requires supervision/correction data.
- [ThriftyDAgger](https://arxiv.org/abs/2109.08273): budget-aware intervention and interactive imitation learning.
- [RecoveryChaining](https://arxiv.org/abs/2410.13979): motivates chaining recovery behaviors from failures; embodiment and task fit need checking.
- [RESPRECT](https://arxiv.org/abs/2401.14858): residual reinforcement learning for manipulation; motivates learning a correction on top of a base policy after reliable outcome labels exist.
- [V-JEPA 2](https://github.com/facebookresearch/vjepa2): action-conditioned predictive world-model research; check robot checkpoint and compute compatibility before use.
- [SO-101 ACT data-collection lessons](https://huggingface.co/blog/sherryxychen/train-act-on-so-101): camera consistency, pose coverage, staged success metrics, and held-out evaluation.
- [MuJoCo](https://github.com/google-deepmind/mujoco): local physics simulation used for reproducible trials and privileged outcome labels.

## Repository and artifact policy

- This is a private GitHub repository. All source, plans, configs, tests, and compact results enter through a branch and PR; merge only after review/checks.
- Keep Hugging Face and Google credentials out of source and notebooks. Use Colab secret names already configured by the user (`hf_token`, `github_token`) without printing values.
- Dataset/checkpoint/video size may make Git unsuitable. Store large artifacts in Drive/Hugging Face or Git LFS; commit a manifest with source, revision, hashes, paths, and reproduction commands.
- Preserve provenance and label simulation privileged state clearly. Never represent a simulation result as a physical-robot result.

## Current status snapshot (2026-11-26)

| Area | Status | Evidence/next action |
|---|---|---|
| Local SO-101 simulator | Initial smoke test passed | Expand deterministic scene/pose acceptance tests |
| Dataset and ACT checkpoint | Dataset revision and 20k checkpoint available | Add checkpoint provenance/hash manifest if archive is retained |
| Expert replay | Episode 0 passes | One demonstration only |
| ACT baseline | Failed on matched pose and seed 1001 | Two failure traces; more disjoint trials needed |
| Phase 1 monitor | Minimal exploratory evaluation complete | 3/3 failures alerted; 1/1 success abstained; 4 unique trials; no deployment claim |
| MATLAB trace analysis | Ran on initial failure | 9.17 mm max lift; extend to labeled batch analysis |
| Phase 2 object-aware correction | Geometry prototype in progress | Initial localization evaluated; alert-frame occlusion blocks reliable correction |
| Online RL/memory | Planned | Wait for safe candidate set and reliable reward labels |
| Physical arm recovery | Not yet evaluated | Complete simulation gates and safety checklist first |

