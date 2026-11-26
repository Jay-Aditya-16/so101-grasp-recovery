# SO-101 grasp recovery: initial component

The immediate milestone is a working local SO-101 simulator and policy evaluation. This existing recovery experiment is retained for later; it is not integrated into that milestone.

## Local simulator smoke test

Clone [Juxi-Technology/so101-mujoco](https://github.com/Juxi-Technology/so101-mujoco) and run the smoke test from this repository using a Python environment with MuJoCo and NumPy installed:

```sh
gh repo clone Juxi-Technology/so101-mujoco ../so101-mujoco
python scripts/smoke_so101.py ../so101-mujoco
```

On the MacBook Air M1, MuJoCo 3.17.0 compiled simulator commit `2c6a6ed455b2949b07f2eaf2bd2a2a81145fd20c`, stepped the scene for two simulated seconds, and moved the base joint by 0.248 rad in response to a 0.25 rad target. The scene uses the upstream SO-ARM100 Menagerie asset as the SO-101 approximation. Camera rendering, dataset recording, and ACT rollout have separate acceptance checks.

## Pretrained reach-and-close baseline

The [SO-101 Sim2Real project](https://github.com/isaka1022/so101-sim2real) bundles a behavior-cloned ONNX policy and its matching MuJoCo scene. The policy takes a 17-value state vector that includes simulator-provided cube position and returns four control values. The upstream metric requires the gripper to be closed within 5 cm of the cube. Lift and placement need separate tests.

Run with Python 3.13, MuJoCo 3.3.1, ONNX Runtime 1.32.0, and the scene package's Gymnasium and NumPy dependencies:

```sh
gh repo clone isaka1022/so101-sim2real ../so101-sim2real
git -C ../so101-sim2real checkout addc4e13f22f6f8ac1e29d15b78380b02387a1e5
cd ../so101-sim2real
python -m pip install 'mujoco==3.3.1' 'onnxruntime==1.32.0' ./packages/lerobot_env_so101
python sim/eval_reachclose_policy.py --episodes 20 --max-steps 100 --seed 1000
```

The M1 CPU run reached and closed in 20 of 20 seeded trials. Mean final end-effector distance from the cube was 30.7 mm. In a separately rendered seed-1000 rollout, the metric first passed at step 8. These results cover the upstream randomized cube positions and state inputs.

## Custom ACT training on Colab

Use [caspar/so101_pickplace_sim_150](https://huggingface.co/datasets/caspar/so101_pickplace_sim_150) at revision `b4fb342927bb0ee7dcbe4b90472a0c257ce4f5de`. Despite the repository name, this revision contains 117 episodes and 20,124 frames at 20 Hz. The task is to place a red cube in a blue box. The dataset has top and wrist RGB video at 320 × 240, six measured joint positions, and six joint targets. The first, middle, and last frames decoded locally with PyAV through LeRobot and matched that task and input contract.

Open [the training notebook](notebooks/train_so101_act_colab.ipynb) in an NVIDIA L4 Colab runtime and give it access to the secrets `github_token` and `hf_token`. Run the 20-step smoke test at the intended batch size of 8 before starting the separate 20,000-step run. If the smoke test runs out of GPU memory, set `BATCH_SIZE = 4` and rerun it. The notebook exports completed checkpoints with normalization processors and training state to `MyDrive/so101-act`. L4 throughput and headroom need to be measured in Colab.

LeRobot 0.6.1 holds out the last 12 of 117 episodes for offline validation. Normalization statistics come from the 105 training episodes. The model predicts 50 joint targets per camera observation at 20 Hz and executes 20 before replanning. Use the saved 5k, 10k, 15k and 20k checkpoints to compare complete rollouts. A small validation loss helps diagnose training while task completion remains the selection metric. [Original ACT paper](https://arxiv.org/abs/2304.13705) · [Robomimic study on checkpoint selection](https://proceedings.mlr.press/v164/mandlekar22a.html)

For local evaluation, use the [Physical AI tutorial's chapter 6 simulator and ACT evaluator](https://github.com/ady95/physicalai_tutorial/tree/ea970233882d19e43fab3acf69cba1b26d10d495/ch06). Its recorder and evaluator match the dataset's task, camera keys, resolution, control rate and six-joint interface. Verify scene compatibility by replaying a demonstration before scoring the trained policy. Compare checkpoints on the same development poses, then evaluate the selected checkpoint on a separate set of test poses. Report grasp, lift and settled placement separately. Keep camera placement and calibration fixed and open a live MuJoCo view before the first rollout. [SO-101 setup and pose coverage experiments](https://huggingface.co/blog/sherryxychen/train-act-on-so-101)

The local live runner uses the tutorial scene and the saved ACT processors. On macOS, run it with `mjpython` from an environment with MuJoCo 3.3.1 and LeRobot 0.6.1. Given a downloaded Colab checkpoint archive and a checkout of the tutorial revision above:

```sh
unzip ~/Downloads/020000.zip 'pretrained_model/*' -d checkpoints/single-cube-020000
mjpython scripts/live_act_eval.py --tutorial ../so101-physicalai-tutorial \
  --checkpoint checkpoints/single-cube-020000/pretrained_model --episodes 1
```

The viewer opens before motion. Closing it stops the evaluation. Each completed episode reports bilateral contact, peak lift, and whether the cube lifted at least 5 cm and then stayed supported by the box floor for at least half a second. A camera-only preflight is available with `--scene-check`. Reuse the same `--seed` and episode count for checkpoint comparisons.

To check the simulator independently of ACT, replay targets from the pinned training dataset with the same live viewer:

```sh
mjpython scripts/replay_act_demo.py --tutorial ../so101-physicalai-tutorial \
  --dataset datasets/caspar-single-cube --episode 0
```

The replay estimates the cube's start from the first overhead frame and checks that the box and starting joints match the scene. It applies the saved joint targets at 20 Hz and stops once the placement metric passes. Episode 0 passed locally. It lifted the cube 154 mm and settled it on the box floor after 154 targets, with a mean absolute joint tracking error of 0.01083 rad against the saved observations. This checks one trajectory and provides a successful reference for policy debugging.

Run ACT on the same reconstructed start and capture its transitions with `--demo-dataset datasets/caspar-single-cube --demo-episode 0 --capture datasets/local-rollouts/act-020000-demo-000.npz`. The viewer stays live during motion. The archive keeps both RGB streams, MuJoCo positions and velocities, and cube positions for the initial observation and every subsequent observation. For transition `i`, use observation `i`, `action[i]`, and observation `i+1`. Contact, box support, settled state, and reward describe the result of that action. Reward is 1 only for verified placement. `terminated` marks placement and `truncated` marks the time limit or a closed viewer. Reference joint observations and expert commands are separate arrays and share the same 20 Hz clock.

Keep the original LeRobot dataset as the source of expert camera frames and imitation targets. The policy capture is a separate local dataset for failure analysis, recovery prediction, or RL. Failed commands should not become imitation targets. A correction demonstrated from a failed state can provide new imitation targets once its lift or placement is verified.

In the matched episode-0 trial, the 20k checkpoint reached no bilateral contact, lifted the cube 9.2 mm at most, and timed out without placing it. In the target and kinematics logs, the first gripper command below 0.1 rad came at 3.25 s for ACT and 2.70 s for the demonstration. At that point, ACT's gripper site was about 20 mm off the initial cube center horizontally, compared with about 1.5 mm in the recorded joint state. This timing and offset describe one trajectory. Use additional independent rollouts before attributing failures to the same cause.

This is a tested, offline candidate-ranking prototype, not a robot policy or a simulator. It accepts externally supplied object regions, estimated widths, stability and reachability, and ranks feasible regions using observed outcomes. Exact object-class matching is deliberate: no visual similarity or physical understanding is claimed. Scores are heuristics, not calibrated probabilities. The caller must validate all inputs and robot motion independently.

Run `python3 -m unittest discover -s tests -v` from this directory. `Memory.save` and `Memory.load` persist JSON experience records. Do not use unverified candidates to move hardware.

## Integration needed

1. Install and validate the SO-101 MuJoCo/LeRobot simulator at https://github.com/Juxi-Technology/so101-mujoco using its documented pinned fork. Verify joint ordering, camera observations, recording, and a scripted grasp before training.
2. Collect compatible simulator demonstrations, train an ACT baseline on a cloud GPU, and preserve checkpoints and dataset metadata. Measure baseline performance on held-out conditions.
3. Implement observation-based success detection and object-region extraction, with a human-labeled fallback. Convert pixel regions to calibrated feasible grasps; pass measured widths and reachability to this module.
4. Integrate controlled retries, logging and memory updates only after verified outcomes. Compare baseline, retry-only, diagnosis without memory, and diagnosis with memory on matched trials.
5. Calibrate and validate separately on the real arm. A sim-trained policy is not assumed to transfer.

## Project plan and Phase 1 status

See [the complete project roadmap](docs/project-roadmap.md) for the phased plan, research question, hardware constraints, evaluation gates, and physical deployment sequence. [Phase 1 research and results](docs/phase1-research.md) documents the monitor and the four unique exploratory ACT trials. The monitor flagged all three observed no-lift failures and abstained on the one successful lift/place because the cube was occluded; this small sample does not establish generalized accuracy and is not sufficient to enable automatic recovery.

The trained ACT baseline has been evaluated in simulation; physical-arm trials and the recovery layer are pending.
