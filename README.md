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

Open [the training notebook](notebooks/train_so101_act_colab.ipynb) in an A100 Colab runtime and give the notebook access to the secrets `github_token` and `hf_token`. The notebook checks private repository access, downloads and checks the pinned dataset, and runs a 20-step smoke test before the separate 20,000-step ACT run. It saves an inference checkpoint with its normalization processors to `MyDrive/so101-act`. The Colab GPU run still needs to be executed.

For local evaluation, use the [Physical AI tutorial's chapter 6 simulator and ACT evaluator](https://github.com/ady95/physicalai_tutorial/tree/ea970233882d19e43fab3acf69cba1b26d10d495/ch06). Its recorder and evaluator match the dataset's task, camera keys, resolution, control rate and six-joint interface. Verify scene compatibility by replaying a demonstration before scoring the trained policy. Enable a live MuJoCo view for the first rollout so the operator can watch the arm during evaluation.

This is a tested, offline candidate-ranking prototype, not a robot policy or a simulator. It accepts externally supplied object regions, estimated widths, stability and reachability, and ranks feasible regions using observed outcomes. Exact object-class matching is deliberate: no visual similarity or physical understanding is claimed. Scores are heuristics, not calibrated probabilities. The caller must validate all inputs and robot motion independently.

Run `python3 -m unittest discover -s tests -v` from this directory. `Memory.save` and `Memory.load` persist JSON experience records. Do not use unverified candidates to move hardware.

## Integration needed

1. Install and validate the SO-101 MuJoCo/LeRobot simulator at https://github.com/Juxi-Technology/so101-mujoco using its documented pinned fork. Verify joint ordering, camera observations, recording, and a scripted grasp before training.
2. Collect compatible simulator demonstrations, train an ACT baseline on a cloud GPU, and preserve checkpoints and dataset metadata. Measure baseline performance on held-out conditions.
3. Implement observation-based success detection and object-region extraction, with a human-labeled fallback. Convert pixel regions to calibrated feasible grasps; pass measured widths and reachability to this module.
4. Integrate controlled retries, logging and memory updates only after verified outcomes. Compare baseline, retry-only, diagnosis without memory, and diagnosis with memory on matched trials.
5. Calibrate and validate separately on the real arm. A sim-trained policy is not assumed to transfer.

The pretrained ACT rollout and physical trials are pending.
