# SO-101 grasp recovery: initial component

The immediate milestone is a working local SO-101 simulator and policy evaluation. This existing recovery experiment is retained for later; it is not integrated into that milestone.

## Local simulator smoke test

Clone [Juxi-Technology/so101-mujoco](https://github.com/Juxi-Technology/so101-mujoco) and run the smoke test from this repository using a Python environment with MuJoCo and NumPy installed:

```sh
gh repo clone Juxi-Technology/so101-mujoco ../so101-mujoco
python scripts/smoke_so101.py ../so101-mujoco
```

On the MacBook Air M1, MuJoCo 3.17.0 compiled simulator commit `2c6a6ed455b2949b07f2eaf2bd2a2a81145fd20c`, stepped the scene for two simulated seconds, and moved the base joint by 0.248 rad in response to a 0.25 rad target. The scene uses the upstream SO-ARM100 Menagerie asset as the SO-101 approximation. Camera rendering, dataset recording, and ACT rollout have separate acceptance checks.

This is a tested, offline candidate-ranking prototype, not a robot policy or a simulator. It accepts externally supplied object regions, estimated widths, stability and reachability, and ranks feasible regions using observed outcomes. Exact object-class matching is deliberate: no visual similarity or physical understanding is claimed. Scores are heuristics, not calibrated probabilities. The caller must validate all inputs and robot motion independently.

Run `python3 -m unittest discover -s tests -v` from this directory. `Memory.save` and `Memory.load` persist JSON experience records. Do not use unverified candidates to move hardware.

## Integration needed

1. Install and validate the SO-101 MuJoCo/LeRobot simulator at https://github.com/Juxi-Technology/so101-mujoco using its documented pinned fork. Verify joint ordering, camera observations, recording, and a scripted grasp before training.
2. Collect compatible simulator demonstrations, train an ACT baseline on a cloud GPU, and preserve checkpoints and dataset metadata. Measure baseline performance on held-out conditions.
3. Implement observation-based success detection and object-region extraction, with a human-labeled fallback. Convert pixel regions to calibrated feasible grasps; pass measured widths and reachability to this module.
4. Integrate controlled retries, logging and memory updates only after verified outcomes. Compare baseline, retry-only, diagnosis without memory, and diagnosis with memory on matched trials.
5. Calibrate and validate separately on the real arm. A sim-trained policy is not assumed to transfer.

The pretrained ACT rollout and physical trials are pending.
