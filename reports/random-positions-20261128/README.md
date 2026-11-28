# Random-position cube evaluation, 28 November 2026

These are exploratory, matched MuJoCo trials of a fixed 20,000-step ACT checkpoint in the tutorial SO-101 scene. Both batches used seed 2000, 20 sampled starting positions, a 25-second simulated horizon and the same checkpoint (SHA-256 `974eda64112cb6505fa45d06bc9f74c9f848cc0db4416b8eab86a7021140a142`). The JSON files record identical start positions in the same order. Baseline was run first, followed by recovery; order was not randomized. The runs used `--no-capture`, so no camera/action trajectory archives exist for this batch. Outcome scores are simulator-derived and no physical-robot claim follows.

| Outcome | ACT alone | ACT + recovery |
| --- | ---: | ---: |
| Settled pick and place | 1/20 | 18/20 |
| Lifted 5 cm | 1/20 | 18/20 |
| Bilateral jaw contact observed | 4/20 | 20/20 |

Seventeen positions failed without recovery and succeeded with it; one succeeded in both and two failed in both. The recovery condition enabled `--recovery --recover-unknown-empty --recover-unknown-stationary --experimental-cone-ik --shadow-hold-verifier`; the hold verifier was shadow-only, and neither the grasp nor placement bandit selected actions. Therefore this is not evidence that online learning, memory, depth estimation, or the verifier caused improvement. The failures were a sampled collision rejection (episode 2) and no validated IK solution (episode 20). Contact and lift alone do not establish a stable grasp.

This is one cube, one scene, one checkpoint, a small sample, and not an external benchmark. It is not a comparison against another project's published score. The `retry.log` records a subsequent fixed-retry comparison intentionally stopped by the user after two completed episodes. It is incomplete and must not be presented as a 20-trial result. No per-step archives were requested in this batch. Paths in JSON refer to the original local environment.

Reproduce each mode serially in the local environment described in `docs/phase3-grasp-bandit-study.md`, using `scripts/live_act_eval.py --seed 2000 --episodes 20 --seconds 25 --no-capture`, first without recovery flags and then with the flags listed above. Record the full command, versions, action/camera traces, randomized order and independent held-out seeds for future studies. Results use the simulator's task-success check, not an external benchmark protocol.
