# Paired held-out evaluation

`scripts/evaluate_phase3_pairs.py MANIFEST MEMORY` analyzes a fixed set of paired Phase II and frozen Phase III cube trials. The inputs are a predeclared manifest JSON and the **exact memory file used in the frozen runs**. It refuses seeds appearing in that memory, missing/duplicated trials, interrupted trials, mismatched starts, checkpoint/horizon/scenario mismatches, and missing or inconsistent before/after memory SHA-256s. It does not collect episodes or establish that the supplied records and hashes are authentic. The runner now saves checkpoint and memory hashes and horizon in its summary, and includes recovery settings in the NPZ metadata. The manifest consistency check does not verify the scene revision, every controller flag, order, independent poses or source authenticity. The operator must record those details contemporaneously and keep evaluation records separate from development.

Required manifest structure (fill in real data, do not treat this as results):

```json
{
  "schema": "phase3-paired-evaluation-v1",
  "heldout_seeds": [2000],
  "checkpoint_sha256": "64 lowercase hexadecimal digits from model.safetensors",
  "horizon_s": 25,
  "trials": [
    {
      "condition": "phase2",
      "seed": 2000,
      "scenario": "cube",
      "checkpoint_sha256": "same actual SHA-256 as above",
      "horizon_s": 25,
      "start_xy": [0.20, 0.01],
      "task_success": false,
      "interrupted": false
    },
    {
      "condition": "phase3_frozen",
      "seed": 2000,
      "scenario": "cube",
      "checkpoint_sha256": "same actual SHA-256 as above",
      "horizon_s": 25,
      "start_xy": [0.20, 0.01],
      "task_success": true,
      "interrupted": false,
      "frozen": true,
      "memory_sha256_before": "64 lowercase hexadecimal digits from the actual memory",
      "memory_sha256_after": "same actual SHA-256 as before"
    }
  ]
}
```

The numbers and outcomes above are placeholders. Preserve all requested trials, including failures and interrupted trials; the script refuses interrupted trials rather than silently excluding them. Before collecting data, define how an interruption is handled and rerun according to that protocol. Use identical settings for both conditions apart from the Phase III bandit, and randomize the run order. `shasum -a 256 PATH` calculates file hashes on macOS. Keep memory unchanged and use `--bandit-frozen` for the Phase III condition.

The tool reports the four paired success counts, each condition's successes, the paired difference `(only_phase3 - only_phase2) / N`, and a two-sided exact McNemar p-value. The p-value tests the symmetry of discordant pairs under its assumptions; it does not measure the size or certainty of an improvement, and a small pilot cannot establish equivalence. Training-seed disjointness alone does not prove that positions are independent, and the existing model checkpoints, scene and perception may also be selected using development data. A valid evaluation requires a genuinely held-out selection protocol and unedited source captures.
